// Updated Bayesian Integration model with prior Response weight, prior Evidence (orientation) weight and average Evidence weight (central tendency)
data {
  int <lower=1> DataN;
  int <lower=1> TaskN;
  int <lower=1> ParticipantsN;
  int <lower=1> Ori_max;
  int <lower=1> Ori_min;
  
  array[DataN] real Ori; // number of dots
  array[DataN] real Response;
  array[DataN] int<lower=0, upper=1> IsFirstTrial;            // 1 - first trial, 0 - all other trials
  array[DataN] int<lower=1, upper=TaskN> Task;                // 1 - single, 2 - dual
  array[DataN] int<lower=1, upper=ParticipantsN> Participant;
}

transformed data {
  int ScaleParamsN = 4; 
  // 1) U_max (point of maximal uncertainty on 0..1 range), 
  // 2) kappa (precision for uncertainty distribution), 
  // 3) a_sigma, 
  // 4) b_sigma (linear model for sigma from uncertainty)
  
  int ParamsN = 4;
  // 1) W_response_max (maximal weight for prior response), 
  // 2) W_ori_max (maximal weight for prior numerosity),
  // 3) W_ct_max (maximal weight for average numerosity), 
  // 4) lambda (relevance of prior response based on how different it is)

  // optimization
  //vector[DataN] LogOri  = log(to_vector(Ori));
  //vector[DataN] LogResponse  = log(to_vector(Response));
  
  real shift = 0.005;
  // orientation on 0..1 scale for beta distribution
  vector[DataN] Ori01 = (to_vector(Ori) - Ori_min)./(Ori_max - Ori_min);
  for(i in 1:DataN) {
    if (Ori01[i] == 0) {Ori01[i]= Ori01[i] + shift;}
    if (Ori01[i] == 1) {Ori01[i]= Ori01[i] - shift;}
  }
  //Rescale to 0..1 but without 0 and 1 (e.g. 0.005, 0.995)
}

parameters {
  vector[ScaleParamsN * TaskN] mu_scale_params;
  cholesky_factor_corr[ScaleParamsN * TaskN] l_rho_scale_params;
  vector<lower=0>[ScaleParamsN * TaskN] sigma_scale_params;
  matrix[ScaleParamsN * TaskN, ParticipantsN] z_scale_params;
  
  vector[ParamsN * TaskN] mu_params;
  cholesky_factor_corr[ParamsN * TaskN] l_rho_params;
  vector<lower=0>[ParamsN * TaskN] sigma_params;
  matrix[ParamsN * TaskN, ParticipantsN] z_params;
  
  vector<lower=0>[TaskN] sigma_task;
}

transformed parameters {
  vector[DataN] mu;
  vector[DataN] sigma;
  {
    matrix[ScaleParamsN * TaskN, ParticipantsN] scale_params = rep_matrix(mu_scale_params, ParticipantsN) + diag_pre_multiply(sigma_scale_params, l_rho_scale_params) * z_scale_params;
    matrix[TaskN, ParticipantsN] U_max = inv_logit(block(scale_params, 0 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] U_kappa = 1 ./ exp(block(scale_params, 1 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] sigma_a = exp(block(scale_params, 2 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] sigma_b = exp(block(scale_params, 3 * TaskN + 1, 1, TaskN, ParticipantsN));

    matrix[ParamsN * TaskN, ParticipantsN] params = rep_matrix(mu_params, ParticipantsN) + diag_pre_multiply(sigma_params, l_rho_params) * z_params;
    matrix[TaskN, ParticipantsN] W_response_max = block(params, 0 * TaskN + 1, 1, TaskN, ParticipantsN);
    matrix[TaskN, ParticipantsN] W_ori_max = block(params, 1 * TaskN + 1, 1, TaskN, ParticipantsN);
    matrix[TaskN, ParticipantsN] W_ct_max = block(params, 2 * TaskN + 1, 1, TaskN, ParticipantsN);
    matrix[TaskN, ParticipantsN] lambda = 2 * exp(block(params, 3 * TaskN + 1, 1, TaskN, ParticipantsN)).^2;

    // weight for uncertainty normalization (maximum probability density at the mode)
    matrix[TaskN, ParticipantsN] U_norm;
    for(iTask in 1:TaskN){
      for(iP in 1:ParticipantsN) {
        U_norm[iTask, iP] = exp(beta_proportion_lpdf(U_max[iTask, iP] | U_max[iTask, iP], U_kappa[iTask, iP]));
      }
    }
    
    real totalEvidence = 0;
    int iEvidence = 0;
    for(i in 1:DataN) {
      real uncertainty = exp(beta_proportion_lpdf(Ori01[i] | U_max[Task[i], Participant[i]], U_kappa[Task[i], Participant[i]])) / U_norm[Task[i], Participant[i]];

      if (IsFirstTrial[i]) {
        mu[i] = Ori[i];
        totalEvidence = Ori[i]; // logOri
        iEvidence = 1;
      } else {
        real avgEvidence = totalEvidence / iEvidence;
        real Relevance_response = exp(-((Ori[i] - Response[i-1])^2) / lambda[Task[i], Participant[i]]);
        real Relevance_ori = exp(-((Ori[i] - Ori[i-1])^2) / lambda[Task[i], Participant[i]]);
        real Relevance_avg = exp(-((Ori[i] - avgEvidence)^2) / lambda[Task[i], Participant[i]]);
        real W_response = W_response_max[Task[i], Participant[i]] * Relevance_response * uncertainty;
        real W_ori = W_ori_max[Task[i], Participant[i]] * Relevance_ori * uncertainty;
        real W_ct = W_ct_max[Task[i], Participant[i]] * Relevance_avg * uncertainty;
        mu[i] = Ori[i] + 
                W_ct * (avgEvidence - Ori[i]) +
                W_response * (Response[i-1] - Ori[i]) + 
                W_ori *  (Ori[i-1] - Ori[i]);

        totalEvidence += Ori[i];
        iEvidence += 1;
      }
      
      sigma[i] = sigma_a[Task[i], Participant[i]] + sigma_b[Task[i], Participant[i]] * uncertainty;
    }
  }
}

model {
  mu_scale_params[1:2] ~ normal(0, 1);      // U_max
  mu_scale_params[3:4] ~ normal(-2.5, 2);   // sigma, so that kappa = 1/sigma;
  mu_scale_params[5:6] ~ normal(0, 0.5);    // sigma_a; 
  mu_scale_params[7:8] ~ normal(-2.5, 2);   // sigma_b; 
  l_rho_scale_params ~ lkj_corr_cholesky(2);
  sigma_scale_params ~ exponential(1);
  to_vector(z_scale_params) ~ normal(0, 1);
  
  mu_params[1:2] ~ normal(0, 0.5);  // W_response_max;  
  mu_params[3:4] ~ normal(0, 0.5);  // W_ori_max;  
  mu_params[4:5] ~ normal(0, 0.5);  // W_ct;  
  mu_params[7:8] ~ normal(-0.5, 2); // lambda; 
  l_rho_params ~ lkj_corr_cholesky(2);
  sigma_params ~ exponential(1);
  to_vector(z_params) ~ normal(0, 1);

  sigma_task ~ exponential(1);

  Response ~ normal(mu, sigma);
}

generated quantities {
  vector[DataN] log_lik;
  for(i in 1:DataN) log_lik[i] = normal_lpdf(Response[i] | mu[i], sigma[i]);
}
