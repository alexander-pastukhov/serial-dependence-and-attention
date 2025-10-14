// Updated Bayesian Integration (ubi) model with Response weight (can take any value) and Evidence (numerosity) weight (can take any value) and accounts for central tendency
data {
  int <lower=1> DataN;
  int <lower=1> TaskN;
  int <lower=1> ParticipantsN;
  int <lower=1> Numerosity_max;
  
  array[DataN] int<lower=0, upper=Numerosity_max> Numerosity; // number of dots
  array[DataN] real<lower=0, upper=Numerosity_max> Response;
  array[DataN] int<lower=0, upper=1> IsFirstTrial;            // 1 - first trial, 0 - all other trials
  array[DataN] int<lower=1, upper=TaskN> Task;                // 1 - single, 2 - dual
  array[DataN] int<lower=1, upper=ParticipantsN> Participant;
}

transformed data {
  int ParamsN = 3;
  // 1) W_response_max (maximal weight for prior response), 
  // 2) W_numerosity_max (maximal weight for prior numerosity),
  // 3) W_ct 
  // 4) lambda (relevance of prior response based on how different it is)

  // optimization
  vector[DataN] LogNumerosity  = log(to_vector(Numerosity));
  vector[DataN] LogResponse  = log(to_vector(Response));
  
  // numerosity on 0..1 scale for beta distribution
  vector[DataN] Numerosity01 = to_vector(Numerosity) ./ Numerosity_max;
}

parameters {
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
    matrix[ParamsN * TaskN, ParticipantsN] params = rep_matrix(mu_params, ParticipantsN) + diag_pre_multiply(sigma_params, l_rho_params) * z_params;
    matrix[TaskN, ParticipantsN] W_response_max = block(params, 0 * TaskN + 1, 1, TaskN, ParticipantsN);
    matrix[TaskN, ParticipantsN] W_numerosity_max = block(params, 1 * TaskN + 1, 1, TaskN, ParticipantsN);
    matrix[TaskN, ParticipantsN] W_ct_max = block(params, 2 * TaskN + 1, 1, TaskN, ParticipantsN);

    real totalEvidence = 0;
    int iEvidence = 0;
    for(i in 1:DataN) {
      if (IsFirstTrial[i]) {
        mu[i] = Numerosity[i];
        totalEvidence = LogNumerosity[i];
        iEvidence = 1;
      } else {
        real avgEvidence = totalEvidence / iEvidence;
        real W_response = W_response_max[Task[i], Participant[i]];
        real W_numerosity = W_numerosity_max[Task[i], Participant[i]];
        real W_ct = W_ct_max[Task[i], Participant[i]];
        mu[i] = Numerosity[i] + 
                W_ct * (avgEvidence - Numerosity[i]) +
                W_response * (Response[i-1] - Numerosity[i]) + 
                W_numerosity *  (Numerosity[i-1] - Numerosity[i]);

        totalEvidence += LogNumerosity[i];
        iEvidence += 1;
      }
      
      sigma[i] = sigma_task[Task[i]];
    }
  }
}

model {
  mu_params[1:2] ~ normal(0, 0.5);  // W_response_max;  
  mu_params[3:4] ~ normal(0, 0.5);  // W_numerosity_max;  
  mu_params[4:5] ~ normal(0, 0.5);  // W_ct;  
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
