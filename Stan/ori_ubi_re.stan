// Updated Bayesian Integration (ubi) model with Response weight (can take any value) and Evidence (numerosity) weight (can take any value)
data {
  int <lower=1> DataN;
  int <lower=1> TaskN;
  int <lower=1> ParticipantsN;

  array[DataN] real Ori;
  array[DataN] real Response;
  array[DataN] int<lower=0, upper=1> IsFirstTrial;            // 1 - first trial, 0 - all other trials
  array[DataN] int<lower=1, upper=TaskN> Task;                // 1 - single, 2 - dual
  array[DataN] int<lower=1, upper=ParticipantsN> Participant;
}

transformed data {
  int ParamsN = 3; // 1) W_response_max, 2) lambda (relevance)
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
    matrix[TaskN, ParticipantsN] W_ori_max = block(params, 1 * TaskN + 1, 1, TaskN, ParticipantsN);
    matrix[TaskN, ParticipantsN] lambda = 2 * exp(block(params, 2 * TaskN + 1, 1, TaskN, ParticipantsN)).^2;

    for(i in 1:DataN) {
      sigma[i] = sigma_task[Task[i]];
      
      if (IsFirstTrial[i]) {
        mu[i] = Ori[i];
      } else {
        real relevance_reponse = exp(-((Ori[i] - Response[i-1])^2) / lambda[Task[i], Participant[i]]);
        real relevance_ori = exp(-((Ori[i] - Ori[i-1])^2) / lambda[Task[i], Participant[i]]);
        real W_response = W_response_max[Task[i], Participant[i]] * relevance_reponse;
        real W_ori = W_ori_max[Task[i], Participant[i]] * relevance_ori;
        mu[i] = Ori[i] + W_response * (Response[i-1] - Ori[i]) + W_ori * (Ori[i-1] - Ori[i]);
      }
    }
  }
}

model {
 Response ~ normal(mu, sigma);
 
  mu_params[1:4] ~ normal(0, 0.5);     // w_response_max (1:2), w_ori_max (3:4)
  mu_params[5:6] ~ normal(-0.5, 2);    // lambda
  l_rho_params ~ lkj_corr_cholesky(2);
  sigma_params ~ exponential(1);
  to_vector(z_params) ~ normal(0, 1);

  sigma_task ~ exponential(1);
}

generated quantities {
  vector[DataN] log_lik;
  for(i in 1:DataN) log_lik[i] = normal_lpdf(Response[i] | mu[i], sigma[i]);
}
