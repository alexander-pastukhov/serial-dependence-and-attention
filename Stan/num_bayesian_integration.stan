// Bayesian integration model
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
  int ParamsN = 3; // 1) S_response (response scaling), 2) A (power law constant), 3) S_sigma (uncertainty scaling)
  
  // precomputed for optimization purposes
  vector[DataN] NR2 = rep_vector(0, DataN);
  for(i in 1:DataN){
    if (IsFirstTrial[i] == 0) NR2[i] = (Numerosity[i] - Response[i - 1])^2;
  }
}

parameters {
  vector[ParamsN * TaskN] mu_params; // 1) S_response (response scaling), 2) A (power law constant), 3) S_sigma (uncertainty scaling)
  cholesky_factor_corr[ParamsN * TaskN] l_rho_params;
  vector<lower=0>[ParamsN * TaskN] sigma_params;
  matrix[ParamsN * TaskN, ParticipantsN] z_params;
}

transformed parameters {
  vector[DataN] mu;
  vector[DataN] sigma;
  {
    matrix[ParamsN * TaskN, ParticipantsN] params = rep_matrix(mu_params, ParticipantsN) + diag_pre_multiply(sigma_params, l_rho_params) * z_params;
    matrix[TaskN, ParticipantsN] S_response = inv_logit(block(params, 0 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] A = exp(block(params, 1 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] S_sigma = exp(block(params, 2 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] S_sigma_sqr = S_sigma.^2;

    for(i in 1:DataN) {
      if (IsFirstTrial[i]) {
        mu[i] = S_response[Task[i], Participant[i]] * Numerosity[i];
      } else {
        real k2Na2 = Scale_sigma_sqr[Task[i], Participant[i]] * Numerosity[i]^(2 * A[Task[i], Participant[i]]);
        real W_response = k2Na2 / (k2Na2 + S_sigma_sqr[Task[i], Participant[i]] * Numerosity[i-1]^(2 * A[Task[i], Participant[i]]) + NR2[i]);
        mu[i] = S_response[Task[i], Participant[i]] * ((1 - W_response) * Numerosity[i] + W_response * Response[i-1]);
      }
      sigma[i] = S_sigma[Task[i], Participant[i]] * Numerosity[i] ^ A[Task[i], Participant[i]];
    }
  }
} 

model {
  Response ~ normal(mu, sigma);
  mu_params[1:2] ~ normal(logit(0.8), 0.5); // S_response
  mu_params[3:4] ~ normal(0, 1);            // A
  mu_params[5:6] ~ normal(0, 1);            // S_sigma
  sigma_params ~ exponential(1);
  l_rho_params ~ lkj_corr_cholesky(2);
  to_vector(z_params) ~ normal(0, 1);
}

generated quantities {
  vector[DataN] log_lik;
  for(i in 1:DataN) log_lik[i] = normal_lpdf(Response[i] | mu[i], sigma[i]);
}
