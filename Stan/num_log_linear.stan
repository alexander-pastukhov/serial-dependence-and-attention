// Mixture of linear and logarithmic components
data {
  int <lower=1> DataN; // number of data points
  
  int <lower=1> TaskN;
  int <lower=1> ParticipantsN;
  int <lower=1> Numerosity_max;  // upper range for numerosity judgement 
  
  array[DataN] int<lower=0, upper=Numerosity_max> Numerosity; // number of dots
  array[DataN] real<lower=0, upper=Numerosity_max> Response;
  array[DataN] int<lower=1, upper=TaskN> Task;                // 1 - single, 2 - dual
  array[DataN] int<lower=1, upper=ParticipantsN> Participant;
}

transformed data {
  int ParamsN = 4; // 1) S_response (response scaling), 2) W_log (log-component weight), 3) A (power law constant), 4) S_sigma (uncertainty scaling)
  
  vector[DataN] normalizedLogN = (Numerosity_max / log(Numerosity_max)) * log(to_vector(Numerosity));
}

parameters {
  vector[ParamsN * TaskN] mu_params; // 1) S_response (response scaling), 2) W_log (log-component weight), 3) A (power law constant), 4) S_sigma (uncertainty scaling)
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
    matrix[TaskN, ParticipantsN] W_log = inv_logit(block(params, 1 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] A = exp(block(params, 2 * TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] S_sigma = exp(block(params, 3 * TaskN + 1, 1, TaskN, ParticipantsN));

    for(i in 1:DataN) {
      mu[i] = S_response[Task[i], Participant[i]] * ((1 - W_log[Task[i], Participant[i]]) * Numerosity[i] + W_log[Task[i], Participant[i]] * normalizedLogN[i]);
      sigma[i] = S_sigma[Task[i], Participant[i]] * Numerosity[i] ^ A[Task[i], Participant[i]];
    }
  }
}

model {
  Response ~ normal(mu, sigma);

  // 4LENA: could you please check the priors and correct both mu and sigma?
  mu_params[1:2] ~ normal(logit(0.8), 0.5); // S_response
  mu_params[3:4] ~ normal(logit(0.5), 1); // W_log
  mu_params[5:6] ~ normal(0, 1);            // A
  mu_params[7:8] ~ normal(0, 1);            // S_sigma
  sigma_params ~ exponential(1);
  l_rho_params ~ lkj_corr_cholesky(2);
  to_vector(z_params) ~ normal(0, 1);
}

generated quantities {
  vector[DataN] log_lik;
  for(i in 1:DataN) log_lik[i] = normal_lpdf(Response[i] | mu[i], sigma[i]);
}
