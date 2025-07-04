data {
  int <lower=1> N;
  int <lower=1> TaskN;
  int <lower=1> ParticipantsN;
  
  // // relative orientation as categorical
  // int <lower=1> RelOrisN;
  // array[RelOrisN] real RelOri;
  // array[N] int<lower=1, upper=RelOrisN> iRelOri;

  array[N] real Ori;
  array[N] real Response;
  array[N] real Lag1_dOri;
  array[N] real Lag1_dResponse;
  array[N] int<lower=1, upper=TaskN> Task;
  array[N] int<lower=1, upper=ParticipantsN> Participant;
  array[N] int<lower=0, upper=1> FirstTrial;
}

parameters {
  vector[3 * TaskN] mu_wrest_wori_lambda;
  cholesky_factor_corr[3 * TaskN] l_rho_wrest_wori_lambda;
  vector<lower=0>[3 * TaskN] sigma_wrest_wori_lambda;
  matrix[3 * TaskN, ParticipantsN] z_wrest_wori_lambda;

  real<lower=0> sigma;
}

transformed parameters {
  vector[N] mu;
  {
    matrix[3 * TaskN, ParticipantsN] wrest_wori_lambda = rep_matrix(mu_wrest_wori_lambda, ParticipantsN) + diag_pre_multiply(sigma_wrest_wori_lambda, l_rho_wrest_wori_lambda) * z_wrest_wori_lambda;
    matrix[TaskN, ParticipantsN] W_response_max = block(wrest_wori_lambda, 1, 1, TaskN, ParticipantsN);
    matrix[TaskN, ParticipantsN] W_ori_max = block(wrest_wori_lambda, TaskN + 1, 1, TaskN, ParticipantsN);
    matrix[TaskN, ParticipantsN] lambda = exp(block(wrest_wori_lambda, 2 * TaskN + 1, 1, TaskN, ParticipantsN));

    for(i in 1:N) {
      if (FirstTrial[i]) {
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
 
  mu_wrest_wori_lambda[1:4] ~ normal(0, 0.5);  // wprev
  mu_wrest_wori_lambda[5:6] ~ normal(0, 0.5);  // lambda
  l_rho_wrest_wori_lambda ~ lkj_corr_cholesky(3);
  sigma_wrest_wori_lambda ~ exponential(1);

  sigma ~ normal(0, 5);
}

generated quantities {
  vector[N] log_lik;
  for(i in 1:N) log_lik[i] = normal_lpdf(Response[i] | mu[i], sigma);
}
