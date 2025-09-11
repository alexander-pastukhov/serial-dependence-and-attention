// Bayesian integration model
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
  int ParamsN = 2; // 1) A (power law constant), 2) K (uncertainty scaling)
}

parameters {
  vector[ParamsN * TaskN] mu_params; // 1) A (power law constant), 2) K (uncertainty scaling)
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
    
    matrix[TaskN, ParticipantsN] A = exp(block(params, 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] K = exp(block(params, TaskN + 1, 1, TaskN, ParticipantsN));
    matrix[TaskN, ParticipantsN] K_sqr = K.^2;

    for(i in 1:DataN) {
      sigma[i] = sigma_task[Task[i]];
      
      if (IsFirstTrial[i]) {
        mu[i] = Ori[i];
      } else {
        real k2Na2 = K_sqr[Task[i], Participant[i]] * Ori[i]^(2 * A[Task[i], Participant[i]]);
        real W_response = k2Na2 / (k2Na2 + K_sqr[Task[i], Participant[i]] * Ori[i-1]^(2 * A[Task[i], Participant[i]]) + (Ori[i] - Response[i - 1])^2);
        mu[i] = (1 - W_response) * Ori[i] + W_response * Response[i-1];
      }
    }
  }
} 

model {
  Response ~ normal(mu, sigma);
  
  mu_params[1:2] ~ normal(log(0.25), log(2)); // A
  mu_params[3:4] ~ normal(log(4), log(2));    // K
  sigma_params ~ exponential(1);
  to_vector(z_params) ~ normal(0, 1);
  
  sigma_task ~ exponential(1);
}

generated quantities {
  vector[DataN] log_lik;
  for(i in 1:DataN) log_lik[i] = normal_lpdf(Response[i] | mu[i], sigma[i]);
}
