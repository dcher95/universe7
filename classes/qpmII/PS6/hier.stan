data {
  int<lower=0> N;              // Number of observations
  int<lower=0> J;              // Number of subjects
  int<lower=1, upper=J> subject[N]; // Subject IDs
  vector[N] y;                 // Outcome variable
  vector[N] Surface2;          // Dummy variable for Surface=2
  vector[N] Vision2;           // Dummy variable for Vision=2
  vector[N] Vision3;           // Dummy variable for Vision=3
  vector[N] Surface2_Vision2;  // Interaction term: Surface2 × Vision2
  vector[N] Surface2_Vision3;  // Interaction term: Surface2 × Vision3
}

parameters {
  real beta_0;                 // Intercept
  real beta_Surface2;          // Effect of Surface=2
  real beta_Vision2;           // Effect of Vision=2
  real beta_Vision3;           // Effect of Vision=3
  real beta_Surface2_Vision2;  // Interaction: Surface2 × Vision2
  real beta_Surface2_Vision3;  // Interaction: Surface2 × Vision3
  real<lower=0> sigma;         // Residual SD
  real<lower=0> sigma_u;       // Subject SD
  vector[J] u;                 // Random effects for subjects
}

model {
  // Priors informed by lme4 are too restrictive --> use less restrictive priors
  beta_0 ~ normal(0, 10);
  beta_Surface2 ~ normal(0, 5);
  beta_Vision2 ~ normal(0, 5);
  beta_Vision3 ~ normal(0, 5);
  beta_Surface2_Vision2 ~ normal(0, 5);
  beta_Surface2_Vision3 ~ normal(0, 5);
  sigma ~ cauchy(0, 2);
  sigma_u ~ cauchy(0, 2);
  u ~ normal(0, sigma_u);

  // Likelihood
  y ~ normal(
    beta_0 +
    beta_Surface2 * Surface2 +
    beta_Vision2 * Vision2 +
    beta_Vision3 * Vision3 +
    beta_Surface2_Vision2 * Surface2_Vision2 +
    beta_Surface2_Vision3 * Surface2_Vision3 +
    u[subject],
    sigma
  );
}

generated quantities {
  vector[N] y_rep;

  // Posterior predictive samples
  for (n in 1:N) {
    y_rep[n] = normal_rng(
      beta_0 +
      beta_Surface2 * Surface2[n] +
      beta_Vision2 * Vision2[n] +
      beta_Vision3 * Vision3[n] +
      beta_Surface2_Vision2 * Surface2_Vision2[n] +
      beta_Surface2_Vision3 * Surface2_Vision3[n] +
      u[subject[n]],
      sigma
    );
  }
}
