data { 
  int<lower=0> N;         // Number of observations (countries)
  int<lower=0> y[N];      // Total number of wars for each country
  real<lower=0> n[N];     // Number of years for each country
}

parameters {
  real<lower=0> alpha_gamma;  // Shape parameter for Gamma distribution
  real<lower=0> beta_gamma;   // Rate parameter for Gamma distribution
  real<lower=0> theta[N];     // Expected number of wars per country-year
}

model {
  // Priors for alpha and beta
  alpha_gamma ~ normal(0, 10); 
  beta_gamma ~ normal(0, 10);
  
  // Hierarchical prior: Gamma distribution for theta
  for (i in 1:N) {
    theta[i] ~ gamma(alpha_gamma, beta_gamma);  // Prior for theta
  }
  
  // Poisson likelihood for wars
  for (i in 1:N) {
    y[i] ~ poisson(n[i] * theta[i]);  // Likelihood
  }
}

generated quantities {
  real y_sim[N];  // Simulated outcome variable
  
  // Generate posterior predictive samples
  for (i in 1:N) {
    y_sim[i] = poisson_rng(n[i] * theta[i]);  // Simulated wars
  }
}
