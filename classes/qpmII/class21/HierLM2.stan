data { 
  int<lower=0> N; // Length of dataset--should go first
  int<lower=0> J; // Number of countries
  vector[N] y; // outcome variable
  vector[N] X; // War Dummy predictor
  int<lower=0, upper=J> cc[N]; // Index relating country to observation
  vector[N] year; // The variable indicating year
}
parameters {
  real beta; // war dummy coefficient
  real<lower=0> sigma_y; // sd for residuals
  real<lower=0> sigma_alpha; // sd for random intercepts
  real<lower=0> sigma_beta; 
  real mu_alpha; // mean for random intercpets
  real mu_beta; 
  vector[J] alpha_j; // random intercepts
  vector[J] beta_j; // random slopes
}
transformed parameters{
  vector[N] mu;
  for (i in 1:N)
  mu[i]= mu_alpha+alpha_j[cc[i]]+ year[i]*beta_j[cc[i]] + X[i]*beta; // Mean function
}
model{
  beta ~ normal(0,1);  // priors
  mu_alpha~normal(0, 2);
  mu_alpha~normal(0, .1);
  sigma_alpha~cauchy(0, .5);
  sigma_y~normal(0, .2);
  sigma_beta~cauchy(0, .1);
  alpha_j~normal(0, sigma_alpha); // distribution of constants
  beta_j~normal(mu_beta, sigma_beta);//
  y ~ normal(mu, sigma_y); // likelihood
}
