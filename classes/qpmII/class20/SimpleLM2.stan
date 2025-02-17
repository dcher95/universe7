data { 
  int<lower=0> N; // Length of dataset--should go first
  int<lower=0> K; // Number of parameters
  vector[N] y; // outcome variable
  matrix[N, K] X; // Matrix of predictors
}
parameters {
  vector[K] beta; // Now a vector
  real<lower=0> sigma;
}
transformed parameters{
  vector[N] mu;
  mu=X*beta; // Mean function
}
model {
  beta ~ normal(0,10); 
  sigma ~ cauchy(0,5); 
  y ~ normal(mu, sigma); 
}
