data { 
  int<lower=0> N; // Length of dataset--should go first
  int<lower=0> K; // Number of parameters
  vector[N] y; // outcome variable
  matrix[N, K] X; // Matrix of predictors
  vector[K] scale_beta; // Hyperpriors on variance terms for betas
  real loc_sigma; // Hyperprior on sigma
}
parameters {
  vector[K] beta; // Now a vector
  real<lower=0> sigma;
  real nu; // Degrees of freedom in the t 
}
transformed parameters{
  vector[N] mu;
  mu=X*beta; // Mean function
}
model {
  beta ~ normal(0.0,scale_beta); 
  sigma ~ exponential(loc_sigma); 
  nu ~ gamma(2, 0.1);
  
  //Likelihood
  y ~ student_t(nu, mu, sigma); 
}
generated quantities {
  real myBeta; // Set up my constructed quantities
    myBeta=2*beta[1]-beta[5];
}