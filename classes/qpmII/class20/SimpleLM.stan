data { 
  int<lower=0> N; // Length of dataset--should go first
  vector[N] y; // outcome variable
  vector[N] x; // 
}
parameters {
  real alpha;
  real beta; 
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0,10); 
  beta ~ normal(0,10); 
  sigma ~ cauchy(0,5); 
  for (n in 1:N)
    y[n] ~ normal(alpha + beta * x[n], sigma); 
}
