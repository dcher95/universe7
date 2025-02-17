data {
  int<lower=1> N; // num rows
  int<lower=1> C; // num countries
  int<lower=1> Y; // num years
  int<lower=1, upper=C> country[N]; // country
  int<lower=1, upper=Y> year[N]; // year
  vector[N] l_pctagreeus_unga; // lagged percent ideological agreement with the USA
  vector[N] DependentVariable;  // Percent of the time you vote against the US in the UNSC
}

parameters {
  real alpha; // intercept
  vector[C] alpha_country; // countries -- random intercepts 
  vector[Y] alpha_year; // years -- random intercepts
  real beta_l_pctagreeus_unga; // beta for l_pctagreeus_unga
  real<lower=0> sigma;  // error sd
  real<lower=0> sigma_country; // countries -- random intercepts standard deviation
  real<lower=0> sigma_year; // years -- random intercepts standard deviation
}

model {
  // weak priors
  alpha ~ normal(0, 10);
  alpha_country ~ normal(0, sigma_country);
  alpha_year ~ normal(0, sigma_year);
  beta_l_pctagreeus_unga ~ normal(0, 10);
  
  // using log-normal because standard deviations are non-negative. Still pretty weak.
  
  sigma ~ normal(0, 1); 
  sigma_country ~ lognormal(0, 1);
  sigma_year ~ lognormal(0, 1);

  // likelihood
  DependentVariable ~ normal(alpha + alpha_country[country] + alpha_year[year] 
                              + beta_l_pctagreeus_unga .* l_pctagreeus_unga, sigma);
}
