
#### Below I have included just the code w/ comments. I'm not sure if the .stan file is interpreable in canvas, so I also included that in the end.
#### Otherwise, it is very similar to the .Rmd file.

library(haven)
library(dplyr)
library(sjPlot)
library(ggplot2)
library(estimatr)
library(rstan)

mechturk_data <- read.csv("MechanicalTurk.csv")
unsc_data <- read.csv("UNSC.csv")

### 1. First let’s use BMI as our outcome variable of interest. We are interested in understanding if master workers are different than other workers when it comes to BMI. Fit a standard (classical) linear regression model to this data. You can go ahead and just control for all of the covariates (except exercise).

#### a. Interpret the results. We are particularly interested in the results for the TurkMaster. Be sure to provide a correct interpretation for the p-values/stars. What conclusions do you draw about whether master workers are different when it comes to BMI?

# fit model
model <- lm(bmi ~ TurkMaster + agecat + race_cat + female + income_cat, data = mechturk_data)

# view summary stats
summary(model)

### 2. In problem 1, you made use of the Null Hypothesis Significance Testing framework. Answer the questions below focusing again on the TurkMaster variable.

#### d. The significance test here is calculated simply by taking the coefficient and dividing by the standard error to derive a t-statistic with n-p degrees of freedom. Assume that the standard errors are fixed and plot the “power curve” for different values of $\beta$. Is your test well powered? Why or why not?

# params
beta_hat <- 1.0635
SE <- 0.4391
alpha <- 0.05
t_critical <- qnorm(1 - alpha / 2)

# power for test-statistic found in glm. Is our test powerful?
power_test <- 1 - (pnorm(t_critical, mean = beta_hat / SE, sd = 1) - 
                     pnorm(-t_critical, mean = beta_hat / SE, sd = 1))

# power curve for different betas
beta_range <- seq(0, 3, by = 0.1)
power <- 1 - (pnorm(t_critical, mean = beta_range / SE, sd = 1) - 
                pnorm(-t_critical, mean = beta_range / SE, sd = 1))

# plot
plot(beta_range, power, type = "l", col = "blue", lwd = 2,
     xlab = "Actual Beta", ylab = "Power",
     main = "Power Curve for TurkMaster Variable")
abline(v = beta_hat, col = "green", lty = 2)  
legend("bottomright", legend = c("Power Curve", "Beta Hat"),
       col = c("blue",  "green"), lty = c(1, 2), lwd = c(2, 1))

cat("Power:", power_test)

### 3. Your co-author theorizes that the differences in BMI could be a result of differing lifestyles. Perhaps master workers are more (or less) likely to exercise.
#### a. Fit an appropriate generalized linear model using exercise as the dependent variable.

# remove NAs
mechturk_data_clean <- mechturk_data[!is.na(mechturk_data$exercise), ]

# logistic regression b/c exercise is binary variable
glm_model <- glm(exercise ~ TurkMaster + agecat + race_cat + female + income_cat, 
                 data = mechturk_data_clean, 
                 family = binomial(link = "logit"))

summary(glm_model)

# look @ predicted probs for graph
mechturk_data_clean$predicted_probs <- predict(glm_model, type = "response")

# plot
ggplot(mechturk_data_clean, aes(x = factor(TurkMaster), y = predicted_probs, color = factor(TurkMaster))) +
  geom_boxplot() +
  labs(x = "TurkMaster status (0 = non-master, 1 = master)", 
       y = "Predicted Probability of Exercising", 
       color = "TurkMaster status") +
  theme_minimal()

cat('Odds ratio of TurkMaster:', exp(glm_model$coefficients[2]))

## United Nations Security Council

### 4. Using the lm_robust function, fit a model that (a) includes fixed effects for each country and separate fixed effects for each year and (b) has clustered standard errors at the country level. The only other covariate you need is l_pctagreeus_unga.

# model w/ fixed effects on country & year + clustered SE 
fe_model <- lm_robust(DependentVariable ~ l_pctagreeus_unga + factor(wdicode) + factor(year),
                      data = unsc_data,
                      clusters = unsc_data$wdicode)

summary(fe_model)

#### 5. Now fit a Bayesian linear model with random intercepts (not slopes) for each country and year. Be sure to carry out appropriate diagnostics. Compare your results to those above. (Note: You may end up with more countries in this analysis than in problem 4. Countries that only appear in the dataset once may be dropped from the fixed effects model depending on how you implemented it.)

# some rows with NA to remove
unsc_data_clean <- unsc_data %>%
  filter(complete.cases(.))

stan_data <- list(
  N = nrow(unsc_data_clean),
  C = length(unique(unsc_data_clean$wdicode)),
  Y = length(unique(unsc_data_clean$year)),
  country = as.integer(factor(unsc_data_clean$wdicode)),
  year = as.integer(factor(unsc_data_clean$year)),
  l_pctagreeus_unga = unsc_data_clean$l_pctagreeus_unga,
  DependentVariable = unsc_data_clean$DependentVariable
)

# add some warmup and seed for reproducibility
fit <- stan(
  file = "RandomIntercepts.stan",
  data = stan_data,
  warmup = 500,
  iter = 2000,
  chains = 2,
  seed = 42
)

# view all outputs
print(fit, pars = c("alpha", "beta_l_pctagreeus_unga", "sigma", "sigma_country", "sigma_year"))
traceplot(fit, pars = c("alpha", "beta_l_pctagreeus_unga", "sigma", "sigma_country", "sigma_year"))
print(summary(fit)$summary)


######### RandomIntercepts.stan file ###########
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

