# Load necessary package
library(ggplot2)

# Define the Beta prior parameters
alpha_prior <- 1
beta_prior <- 1

# Create a sequence of possible values for pi
pi_values <- seq(0, 1, length.out = 100)

# Compute the prior density for each pi value
prior_density <- dbeta(pi_values, alpha_prior, beta_prior)

# Plot the Beta prior
ggplot(data.frame(pi = pi_values, density = prior_density), aes(x = pi, y = density)) +
  geom_line(color = "blue", size = 1) +
  ggtitle("Beta(1, 1) Prior Distribution") +
  xlab(expression(pi)) +
  ylab("Density")

# Define the posterior Beta distribution parameters
alpha_post <- 558 + alpha_prior
beta_post <- 490 + beta_prior

# Create a sequence of possible values for pi
pi_values <- seq(0, 1, length.out = 100)

# Compute the posterior density for each pi value
posterior_density <- dbeta(pi_values, alpha_post, beta_post)

# Plot the Beta posterior
ggplot(data.frame(pi = pi_values, density = posterior_density), aes(x = pi, y = density)) +
  geom_line(color = "red", size = 1) +
  ggtitle("Beta(568, 500) Posterior Distribution") +
  xlab(expression(pi)) +
  ylab("Density")

credible_interval <- qbeta(c(0.025, 0.975), alpha_post, beta_post)
credible_interval
