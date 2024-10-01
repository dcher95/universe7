# Load necessary package
library(ggplot2)

# Define prior parameters for two options
a1 <- 1; b1 <- 1   # Weak prior
a2 <- 5; b2 <- 100 # Moderately informative prior

# Create a sequence of lambda values
lambda_values <- seq(0, 0.01, length.out = 1000)

# Compute the prior density for both priors
prior1_density <- dgamma(lambda_values, a1, rate = b1)
prior2_density <- dgamma(lambda_values, a2, rate = b2)

# Plot the two prior distributions
prior_df <- data.frame(lambda = lambda_values, 
                       prior1 = prior1_density, 
                       prior2 = prior2_density)

ggplot(prior_df, aes(x = lambda)) +
  geom_line(aes(y = prior1, color = "Gamma(1, 1)"), size = 1) +
  geom_line(aes(y = prior2, color = "Gamma(5, 100)"), size = 1) +
  labs(title = "Prior Distributions", x = expression(lambda), y = "Density") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()

# Define posterior parameters for both priors
a1_post <- a1 + 50
b1_post <- b1 + 1000
a2_post <- a2 + 50
b2_post <- b2 + 1000

# Compute the posterior density for both cases
post1_density <- dgamma(lambda_values, a1_post, rate = b1_post)
post2_density <- dgamma(lambda_values, a2_post, rate = b2_post)

# Plot the two posterior distributions
posterior_df <- data.frame(lambda = lambda_values, 
                           post1 = post1_density, 
                           post2 = post2_density)

ggplot(posterior_df, aes(x = lambda)) +
  geom_line(aes(y = post1, color = "Posterior (Gamma(1, 1) Prior)"), size = 1) +
  geom_line(aes(y = post2, color = "Posterior (Gamma(5, 100) Prior)"), size = 1) +
  labs(title = "Posterior Distributions", x = expression(lambda), y = "Density") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()

# Compute the 95% HPD interval for both posteriors
hpd1 <- qgamma(c(0.025, 0.975), shape = a1_post, rate = b1_post)
hpd2 <- qgamma(c(0.025, 0.975), shape = a2_post, rate = b2_post)

hpd1
hpd2
