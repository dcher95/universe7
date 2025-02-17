
df <- read.csv("dataset-anes-2012-subset1.csv")

detach(df)
attach(df)

L1 <- glm(vote_obama ~ ft_dem + ft_rep + black + hispanic + other + income + sex,
          family=binomial(link=logit))


##### Q1 #####

length(df[(df$ft_dem==-9 | df$ft_dem==-8 | df$ft_dem==-2
           | df$ft_rep==-9 | df$ft_rep==-8 | df$ft_rep==-2),])

df<-df[!(df$ft_dem==-9 | df$ft_dem==-8 | df$ft_dem==-2
         | df$ft_rep==-9 | df$ft_rep==-8 | df$ft_rep==-2),]

# detach & attach the dataframe
detach(df)
attach(df)

L1 <- glm(vote_obama ~ ft_dem + ft_rep + black + hispanic + other + income + sex,
          family=binomial(link=logit))

summary(L1)


##### Q2 #####

# the best way to explore changes in predicted probabilities involving a continuous
#   independent variable is to produce a figure like that shown below, which shows how 
#   the predicted probability of voting for Obama changes as values of the Democratic 
#   Party feeling thermometer change from low to high, setting all of the other 
#   variables in the model constant at their means

# used for dem feeling thermometer
# by - increment of the sequence
therm <- seq(0, 100, by=1)

# N = 101
N <- length(therm)

# create a N*4 matrix, each element of which is NA
dt.probs <- matrix(NA, N, 4)

mean_ft_rep <- mean(ft_rep, na.rm=TRUE)
mean_black <- mean(black, na.rm=TRUE)
mean_hispanic <- mean(hispanic, na.rm=TRUE)
mean_other <- mean(other, na.rm=TRUE)
mean_income <- mean(income, na.rm=TRUE)
mean_sex <- mean(sex, na.rm=TRUE)

for(i in 1:N){
  # only change dem feeling thermometer and keep other variables constant at means
  profile.dt <- with(df, data.frame(ft_dem = therm[i],
                                    ft_rep = mean_ft_rep,
                                    black = mean_black,
                                    hispanic = mean_hispanic,
                                    other = mean_other,
                                    income = mean_income,
                                    sex = mean_sex))
  # predicted probability
  dt.prob<- as.numeric(predict(L1, newdata = profile.dt, type="response", 
                               se.fit=TRUE) [1])
  
  # predicted standard error
  dt.prob.se <- as.numeric(predict(L1, newdata = profile.dt, type="response", 
                                   se.fit=TRUE) [2])
  
  # predicted confidence interval
  dt.prob.ci <- c(dt.prob, dt.prob-1.96*dt.prob.se, dt.prob+1.96*dt.prob.se)
  
  # store predicted CI for a given dem feeling thermometer into 4 elements of a row
  dt.probs[i,] <- c(therm[i], dt.prob.ci)
}

# set the margin (bottom, left, top, right) of the plot
par(mar = c(4, 5, 1, 1))

plot(dt.probs[,1], dt.probs[, 2], type = "n", #xlim = c(0, 100),
     ylim = c(-.025, 1.025), xlab = "", ylab = "", axes = FALSE)

# lwd: line width; lty: line type (1 = solid, 2 = dashed)
# dt.probs[, 2]: mean predicted probability
lines(dt.probs[,1], dt.probs[,2], lwd = 4, lty = 1, col="red")

# dt.probs[, 3]: lower bound of predicted probability of CI
lines(dt.probs[,1], dt.probs[,3], lwd = 2, lty = 2, col="blue")

# dt.probs[, 4]: upper bound of predicted probability of CI
lines(dt.probs[,1], dt.probs[,4], lwd = 2, lty = 2, col="blue")

# line: distance to the plot from xlab/ylab 
title(ylab = expression("Predicted Prob. of Voting for Obama"), line = 3)
title(xlab = expression("Democratic Party Feeling Thermometer"), line = 2.75)

# las: labels are parallel (=0) or perpendicular(=2) to axis
axis(1, at = seq(0, 100, 10), las = 1)
axis(2, at = seq(0, 1, .1), las = 2)

# draws a box around the current plot 
box()

# rug plot with jitter (small amount of noise)
rug(jitter(ft_dem, factor=2), ticksize = .015)

legend("topleft", bty = "n",
       c( expression("Point Est."),
          expression("95% CI")),
       col=c("red" ,"blue"), lty = c(1, 2), lwd = c(4, 2), cex = 1)


L2 <- glm(vote_obama ~ ft_rep + race + income + sex,
          family=binomial(link=logit))


##### Q3 #####

#install.packages("see", dependencies = TRUE)
library(sjPlot)
library(ggplot2)
plot_model(L2, type = "pred", terms = c("income", "ft_rep", "race", "sex"))

##### Q3 #####

plot_model(L2, type = "pred", terms = c("income", "ft_rep [0, 25, 50, 75, 100]", "race [1, 4]", "sex [1]"))
