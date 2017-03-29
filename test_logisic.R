library(Rcpp)
library(RcppArmadillo)
library(numDeriv)

setwd("~/Box Sync/Studies/Cpp/RCpp/kalman/efficientkalman/")
sourceCpp("kalmanRCpp_notest.cpp")

k <- 100
p0 <- 0.1*k
r <- 0.2
deltaT <- 0.1

logistG <- function(x, k = 100, t = deltaT){
  return(c(x[1], k * x[2] * exp(x[1]*t) / (k + x[2] * (exp(x[1]*t) - 1))))
}

# Let's create some sample data:
# set.seed(12345)

obsVariance <- 25
nObs = 1000
nu <- rnorm(nObs, mean=0, sd=sqrt(obsVariance)) 

# Evolution error
Q <- diag(c(0, 0))
# Observation error
R <- obsVariance
# Prior
x <- c(r, p0)
Sigma <- diag(c(144, 25))

pop <- logistG(x, k = 100, t = (1:(nObs-1))*deltaT) + nu

y = c()
for(i in 1:nObs){
  # Observation
  xobs <- c(0, pop[i])
  y[i] <- t(as.matrix(c(0,1))) %*% xobs
}

result = extkalmanC(x0 = as.matrix(x), y = as.matrix(y), Sigma0 = Sigma, Q = Q, R = as.matrix(R), 
                    smooth = F, f = logistG, h = function(x){return(t(as.matrix(c(0,1))) %*% x)})

plot(1:nObs, y, type = "l", col = "black")
lines(1:nObs, result$xfilt[,2], type = "l", col = "orange")
# lines(1:nObs, result$xsmooth[,2], type = "l", col = "red")

# plot(1:nObs, rep(r,1000), type = "l", col = "black")
# lines(1:nObs, result$xfilt[,1], type = "l", col = "orange")
# lines(1:nObs, result$xsmooth[,1], type = "l", col = "red")


