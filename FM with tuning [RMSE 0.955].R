# recosystem
# Matrix Factorization with GD
# MF is a popular technique to solve recommender system problem. The main idea is to approximate the matrix R(m x n) by the product of two matrixes of lower dimension: P (k x m) and Q (k x n).

# Matrix P represents latent factors of users. So, each k-elements column of matrix P represents each user. Each k-elements column of matrix Q represents each item. So, to find rating for item i by user u we simply need to compute two vectors: P[,u]’ x Q[,i]. Short and full description of package is available here.

# to clear environment
# rm(list=ls())

#########################
###### Load Libraries ###
#########################

library(tidyverse)
library(recosystem)
library(Matrix)

#########################
### Data Preparation ####
#########################

invisible(gc())

# Create Ratings df
ratings <- select(train, user_id, item_id, rating)

# First step is to set a random seed to ensurre we get the same result each time
# All random number generators use a seed 
set.seed(40)

in_train <- rep(TRUE, nrow(ratings))
in_train[sample(1:nrow(ratings), size = round(0.2 * length(unique(ratings$user_id)), 0) * 5)] <- FALSE

ratings_train <- ratings[(in_train)]
ratings_test <- ratings[(!in_train)]

write.table(ratings_train, file = "trainset.txt", sep = " ", row.names = FALSE, col.names = FALSE)
write.table(ratings_test, file = "testset.txt", sep = " ", row.names = FALSE, col.names = FALSE)

###############################
##### Build the model #########
###############################

# Next step is to build Recommender object:
r = Reco()

# read in
train_data <- data_file('trainset.txt', index1 = TRUE)
test_data <- data_file('testset.txt', index1 = TRUE)

# Now we could tune parameters of Recommender: number of latent factors (dim), gradient descend step rate (lrate), and penalty parameter to avoid overfitting (cost). To make this easier and fester to run, cost could be set to some small value and dim while tunning will adopt to it:
# documentation used  = (??recosystem)

# tune model, select best tuning parameters
# NOTE - Code ran for ~ 4 hours

opts2 <- r$tune(train_data, 
                opts = list(dim = c(1:20),
                            lrate = c(0.05),
                            nthread = 4,
                            costp_l1 = c(0, 0.1),
                            costp_l2 = c(0.01, 0.1),
                            costq_l1 = c(0, 0.1),
                            costq_l2 = c(0.01, 0.1),
                            niter = 200,
                            nfold = 10,
                            verbose = FALSE))
# print results
opts2

# save (opts2, file = 'opts2.RData')
attach('opts2.RData')

# And now model can be trained with the best tuned parameters:
r$train(train_data, opts = c(opts2$min, nthread = 1))

# iter      tr_rmse          obj
# 0       1.8467   2.0153e+05
# 1       1.0878   8.6693e+04
# 2       1.0026   7.7120e+04
# 3       0.9711   7.3499e+04
# 4       0.9551   7.1504e+04
# 5       0.9457   7.0214e+04
# 6       0.9398   6.9301e+04
# 7       0.9357   6.8616e+04
# 8       0.9324   6.8016e+04
# 9       0.9304   6.7588e+04
# 10       0.9283   6.7151e+04
# 11       0.9268   6.6767e+04
# 12       0.9259   6.6481e+04
# 13       0.9248   6.6193e+04
# 14       0.9240   6.5957e+04
# 15       0.9231   6.5691e+04
# 16       0.9225   6.5473e+04
# 17       0.9221   6.5295e+04
# 18       0.9216   6.5104e+04
# 19       0.9212   6.4939e+04

###################################
#### Write predictions to file ####
###################################

# predict
out_pred = out_file(tempfile())
r$predict(test_data, out_pred)

# calculating RMSE on test set & output predictions for whole test file
scores_real <- read.table('testset.txt', header = FALSE, sep = " ")$V3
scores_pred <- scan(out_pred@dest)
# Read 12884 items

rmse_mf <- sqrt(mean((scores_real-scores_pred) ^ 2))
rmse_mf
# [1] 0.9556666
