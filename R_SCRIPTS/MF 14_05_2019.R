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
opts = r$tune(train_data, opts = list(dim = c(1:20), lrate = c(0.05), costp_l1 = 0, costq_l1 = 0, nthread = 1))
# print results
opts

# save (opts, file = 'opts.RData')
attach('opts.RData')

# And now model can be trained with the best tuned parameters:
r$train(train_data, opts = c(opts$min, nthread = 1))

###################################
#### Write predictions to file ####
###################################

# predict
out_pred = out_file(tempfile())
r$predict(test_data, out_pred)

# calculating RMSE on test set & output predictions for whole test file
scores_real <- read.table('testset.txt', header = FALSE, sep = " ")$V3
scores_pred <- scan(out_pred@dest)

rmse_mf <- sqrt(mean((scores_real-scores_pred) ^ 2))
rmse_mf
