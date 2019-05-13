# UBCF v ICBF

# UBCF groups users according to prior usage behavior or according to their preferences, and then recommends an item that a similar user in the same group viewed or liked. UBCF mimics how word-of-mouth recommendations work in real life.

# User-Based Collaborative Filtering is used to generate a top-10 recommendation list for user 1 (User1 used for the purpose of this exercise)

####### Data Preprocessing ##########
# Create ratings matrix with  dcast() function in reshape2
library(recommenderlab)
library(reshape2)

# Create ratings matrix. Rows = userId, Columns = movieId
rating_matrix <- dcast(ratings, user_id~item_id, value.var = "rating", na.rm=FALSE)
rating_matrix <- as.matrix(rating_matrix[,-1]) #remove userIds

###########################
##### Predict Top 10 #####
###########################
# Method: UBCF
# Similarity Calculation Method: Cosine Similarity
# Nearest Neighbors: 30

# The predicted item ratings of the user will be derived from the 5 nearest neighbors in its neighborhood. When the predicted item ratings are obtained, the top 10 most highly predicted ratings will be returned as the recommendations.

# Convert rating matrix into a recommenderlab matrix
rating_matrix <- as(rating_matrix, "realRatingMatrix")

# Normalize the data
rating_mat_norm <- normalize(rating_matrix)

# Create Recommender Model
recommender_model <- Recommender(rating_mat_norm, method = "UBCF", param=list(method="Cosine",nn=30))

# Get top 10 recommendations for uer 1
recom <- predict(recommender_model, rating_matrix[1], n=10) 

# convert recommenderlab object to readable list
recom_list <- as(recom, "list") 

# Get recommendations
recom_result <- matrix(0,10)
for (i in c(1:10)){
  recom_result[i] <- movies[as.integer(recom_list[[1]][i]),2]
}
# print results  for user 1
recom_result

# [[1]]
# [1] "Seven (Se7en) (1995)"
# [[2]]
# [1] "Cop Land (1997)"
# [[3]]
# [1] "Liar Liar (1997)"
# [[4]]
# [1] "Ulee's Gold (1997)"
# [[5]]
# [1] "L.A. Confidential (1997)"
# [[6]]
# [1] "Lost Highway (1997)"
# [[7]]
# [1] "My Best Friend's Wedding (1997)"
# [[8]]
# [1] "E.T. the Extra-Terrestrial (1982)"
# [[9]]
# [1] "Fish Called Wanda, A (1988)"
# [[10]]
# [1] "Apostle, The (1997)"

# Evaluate Model
# k=5 fold cross validation, given-3 protocol
evaluation_scheme <- evaluationScheme(rating_matrix, method="cross-validation", k=5, given=3, goodRating=5) 

evaluation_results <- evaluate(evaluation_scheme, method="UBCF", n=c(1,3,5,10,15,20))
eval_results <- getConfusionMatrix(evaluation_results)[[1]]
# print results
eval_results
#           TP         FP       FN       TN precision     recall
# 1  0.1675393  0.8115183 17.51832 1660.503 0.1711230 0.01352183
# 3  0.5445026  2.3926702 17.14136 1658.921 0.1853832 0.04421352
# 5  0.8167539  4.0785340 16.86911 1657.236 0.1668449 0.06595930
# 10 1.3821990  8.4083770 16.30366 1652.906 0.1411765 0.10887985
# 15 1.8219895 12.8638743 15.86387 1648.450 0.1240642 0.13625481
# 20 2.1989529 17.3821990 15.48691 1643.932 0.1122995 0.16032331
# TPR          FPR
# 1  0.01352183 0.0004876708
# 3  0.04421352 0.0014376759
# 5  0.06595930 0.0024509791
# 10 0.10887985 0.0050544370
# 15 0.13625481 0.0077327501
# 20 0.16032331 0.0104503199

##########################################
##### Predict Ratings UBCF vs IBCF ######
#########################################

# We create two recommenders (user-based and item-based collaborative filtering) using the training data.

r1 <- Recommender(getData(evaluation_scheme, "train"), "UBCF")
r2 <- Recommender(getData(evaluation_scheme, "train"), "IBCF")

# Next, we compute predicted ratings for the known part of the test data (15 items for each user) using the two algorithms.
p1 <- predict(r1, getData(evaluation_scheme, "known"), type="ratings")
#print results
p1
# 191 x 1682 rating matrix of class ‘realRatingMatrix’ with 313973 ratings.

p2 <- predict(r2, getData(evaluation_scheme, "known"), type="ratings")
#print results
p2
# 191 x 1682 rating matrix of class ‘realRatingMatrix’ with 3218 ratings.

# Finally, we can calculate the error between the prediction and the unknown part of the test data.
error <- rbind(UBCF=calcPredictionAccuracy(p1,getData(evaluation_scheme,"unknown")), IBCF=calcPredictionAccuracy(p2,getData(evaluation_scheme,"unknown")))
# print results
error

#         RMSE      MSE       MAE
# UBCF 1.188339 1.412150 0.9133275
# IBCF 1.236265 1.528351 0.9432990

# In this example user-based collaborative filtering produces a smaller prediction error
