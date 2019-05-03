# EDA AT2A 
# as at 3rd May 2019
###################### LOAD & CLEAN FILE  #########################

library(tidyverse)
library(dplyr)      
library(ggplot2)
library(modelr)
library(purrr)
library(broom)
library(ROCR)
library(pROC)
library(cvAUC)
library(AUC)
library(rpart)
library(rpart.plot)
library(mlbench)
library(caret)
library(randomForest)
library(gbm)
library(recommenderlab)
library(reshape2)

# Set working directory
setwd("~/R Projects/DAM/AT2A")

# Change data
train1 <- train
test1 <- test

# CLEANING DATA & ADDING / REMOVING VARIABLES

# Look at structure of train1
str(train1)

# Look at summary of train1
summary(train1)

# OBSERVATIONS
# More than double male to female (59k vs 20k)
# High proportion of students (second only to 'other' category)
# Zip code code have been interesting to look at but very large proportion of unknown data 'other' - will likely remove this variable later
# Unknown genre only has 10 entries that are unknown - consider removing? 
# Video Release date is all N/A data - REMOVE
# Mean Rating is 3.5, Median is 4
# Largest genre by no entries are Drama, Comedy and Action (in order of size largest first)
# Largest Age band is 18-29 Y/Olds followed by 30-44 Release_Year olds - could be some interesting insights of movie release date and age_band - however NOTE the age bands are unequal in size
# Item Mean Rating mean also 3.5 (same as rating) - Could be an indicator of how accurate a reccomendation is if user ratings for an item are similar to mean rating for an item
# Highest number of ratings by Mature rating is in R-rated movies - how would this need to be considered in a recommendation system? e.g. don't want R rated movies being recommended to someone who watches childrens movies a lot
# Decades - Min Release_Year 1922, Max Release_Year 1998 [9 N/A's]
# Movie Title and User ID will probably be needed in output but not as predictor variables?

# Create Release_Year variable - train1           
train1$Release_Year <- train1$release_date 
train1$Release_Year  <- format(train1$Release_Year,"%Y")
train1$Release_Year  = as.integer(train1$Release_Year)
str(train1)

# Create Release_Year variable - test1    
test1$Release_Year <- test1$release_date 
test1$Release_Year  <- format(test1$Release_Year,"%Y")
test1$Release_Year  = as.integer(test1$Release_Year)
str(test1)

# Create movie Decade bands (train1)
train1$movie_decade <- factor(case_when(
  train1$Release_Year >= 1922 & train1$Release_Year <= 1929 ~ '1920s',
  train1$Release_Year >= 1930 & train1$Release_Year <= 1939 ~ '1930s',
  train1$Release_Year >= 1940 & train1$Release_Year <= 1949 ~ '1940s',
  train1$Release_Year >= 1950 & train1$Release_Year <= 1959 ~ '1950s',
  train1$Release_Year >= 1960 & train1$Release_Year <= 1969 ~ '1960s',
  train1$Release_Year >= 1970 & train1$Release_Year <= 1979 ~ '1970s',
  train1$Release_Year >= 1980 & train1$Release_Year <= 1989 ~ '1980s',
  train1$Release_Year >= 1990 & train1$Release_Year <= 1999 ~ '1990s'),
  levels = c('1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s'),
  ordered = TRUE 
)

# Create movie Decade bands (test1)
test1$movie_decade <- factor(case_when(
  test1$Release_Year >= 1922 & test1$Release_Year <= 1929 ~ '1920s',
  test1$Release_Year >= 1930 & test1$Release_Year <= 1939 ~ '1930s',
  test1$Release_Year >= 1940 & test1$Release_Year <= 1949 ~ '1940s',
  test1$Release_Year >= 1950 & test1$Release_Year <= 1959 ~ '1950s',
  test1$Release_Year >= 1960 & test1$Release_Year <= 1969 ~ '1960s',
  test1$Release_Year >= 1970 & test1$Release_Year <= 1979 ~ '1970s',
  test1$Release_Year >= 1980 & test1$Release_Year <= 1989 ~ '1980s',
  test1$Release_Year >= 1990 & test1$Release_Year <= 1999 ~ '1990s'),
  levels = c('1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s'),
  ordered = TRUE 
)

# Remove Video Release Date from train and test (all N/A data)
train1 <- train1[, -11]
test1 <- test1[, -10]


# ---------------------------- EDA ---------------------------------------

# Count of ratings
A <- ggplot(data = train1) + geom_histogram(mapping = aes(x = rating), binwidth = 0.5) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Count of Ratings")
# print results
A

#  Count of ratings by age band and gender
E <-  train1 %>%
  ggplot(aes(x = gender, y = rating)) +
  geom_bar(stat = "identity", fill = "blue") +
  facet_wrap(~ age_band) +
  labs(title = "Total Number of Ratings: By Age Band & Gender",
       y = "Mean Rating",
       x = "Gender") 
# print results
E
# More ratings by men - need to consider how this could impact overall recommendations if gender is a strong driver of preference (if using user based collaborative filtering)

unknown <- train1 %>% filter(unknown == "TRUE")
action <- train1 %>% filter(action == "TRUE")
adventure <- train1 %>% filter(adventure == "TRUE")
animation <- train1 %>% filter(animation == "TRUE")
childrens <- train1 %>% filter(childrens == "TRUE")
comedy <- train1 %>% filter(comedy == "TRUE")
crime <- train1 %>% filter(crime == "TRUE")
documentary <- train1 %>% filter(documentary == "TRUE")
drama <- train1 %>% filter(drama == "TRUE")
fantasy <- train1 %>% filter(fantasy == "TRUE")
film_noir <- train1 %>% filter(film_noir == "TRUE")
horror <- train1 %>% filter(horror == "TRUE")
musical <- train1 %>% filter(musical == "TRUE")
mystery <- train1 %>% filter(mystery == "TRUE")
romance <- train1 %>% filter(romance == "TRUE")
sci_fi <- train1 %>% filter(sci_fi == "TRUE")
thriller <- train1 %>% filter(thriller == "TRUE")
war <- train1 %>% filter(war == "TRUE")
western <- train1 %>% filter(western == "TRUE")

#  Count of ratings by age band and gender
D <-  train1 %>% filter(action == "TRUE") %>%
  ggplot(aes(x = action, y = item_mean_rating)) +
  geom_bar(stat = "identity", fill = "blue") +
  facet_wrap(~ age_band) +
  labs(title = "TBC",
       y = "Mean Rating",
       x = "Gender") 
# print results
D

# Count of rating by action (picked any genre as an example) by age band and gender
f <- action %>%  ggplot(aes(x = gender, y = rating)) +
  geom_bar(stat = "identity", fill = "blue") +
  facet_wrap(~ age_band) +
  labs(title = "Genre = Action: Count of Ratings",
       y = "Count of Ratings",
       x = "Gender") 
# print results
f
# Doesn't show us much - we need to look at the spread of ratings by genre, not just the count of entries

# unknown
h <- ggplot(data = unknown) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Unknown")

# action
g <- ggplot(data = action) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Action")

# adventure
i <- ggplot(data = adventure) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Adventure")

# animation
j <- ggplot(data = animation) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Animation")

# childrens
k <- ggplot(data = childrens) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Childrens")

# comedy
l <- ggplot(data = comedy) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Comedy")

# crime
m <- ggplot(data = crime) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Crime")

# documentary
n <- ggplot(data = documentary) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Documentary")

# drama
o <- ggplot(data = drama) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Drama")

# fantasy
p <- ggplot(data = fantasy) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Fantasy")

# film_noir
q <- ggplot(data = film_noir) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Film Noir")

# horror
r <- ggplot(data = horror) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Horror")

# musical
s <- ggplot(data = musical) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Musical")

# mystery
t <- ggplot(data = mystery) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Mystery")

# romance
u <- ggplot(data = romance) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Romance")

# sci_fi
v <- ggplot(data = sci_fi) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Sci-fi")

# thriller
w <- ggplot(data = thriller) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Thriller")

# war
x <- ggplot(data = war) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "War")

# western
y <- ggplot(data = western) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Western")

# Arrange the histograms side by side [PLOT A]
plot_A <- grid.arrange(g,h,i,j, nrow=1)
plot_B <- grid.arrange(k,l,m,n,o, nrow=1)
plot_C <- grid.arrange(p,q,r,s,t, nrow=1)
plot_D <- grid.arrange(u,v,w,x,y, nrow=1)

# Create objects for each age_band
ab_under_18 <- train1 %>% filter(age_band == "under_18")
ab_18_to_29 <- train1 %>% filter(age_band == "18_to_29")
ab_30_to_44 <- train1 %>% filter(age_band == "30_to_44")
ab_45_and_over <- train1 %>% filter(age_band == "45_and_over")

# Review distribution by age band
a1 <- ggplot(data = ab_under_18) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Age Band - Under 18")
# print results
a1

a2 <- ggplot(data = ab_18_to_29) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Age Band - 18-29")
# print results
a2

a3 <- ggplot(data = ab_30_to_44) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Age Band - 30-44")
# print results
a3

a4 <- ggplot(data = ab_45_and_over) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Age Band - 45+")
# print results
a4

# Create objects for each age_band
male <- train1 %>% filter(gender == "M")
female <- train1 %>% filter(gender == "F")

# Review distribution by gender
b1 <- ggplot(data = male) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Males")
# print results
b1

b2 <- ggplot(data = female) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Females")
# print results
b2

# Distribution of ratings by year of release
p1 <- ggplot(data = train1, aes(x = Release_Year, y = item_mean_rating)) +
  geom_point(aes(color = gender)) +
  scale_y_continuous(labels = scales::comma) +
  labs (title = "Ratings by Year of Movie Release",
  x = "Release Year", y = "Ratings") 
# print results
p1

# Count of rating by occupation & gender
p2 <- train1 %>%  ggplot(aes(x = gender, y = rating)) +
  geom_bar(stat = "identity", fill = "blue") +
  facet_wrap(~ occupation) +
  labs(title = "Count of Ratings by Occupation",
       y = "Count of Ratings",
       x = "Gender") 
# print results
p2

p3 <- ggplot(data = train1, aes(x = item_mean_rating, y = item_id)) +
  geom_point(aes(color = age_band)) +
  labs (title = "Total Transactions per Month",
        x = "Year", y = "Movie ID") 
# print results
p3

########################
########################
########################

