# =========================================================================== #
#                                                                             #
# File Name: AT2_prep_ADMIN.R                                                 #
# Description: Imports all the preprocessed data, combines, creates new       #
#              features and then outputs a train.rds and test.rds file for    #
#              ML.                                                            #
#                                                                             #
# =========================================================================== #
# == Clean ================================================================
rm(list=ls())

# == Libraries ================================================================
# Load packages
library(tidyverse)
library(lubridate)
library(forcats)
library(caret)
library(DataExplorer)
library(Amelia)
library(corrplot)
library(hydroGOF)
library(DMwR)
library(dplyr)
library(utils)
library(gbm)
library(parallel)
library(doParallel)
library(InformationValue)
library(vip)
library(caret)
library(xgboost)
library(foreach)

# == Import preprocessed data =================================================

# Webscraping dataa
scrape <- readRDS('scrape.rds')

# Raw testing and training data
test <-  readRDS('test_raw.rds')
train<- readRDS('train_raw.rds')

# = Cleaning ==========================================================

# IMDB webscraping general item data
scrape_item_general <- scrape %>% select(movie_id, rating_of_ten:length,
                                         imdb_staff_votes:top_1000_voters_average)

# Add item_imdb_ to column names to disambiguate
names(scrape_item_general)[c(-1,-6,-7)] <- paste0('imdb_', names(scrape_item_general)[c(-1,-6,-7)])
names(scrape_item_general)[-1] <- paste0('item_', names(scrape_item_general)[-1])

# User age bands (Train)
train$age_band <- factor(case_when(
  train$age < 18 ~ 'under_18',
  train$age >= 18 & train$age <= 29 ~ '18_to_29',
  train$age >= 30 & train$age <= 44 ~ '30_to_44',
  train$age >= 45 ~ '45_and_over'),
  levels = c('under_18', '18_to_29', '30_to_44', '45_and_over'),
  ordered = TRUE # Don't order as makes modelling annoying
)

# User age bands (Test)
test$age_band <- factor(case_when(
  test$age < 18 ~ 'under_18',
  test$age >= 18 & test$age <= 29 ~ '18_to_29',
  test$age >= 30 & test$age <= 44 ~ '30_to_44',
  test$age >= 45 ~ '45_and_over'),
  levels = c('under_18', '18_to_29', '30_to_44', '45_and_over'),
  ordered = TRUE # Don't order as makes modelling annoying
)

# == Joining data and aggregate feature engineering ===========================

# = General features ==========================================================

# Mean rating per item
item_mean_ratings_train <- train %>% 
  group_by(item_id) %>% 
  summarise(item_mean_rating = mean(rating))

# = Demographic-wide features =================================================

# - Gender ratings per item
user_gender_item_mean_ratings_train <- train %>% 
  group_by(gender, item_id) %>% 
  summarise(user_gender_item_mean_rating = mean(rating)) %>% 
  ungroup()

# - Age band ratings per item
user_age_band_item_mean_ratings_train <- train %>% 
  group_by(age_band, item_id) %>% 
  summarise(user_age_band_item_mean_rating = mean(rating)) %>% 
  ungroup()

# = User specific IMDB demographic-wide features ==============================

# - IMDB Gender ratings per item
scrape_gender_specific <- scrape %>% 
  select(movie_id, males_votes:females_average) %>% 
  gather(key = gender_score, value = value, -movie_id, na.rm = TRUE) %>% 
  separate(col = gender_score, into = c('gender', 'score_type')) %>% 
  spread(key = score_type, value = value) %>% 
  mutate(gender = fct_recode(gender, 'M' = 'males', 'F' = 'females'),
         gender = fct_rev(gender)) %>% # To get the factor levels right
  rename(item_id = movie_id,
         user_gender_item_imdb_mean_rating = average,
         user_gender_item_imdb_votes = votes)

# Check levels are correct
identical(levels(train$gender), levels(scrape_gender_specific$gender))

# - Age band ratings per item
scrape_age_specific <- scrape %>% 
  select(movie_id, starts_with('aged')) %>% 
  gather(key = age_score, value = value, -movie_id, na.rm = TRUE) %>% 
  separate(col = age_score, into = c('age_band', 'score_type'), sep = '_a|_v') %>% 
  spread(key = score_type, value = value) %>% 
  mutate(age_band = fct_recode(age_band,
                               'under_18' = 'aged_under_18',
                               '18_to_29' = 'aged_18-29',
                               '30_to_44' = 'aged_30_44',
                               '45_and_over' = 'aged_45'),
         age_band = factor(age_band,
                           levels = c('under_18', '18_to_29', '30_to_44', '45_and_over'),
                           ordered = TRUE)) %>% # To get the factor levels the same as ml_data 
  rename(item_id = movie_id,
         user_age_band_item_imdb_mean_rating = verage,
         user_age_band_item_imdb_votes = otes)

# Check levels are correct
identical(levels(train$age_band), levels(scrape_age_specific$age_band))

# - Gender-Age band ratings per item
scrape_gender_age_specific <- scrape %>% 
  select(movie_id, males_under_18_votes:females_under_18_average,
         males_18_29_votes:females_18_29_average,
         males_30_44_votes:females_30_44_average,
         males_45_votes:females_45_average) %>% 
  gather(key = gender_age_score, value = value, -movie_id, na.rm = TRUE) %>% 
  separate(col = gender_age_score, into = c('gender', 'age_score_type'),
           extra = 'merge') %>% 
  separate(col = age_score_type, into = c('age_band', 'score_type'), sep = '_a|_v') %>% 
  spread(key = score_type, value = value) %>% 
  mutate(gender = fct_recode(gender, 'M' = 'males', 'F' = 'females'),
         gender = fct_rev(gender),
         age_band = fct_recode(age_band,
                               'under_18' = 'under_18',
                               '18_to_29' = '18_29',
                               '30_to_44' = '30_44',
                               '45_and_over' = '45'),
         age_band = factor(age_band,
                           levels = c('under_18', '18_to_29', '30_to_44', '45_and_over'),
                           ordered = TRUE)) %>% # To get the factor levels the same as ml_data 
  rename(item_id = movie_id,
         user_gender_age_band_item_imdb_mean_rating = verage,
         user_gender_age_band_item_imdb_votes = otes)

# Check levels are correct
identical(levels(train$gender), levels(scrape_gender_age_specific$gender))
identical(levels(train$age_band), levels(scrape_gender_age_specific$age_band))

# Now remove the scrape dataset
rm(scrape)

# = Join everything together =================================================

# Joing Variables onto Train Data
train <- train %>% 
  
  # Train set specific joins
  left_join(item_mean_ratings_train, by = 'item_id') %>% 
  left_join(user_age_band_item_mean_ratings_train, by = c('age_band', 'item_id')) %>% 
  left_join(user_gender_item_mean_ratings_train, by = c('gender', 'item_id')) %>% 
  
  # Scrape General joins (External dataset)
  left_join(scrape_item_general, by = c('item_id' = 'movie_id')) %>%
  left_join(scrape_gender_specific, by = c('gender', 'item_id')) %>% 
  left_join(scrape_age_specific, by = c('age_band', 'item_id')) %>%
  left_join(scrape_gender_age_specific, by = c('gender', 'age_band', 'item_id')) %>% 
  
  # Scrape User_Gender_Age joins (External dataset)
  
  mutate(user_id = factor(user_id),      # Make user and item factors
         item_id = factor(item_id))

# Joing Variables onto Test Data
test <- test %>% 
  
  # Train set specific joins
  left_join(item_mean_ratings_train, by = 'item_id') %>% 
  left_join(user_age_band_item_mean_ratings_train, by = c('age_band', 'item_id')) %>% 
  left_join(user_gender_item_mean_ratings_train, by = c('gender', 'item_id')) %>% 
  
  # Scrape General joins (External dataset)
  left_join(scrape_item_general, by = c('item_id' = 'movie_id')) %>%
  left_join(scrape_gender_specific, by = c('gender', 'item_id')) %>% 
  left_join(scrape_age_specific, by = c('age_band', 'item_id')) %>%
  left_join(scrape_gender_age_specific, by = c('gender', 'age_band', 'item_id')) %>% 

  mutate(user_id = factor(user_id),      # Make user and item factors
         item_id = factor(item_id))

# Remove the variable sets (Train)
rm(item_mean_ratings_train, 
   user_gender_item_mean_ratings_train, user_age_band_item_mean_ratings_train)

# Remove the varibale sets (Scrape)
rm(scrape_age_specific, scrape_gender_age_specific, scrape_gender_specific, scrape_item_general)

# Have a look
glimpse(train)
glimpse(test)

# Check missing
# missing_train <- train %>%
#   gather(col, value) %>%
#   group_by(col) %>%
#   summarize(missing_share = mean(is.na(value)))
# 
# missing_test <- train %>%
#   gather(col, value) %>%
#   group_by(col) %>%
#   summarize(missing_share = mean(is.na(value)))

# Remove missing checks
# rm(missing_train, missing_test)

# = Save out =================================================

# Save train out
saveRDS(train, 'AT2_train_STUDENT.rds')

# Save test out
saveRDS(test, 'AT2_test_STUDENT.rds')


##############################################
######  Data Preparation  ######
##############################################

data_rating <- list(train, test)
plot_str(data_rating)
plot_str(data_rating, type = "r")

#################
## Train/ Test ###
#################

# train
summary(train)

train$unknown= as.numeric(train$unknown)
train$action= as.numeric(train$action)
train$adventure= as.numeric(train$adventure)
train$animation= as.numeric(train$animation)
train$childrens= as.numeric(train$childrens )
train$comedy= as.numeric(train$comedy )
train$crime= as.numeric(train$crime )
train$documentary= as.numeric(train$documentary )
train$drama= as.numeric(train$drama )
train$fantasy= as.numeric(train$fantasy )
train$film_noir= as.numeric(train$film_noir )
train$horror= as.numeric(train$horror )
train$musical= as.numeric(train$musical )
train$mystery= as.numeric(train$mystery )
train$romance= as.numeric(train$romance )
train$sci_fi= as.numeric(train$sci_fi )
train$thriller= as.numeric(train$thriller )
train$war= as.numeric(train$war )
train$western= as.numeric(train$western )


introduce(train)

# test
summary(test)

test$unknown= as.numeric(test$unknown)
test$action= as.numeric(test$action)
test$adventure= as.numeric(test$adventure)
test$animation= as.numeric(test$animation)
test$childrens= as.numeric(test$childrens )
test$comedy= as.numeric(test$comedy )
test$crime= as.numeric(test$crime )
test$documentary= as.numeric(test$documentary )
test$drama= as.numeric(test$drama )
test$fantasy= as.numeric(test$fantasy )
test$film_noir= as.numeric(test$film_noir )
test$horror= as.numeric(test$horror )
test$musical= as.numeric(test$musical )
test$mystery= as.numeric(test$mystery )
test$romance= as.numeric(test$romance )
test$sci_fi= as.numeric(test$sci_fi )
test$thriller= as.numeric(test$thriller )
test$war= as.numeric(test$war )
test$western= as.numeric(test$western )


introduce(train)


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

# Create Release_Year variable - train           
train$Release_Year <- train$release_date 
train$Release_Year  <- format(train$Release_Year,"%Y")
train$Release_Year  = as.integer(train$Release_Year)
str(train)

# Create Release_Year variable - test    
test$Release_Year <- test$release_date 
test$Release_Year  <- format(test$Release_Year,"%Y")
test$Release_Year  = as.integer(test$Release_Year)
str(test)

# Create movie Decade bands (train)
train$movie_decade <- factor(case_when(
  train$Release_Year >= 1922 & train$Release_Year <= 1929 ~ '1920s',
  train$Release_Year >= 1930 & train$Release_Year <= 1939 ~ '1930s',
  train$Release_Year >= 1940 & train$Release_Year <= 1949 ~ '1940s',
  train$Release_Year >= 1950 & train$Release_Year <= 1959 ~ '1950s',
  train$Release_Year >= 1960 & train$Release_Year <= 1969 ~ '1960s',
  train$Release_Year >= 1970 & train$Release_Year <= 1979 ~ '1970s',
  train$Release_Year >= 1980 & train$Release_Year <= 1989 ~ '1980s',
  train$Release_Year >= 1990 & train$Release_Year <= 1999 ~ '1990s'),
  levels = c('1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s'),
  ordered = TRUE 
)

# Create movie Decade bands (test)
test$movie_decade <- factor(case_when(
  test$Release_Year >= 1922 & test$Release_Year <= 1929 ~ '1920s',
  test$Release_Year >= 1930 & test$Release_Year <= 1939 ~ '1930s',
  test$Release_Year >= 1940 & test$Release_Year <= 1949 ~ '1940s',
  test$Release_Year >= 1950 & test$Release_Year <= 1959 ~ '1950s',
  test$Release_Year >= 1960 & test$Release_Year <= 1969 ~ '1960s',
  test$Release_Year >= 1970 & test$Release_Year <= 1979 ~ '1970s',
  test$Release_Year >= 1980 & test$Release_Year <= 1989 ~ '1980s',
  test$Release_Year >= 1990 & test$Release_Year <= 1999 ~ '1990s'),
  levels = c('1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s'),
  ordered = TRUE 
)

# user_item variable
train$user_item <- paste(train$user_id, train$item_id, sep="_")
test$user_item <- paste(test$user_id, test$item_id, sep="_")

###########################################
###########################################
####### Exploratory Data Analysis #######
###########################################
###########################################

# Finding missing values in the data set

# Train
plot_missing(train)
train <- drop_columns(train, "video_release_date")
# video relaease date is 100% missing so remove this (train <- drop_columns(train, "video_release_date")), release_date is 0.01% and imbd_url is 0.02%
plot_missing(train)
plot_intro(train)

# Test

# Remove Video Release Date from train and test (all N/A data)
plot_missing(test)
test <- drop_columns(test, "video_release_date")
# video relaease date is 100% missing so remove this (test <- drop_columns(test, "video_release_date")), release_date is 0.01% and imbd_url is 0.02%
plot_missing(test)
plot_intro(test)


# To visualize the table above (with some light analysis):
plot_bar(train)

# To visualize distributions for all continuous features:

plot_histogram(train)

# Quantile-Qunatile plots
qq_data <- train[, c("age", "item_id", "rating", "user_id")]
plot_qq(qq_data, sampled_rows = 1000L)



# Count of ratings
A <- ggplot(data = train) + geom_histogram(mapping = aes(x = rating), binwidth = 0.5) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Count of Ratings")
# print results
A

#  Count of ratings by age band and gender
E <-  train %>%
  ggplot(aes(x = gender, y = rating)) +
  geom_bar(stat = "identity", fill = "blue") +
  facet_wrap(~ age_band) +
  labs(title = "Total Number of Ratings: By Age Band & Gender",
       y = "Mean Rating",
       x = "Gender") 
# print results
E
# More ratings by men - need to consider how this could impact overall recommendations if gender is a strong driver of preference (if using user based collaborative filtering)

unknown <- train %>% filter(unknown == "TRUE")
action <- train %>% filter(action == "TRUE")
adventure <- train %>% filter(adventure == "TRUE")
animation <- train %>% filter(animation == "TRUE")
childrens <- train %>% filter(childrens == "TRUE")
comedy <- train %>% filter(comedy == "TRUE")
crime <- train %>% filter(crime == "TRUE")
documentary <- train %>% filter(documentary == "TRUE")
drama <- train %>% filter(drama == "TRUE")
fantasy <- train %>% filter(fantasy == "TRUE")
film_noir <- train %>% filter(film_noir == "TRUE")
horror <- train %>% filter(horror == "TRUE")
musical <- train %>% filter(musical == "TRUE")
mystery <- train %>% filter(mystery == "TRUE")
romance <- train %>% filter(romance == "TRUE")
sci_fi <- train %>% filter(sci_fi == "TRUE")
thriller <- train %>% filter(thriller == "TRUE")
war <- train %>% filter(war == "TRUE")
western <- train %>% filter(western == "TRUE")




#  Count of ratings by age band for action
D <-  train %>% filter(action == "TRUE") %>%
  ggplot(aes(x = action, y = item_mean_rating)) +
  geom_bar(stat = "identity", fill = "blue") +
  facet_wrap(~ age_band) +
  labs(title = "TBC",
       y = "Mean Rating",
       x = "Gender") 
# print results
D   # shows that ratingsby action (picked any genre as an example) by age band

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
ab_under_18 <- train %>% filter(age_band == "under_18")
ab_18_to_29 <- train %>% filter(age_band == "18_to_29")
ab_30_to_44 <- train %>% filter(age_band == "30_to_44")
ab_45_and_over <- train %>% filter(age_band == "45_and_over")

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
male <- train %>% filter(gender == "M")
female <- train %>% filter(gender == "F")

# Review distribution by gender
b1 <- ggplot(data = male) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Males")
# print results
b1

b2 <- ggplot(data = female) + geom_histogram(mapping = aes(x = item_mean_rating), binwidth = 0.1) + scale_x_continuous(breaks = c(1:10)) + labs (title = "Females")
# print results
b2

# Distribution of ratings by year of release
p1 <- ggplot(data = train, aes(x = Release_Year, y = item_mean_rating)) +
  geom_point(aes(color = gender)) +
  scale_y_continuous(labels = scales::comma) +
  labs (title = "Ratings by Year of Movie Release",
        x = "Release Year", y = "Ratings") 
# print results
p1

# Count of rating by occupation & gender
p2 <- train %>%  ggplot(aes(x = gender, y = rating)) +
  geom_bar(stat = "identity", fill = "blue") +
  facet_wrap(~ occupation) +
  labs(title = "Count of Ratings by Occupation",
       y = "Count of Ratings",
       x = "Gender") 
# print results
p2

p3 <- ggplot(data = train, aes(x = item_mean_rating, y = item_id)) +
  geom_point(aes(color = age_band)) +
  labs (title = "Total Transactions per Month",
        x = "Year", y = "Movie ID") 
# print results
p3

#############################
# treating NA
#############################
# train 7
m7<-median(train$rating, na.rm=TRUE)
train[,7][is.na(train[,7])]<-m7
which(is.na(train[,7]))

# test 7
m7<-median(test$rating, na.rm=TRUE)
train[,7][is.na(test[,7])]<-m7
which(is.na(test[,7]))

# train$item_mean_rating  
m31<-median(train$item_mean_rating, na.rm=TRUE)
train[,31][is.na(train[,31])]<-m31
which(is.na(train[,31]))
#
# test$item_mean_rating
mt31<-median(test$item_mean_rating, na.rm=TRUE)
test[,31][is.na(test[,31])]<-mt31
which(is.na(test[,31]))

# train$user_age_band_item_mean_rating
m32<-median(train$user_age_band_item_mean_rating, na.rm=TRUE)
train[,32][is.na(train[,32])]<-m32
which(is.na(train[,32]))
# test$user_age_band_item_mean_rating
mt32<-median(test$user_age_band_item_mean_rating, na.rm=TRUE)
test[,32][is.na(test[,32])]<-mt32
which(is.na(test[,32]))
# train$user_gender_item_mean_rating
m33<-median(train$user_gender_item_mean_rating, na.rm=TRUE)
train[,33][is.na(train[,33])]<-m33
which(is.na(train[,33]))
# test$user_gender_item_mean_rating


mt33<-median(test$user_gender_item_mean_rating, na.rm=TRUE)
test[,33][is.na(test[,33])]<-mt33
which(is.na(test[,33]))


#################################################
#################################################
#################  Modelling ###################
#################################################
#################################################

## Linear Regression Model
### Train and Test data split

trainset_size <- floor(0.80 * nrow(train)) 
trainset_size 

# First step is to set a random seed to ensurre we get the same result each time
# All random number generators use a seed 
set.seed(3521) 

# Get indices of observations to be assigned to training set...
# This is via randomly picking observations using the sample function

trainset_indices <- sample(seq_len(nrow(train)), size = trainset_size)
trainset_indices


# Assign observations to training and testing sets

trainset_train <- train[trainset_indices, ]
trainset_train
testset_train <- train[-trainset_indices, ]
testset_train

# Rowcounts to check
nrow(trainset_train)
nrow(testset_train)
nrow(train)

# trainset_train= select(trainset_train, - imdb_url)
# trainset_train= select(trainset_train, - movie_title)

# Linear models for different combinations of dependent variables
lm1 <- lm(formula = rating~ age, data = trainset_train) 
summary(lm1)

predict <- predict(lm1, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 1.128062


###################################

lm2 <- lm(formula = rating~ age + gender, data = trainset_train) 
summary(lm2)

predict <- predict(lm2, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 1.128145

##################################

lm3 <- lm(formula = rating~ age + gender+occupation , data = trainset_train) 
summary(lm3)

predict <- predict(lm3, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 1.119113
###################################

lm4 <- lm(formula = rating~ age + occupation , data = trainset_train) 
summary(lm4)

predict <- predict(lm4, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 1.119748

####################################

lm5 <- lm(formula = rating~ gender + occupation , data = trainset_train) 
summary(lm5)

predict <- predict(lm5, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 1.122133

###################################

lm6 <- lm(formula = rating~ age + gender+occupation + item_mean_rating , data = trainset_train) 
summary(lm6)

predict <- predict(lm6, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 0.9981087

###################################

lm6 <- lm(formula = rating~ age + gender+occupation + item_mean_rating + user_age_band_item_mean_rating+ 
            user_gender_item_mean_rating+ item_imdb_rating_of_ten+ item_imdb_count_ratings+ item_imdb_length+ 
            item_imdb_staff_votes+ item_imdb_staff_average+ user_gender_item_imdb_mean_rating+ 
            user_gender_item_imdb_votes+ user_age_band_item_imdb_votes+ user_age_band_item_imdb_mean_rating+ 
            user_gender_age_band_item_imdb_votes+ user_gender_age_band_item_imdb_mean_rating, 
          data = trainset_train) 
summary(lm6)

predict <- predict(lm6, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 0.9627582

###################################

lm7 <- lm(formula = rating~ age + gender+occupation + item_mean_rating + user_age_band_item_mean_rating+ 
            user_gender_item_mean_rating+ item_imdb_rating_of_ten+ item_imdb_count_ratings+ item_imdb_length+ 
            item_imdb_staff_votes+ item_imdb_staff_average+ user_gender_item_imdb_mean_rating+
            item_imdb_top_1000_voters_votes+ item_imdb_top_1000_voters_average+
            user_gender_item_imdb_votes+ user_age_band_item_imdb_votes+ user_age_band_item_imdb_mean_rating+ 
            user_gender_age_band_item_imdb_votes+ user_gender_age_band_item_imdb_mean_rating, 
          data = trainset_train) 
summary(lm7)

predict <- predict(lm7, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 0.9627508

###################################

lm8 <- lm(formula = rating~ age + gender+occupation + 
            item_imdb_rating_of_ten+ item_imdb_length+ 
            item_imdb_staff_votes+ item_imdb_staff_average+
            item_imdb_top_1000_voters_votes+ item_imdb_top_1000_voters_average+
            user_gender_item_imdb_votes+ user_age_band_item_imdb_votes+ 
            user_gender_age_band_item_imdb_votes, 
          data = trainset_train) 
summary(lm8)

predict <- predict(lm8, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 1.023266

###################################

lm9 <- lm(formula = rating~ age + gender+occupation + item_mean_rating + user_age_band_item_mean_rating+ 
            user_gender_item_mean_rating, 
          data = trainset_train) 
summary(lm9)

predict <- predict(lm9, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 0.9505557

######################################
lm10 <- lm(formula = rating~  item_mean_rating + user_age_band_item_mean_rating+ 
             user_gender_item_mean_rating, 
           data = trainset_train) 
summary(lm10)

predict <- predict(lm10, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 0.955716

###################################

lm11 <- lm(formula = rating~ age + gender+occupation + item_mean_rating + user_age_band_item_mean_rating+ 
             user_gender_item_mean_rating+ item_imdb_rating_of_ten+ item_imdb_count_ratings+ item_imdb_length+ 
             item_imdb_staff_votes+ item_imdb_staff_average+ user_gender_item_imdb_mean_rating+
             item_imdb_top_1000_voters_votes+ item_imdb_top_1000_voters_average+
             user_gender_item_imdb_votes+ user_age_band_item_imdb_votes+ user_age_band_item_imdb_mean_rating+ 
             user_gender_age_band_item_imdb_votes+ user_gender_age_band_item_imdb_mean_rating, 
           data = trainset_train) 
summary(lm11)

predict <- predict(lm11, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 0.9627508


###################################
lm12 <- lm(formula = rating~  item_mean_rating  + item_mean_rating+ item_imdb_top_1000_voters_average + user_age_band_item_mean_rating+ 
             user_gender_item_mean_rating+ 
             item_imdb_staff_average+
              user_age_band_item_imdb_mean_rating+ 
             user_gender_age_band_item_imdb_votes, 
           data = trainset_train)  
summary(lm12)

predict <- predict(lm12, newdata = testset_train, type="response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is 0.955716


#################################

# Prediction for Test Dataset


p1<-predict(lm9, newdata= test, type = "response")
test$rating <- p1



test$user_item <- paste(test$user_id, test$item_id, sep="_")
AT2_sample_upload<- test%>% select(rating,user_item)

# sum(is.na(test$occupation)) #count of missing values



write.table(AT2_sample_upload, file = "AT2_sample_upload.csv", sep = ",",row.names = FALSE)



#######################################################
#######################################################
#########   GBM #####################################
########################################################
#######################################################


# Create training and test sets. This process should be familiar by now
trainset_size <- floor(0.80 * nrow(train)) 
trainset_size 

# First step is to set a random seed to ensurre we get the same result each time
# All random number generators use a seed 
set.seed(3521) 

# Get indices of observations to be assigned to training set...
# This is via randomly picking observations using the sample function

trainset_indices <- sample(seq_len(nrow(train)), size = trainset_size)
trainset_indices



# Assign observations to training and testing sets

trainset_train <- train[trainset_indices, ]
trainset_train
testset_train <- train[-trainset_indices, ]
testset_train

# Rowcounts to check
nrow(trainset_train)
nrow(testset_train)
nrow(train)
#### Train the model ####

# Defining some parameters

gbm_depth = 5 #maximum nodes per tree
gbm_n.min = 15 #minimum number of observations in the trees terminal, important effect on overfitting
gbm_shrinkage=0.01 #learning rate
cores_num = 8 #number of cores
gbm_cv_folds=10 #number of cross-validation folds to perform
num_trees = 1500 # Number of iterations

start <- proc.time()

# fit initial model
gbm_1 = gbm(rating~ age + gender+occupation + item_mean_rating + user_age_band_item_mean_rating+ 
              user_gender_item_mean_rating,
            data=trainset_train,
            distribution='gaussian', #count outcomes
            n.trees=num_trees,
            interaction.depth= gbm_depth,
            n.minobsinnode = gbm_n.min, 
            shrinkage=gbm_shrinkage, 
            cv.folds=gbm_cv_folds,
            verbose = TRUE, #print the preliminary output
            n.cores = cores_num
)

end <- proc.time() - start
end_time <- as.numeric((paste(end[3])))
end_time

# find index for n trees with minimum CV error
min_MSE <- which.min(gbm_1$cv.error)

# get MSE and compute RMSE
sqrt(min(gbm_1$cv.error[min_MSE]))
## [1] 0.9432139

predict = predict(gbm_1, testset_train, n.trees = best_iter, type = "response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is  0.9435267
# caret::RMSE(predict, testset_train$rating)

#############################
gbm_2 = gbm(rating~occupation + user_age_band_item_mean_rating+ 
              user_gender_item_mean_rating,
            data=trainset_train,
            distribution='gaussian', #count outcomes
            n.trees=num_trees,
            interaction.depth= gbm_depth,
            n.minobsinnode = gbm_n.min, 
            shrinkage=gbm_shrinkage, 
            cv.folds=gbm_cv_folds,
            verbose = TRUE, #print the preliminary output
            n.cores = cores_num
)

end <- proc.time() - start
end_time <- as.numeric((paste(end[3])))
end_time

predict = predict(gbm_2, testset_train, n.trees = best_iter, type = "response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is  0.9555975
# caret::RMSE(predict, testset_train$rating)

# find index for n trees with minimum CV error
min_MSE <- which.min(gbm_2$cv.error)

# get MSE and compute RMSE
sqrt(min(gbm_2$cv.error))  #  0.960702

# get MSE and compute RMSE
sqrt(min(gbm_2$cv.error[min_MSE]))
## [1] 0.960702
###########################################
#############################
gbm_3 = gbm(rating~.- item_id - zip_code- timestamp -imdb_url-movie_title-release_date-user_item,
            data=trainset_train,
            distribution='gaussian', #count outcomes
            n.trees=num_trees,
            interaction.depth= gbm_depth,
            n.minobsinnode = gbm_n.min, 
            shrinkage=gbm_shrinkage, 
            cv.folds=gbm_cv_folds,
            verbose = TRUE, #print the preliminary output
            n.cores = cores_num
)

end <- proc.time() - start
end_time <- as.numeric((paste(end[3])))
end_time

predict = predict(gbm_3, testset_train, n.trees = best_iter, type = "response")
testset_train$predictions <- predict
rmse(testset_train$rating, predict)# value is  0.8884474
# caret::RMSE(predict, testset_train$rating)

# find index for n trees with minimum CV error
min_MSE <- which.min(gbm_3$cv.error)

# get MSE and compute RMSE
sqrt(min(gbm_3$cv.error))  #  [1] 0.8907176

# get MSE and compute RMSE
sqrt(min(gbm_3$cv.error[min_MSE]))
## [1] 0.8907176


# AT2_sample_upload<- AT2_sample_upload %>% 
# group_by(user_item)%>%
#  mutate(rating= ifelse(is.na(rating), mean(rating, na.rm = TRUE), rating))


