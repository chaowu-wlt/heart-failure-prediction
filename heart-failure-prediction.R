##########################################################################################################
#
# Load required R libraries for this project  
#
##########################################################################################################

# automatically install the missing packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", dependencies=TRUE, repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(matrixStats)
library(caret)
library(e1071)
library(rpart)
library(kernlab)

# number of significant digits to print
options(digits=5)



##########################################################################################################
#
# Data extraction
# Download data file from GitHub, read the data in, and then remove the downloaded file 
#
##########################################################################################################

# data file URL on GitHub
githubUrl <- "https://raw.githubusercontent.com/chaowu-wlt/heart-failure-prediction/main/heart_failure_clinical_records_dataset.csv"

# generate a temporary file name
tmp_filename <- tempfile() 

# download the data file from the url, give it the temporary name
download.file(githubUrl, tmp_filename) 

# read the data in
heartFailures <- read_csv(tmp_filename) 
heartFailures <- as.data.frame(heartFailures)

# erase the downloaded file
file.remove(tmp_filename) 



##########################################################################################################
#
# Data exploration
#
##########################################################################################################

# look at some data rows in the dataset
head(heartFailures,10)

# number of samples in the dataset
dim(heartFailures)[1]

# number of features in the dataset, the last column is the outcome
dim(heartFailures)[2] -1

# proportion of patients died caused by heart failure in the dataset
mean(heartFailures$DEATH_EVENT == 1)

# check if there are more patients died by heart failure who smoke than the patients who don't smoke
finding2 <- heartFailures %>%
  group_by(smoking) %>%
  summarize(death = mean(DEATH_EVENT == 1)) %>%
  ungroup()
finding2

# check if there are more patients died by heart failure who have diabetes than the patients who don't have
finding3 <- heartFailures %>%
  group_by(diabetes) %>%
  summarize(death = mean(DEATH_EVENT == 1)) %>%
  ungroup()
finding3

# check if there are more patients died by heart failure who have high blood pressure than the patients who don't have
finding4 <- heartFailures %>%
  group_by(high_blood_pressure) %>%
  summarize(death = mean(DEATH_EVENT == 1)) %>%
  ungroup()
finding4

# proportion of death grouped by gender where patients are smoking
finding5 <- heartFailures %>% 
  filter(smoking == 1) %>%
  group_by(sex) %>%
  summarize(death = mean(DEATH_EVENT == 1)) %>%
  ungroup()
finding5

# proportion of death grouped by gender where patients have diabetes					
finding6 <- heartFailures %>% 
  filter(diabetes == 1) %>%
  group_by(sex) %>%
  summarize(death = mean(DEATH_EVENT == 1)) %>%
  ungroup()
finding6

# proportion of death grouped by gender where patients have high blood pressure
finding7 <- heartFailures %>% 
  filter(high_blood_pressure == 1) %>%
  group_by(sex) %>%
  summarize(death = mean(DEATH_EVENT == 1)) %>%
  ungroup()
finding7

# proportion of death grouped by gender where patients are smoking and have diabetes
finding8 <- heartFailures %>% 
  filter(diabetes == 1 & smoking == 1) %>%
  group_by(sex) %>%
  summarize(death = mean(DEATH_EVENT == 1)) %>%
  ungroup()
finding8

# proportion of death grouped by gender where patients are smoking and have high blood pressure
finding9 <- heartFailures %>% 
  filter(high_blood_pressure == 1 & smoking == 1) %>%
  group_by(sex) %>%
  summarize(death = mean(DEATH_EVENT == 1)) %>%
  ungroup()
finding9



##########################################################################################################
#
# Data wrangling
#
##########################################################################################################

# check for missing values in the whole datasets
sum(is.na(heartFailures))

# extract all the features' columns and convert into matrix for matrix operation
all_features <- as.matrix(heartFailures[, 1:(ncol(heartFailures)-1)])

# column that has the highest mean
max(colMeans(all_features))
names(which.max(colMeans(all_features)))

# column that has the lowest mean
min(colMeans(all_features))
names(which.min(colMeans(all_features)))

# apply matrix scaling or standardization, each feature will have zero mean and unit variance 
x_centered <- sweep(all_features, 2, colMeans(all_features))
x_scaled <- sweep(x_centered, 2, colSds(all_features), FUN="/")

# check the standard deviation of the first column after scaling
sd(x_scaled[, 1])

# check the mean of the first column after scaling
mean(x_scaled[, 1])

# compute the average distance between the first "survival" sample and other "survival" samples
d_samples <- dist(x_scaled)
dist_A <- as.matrix(d_samples)[1, heartFailures$DEATH_EVENT == 0]
mean(dist_A[2:length(dist_A)])

# compute the average distance between the first "survival" sample and "death" samples
dist_D <- as.matrix(d_samples)[1, heartFailures$DEATH_EVENT == 1]
mean(dist_D)



##########################################################################################################
# 
# Analyse what data partition ratio to be used:
# Split the data into training and test sets with different ratios, 
# Find the ratio that brings the highest accuracy on training,
# This ratio will then be used to create data partition in all the algorithms later.
#
##########################################################################################################

# convert to data frame for data partition
x_scaled <- as.data.frame(x_scaled)

# using R 3.6 or later
set.seed(1, sample.kind="Rounding")

# ratios for test set 
ratio_on_test_set <- seq(0.1, 0.5, 0.1)

# create data partition with each ratio, then generate accuracy on the training set					   
partition_results <- map_df (ratio_on_test_set, function (p) {
  
  test_index <- createDataPartition(y = heartFailures$DEATH_EVENT, times = 1, p = p, 
                                    list = FALSE)
  
  # split predictors columns and outcome colummn for training set
  # variable x for predictors and y for outcome
  train_x <- x_scaled[-test_index, ]
  train_y <- heartFailures$DEATH_EVENT[-test_index]
  
  # use logistic regression algorithm 
  # to make the code run a bit faster, use 10-fold cross-validation with 10% of the observations each
  control <- trainControl(method = "cv", number = 10, p = .9)
  train_y <- as.factor(train_y)
  train_glm <- train(train_x, train_y, method="glm", trControl = control)	
  
  # store ratio and its corresponding accuracy in a list
  list (Ratio_on_test_set = p, Accuracy_on_training = train_glm$results["Accuracy"][[1]])
  
})

# display the result table
partition_results

# find and display the ratio that gives the highest accuracy
partition_ratio <- ratio_on_test_set[which.max(partition_results$Accuracy_on_training)]
partition_ratio



##########################################################################################################
# 
# Data partition:
# To train the model and optimize the algorithm parameters without using our test set,
# From the analysis above, partition_ratio with 70/30 split gives the highest accuracy.
# Split the dataset into training set and test set with 70/30 ratio.
#
##########################################################################################################

# using R 3.6 or later
set.seed(3, sample.kind="Rounding")

# test set will be 30% of whole heartFailures set
# the rest will become the training set
test_index <- createDataPartition(y = heartFailures$DEATH_EVENT, times = 1, p = partition_ratio, 
                                  list = FALSE)

# separate predictor columns and the outcome column for training and test sets, to make the code look tidy.
# variable x for predictors and y for outcome
test_x <- x_scaled[test_index, ]
test_y <- heartFailures$DEATH_EVENT[test_index]
train_x <- x_scaled[-test_index, ]
train_y <- heartFailures$DEATH_EVENT[-test_index]

# proportion of samples are indicated as survival
mean(heartFailures$DEATH_EVENT == 0)

# check if the training set and test set have similar proportion that indicated as survival
mean(train_y == 0)
mean(test_y == 0)



##########################################################################################################
# 
# Method: Logistic regression
#
##########################################################################################################

# using R 3.6 or later
set.seed(5, sample.kind="Rounding")

train_glm <- train(train_x, as.factor(train_y), method="glm")
glm_preds <- predict(train_glm, test_x)

# Create a table to store accuracy and evaluation results
# FPR - False positive rate
# TPR - Sensitivity
evaluation_results <- tibble(Method = "Logistic regression",
                             Accuracy_on_training = train_glm$results["Accuracy"][[1]],
                             Accuracy_on_test = mean(glm_preds == as.factor(test_y)),
                             FPR = 1 - specificity(factor(glm_preds), as.factor(test_y)),
                             TPR = sensitivity(factor(glm_preds), as.factor(test_y)),
                             Recall = sensitivity(factor(glm_preds), as.factor(test_y)),
                             Precision = precision(factor(glm_preds), as.factor(test_y)),
                             F_1 = F_meas(factor(glm_preds), as.factor(test_y)))

evaluation_results

# plot the preditions and mark the incorrect values
preds_plot <- test_x %>% ggplot(aes(age, anaemia, color = as.factor(glm_preds), shape=ifelse(glm_preds == as.factor(test_y), "Correct","Incorrect"))) + 
  geom_jitter(size=3, width = 0.1, alpha = 0.8) + 
  ggtitle("Predict")
preds_plot

# show important variables in this model
varImp(train_glm)



##########################################################################################################
# 
# Method: KNN
#
##########################################################################################################

# using R 3.6 or later
set.seed(9, sample.kind="Rounding")

# tune the parameters
tuning <- data.frame(k = seq(1, 21, 3))
train_knn <- train(train_x, as.factor(train_y), method="knn", tuneGrid = tuning)

# print the value of k that maximizes accuracy 
train_knn$bestTune

knn_preds <- predict(train_knn, test_x)

# Create a table to store accuracy and evaluation results
# FPR - False positive rate
# TPR - Sensitivity
evaluation_results <- bind_rows(evaluation_results,
                                tibble(Method = "KNN",
                                       Accuracy_on_training = train_knn$results$Accuracy[which(train_knn$bestTune$k == train_knn$results$k)],
                                       Accuracy_on_test = mean(knn_preds == as.factor(test_y)),
                                       FPR = 1 - specificity(factor(knn_preds), as.factor(test_y)),
                                       TPR = sensitivity(factor(knn_preds), as.factor(test_y)),
                                       Recall = sensitivity(factor(knn_preds), as.factor(test_y)),
                                       Precision = precision(knn_preds, as.factor(test_y)),
                                       F_1 = F_meas(factor(glm_preds), as.factor(test_y))))

evaluation_results

# plot the preditions and mark the incorrect values
preds_plot <- test_x %>% ggplot(aes(age, anaemia, color = as.factor(knn_preds), shape=ifelse(knn_preds == as.factor(test_y), "Correct","Incorrect"))) + 
  geom_jitter(size=3, width = 0.1, alpha = 0.8) + 
  ggtitle("Predict")
preds_plot	

# show important variables in this model
varImp(train_knn)



##########################################################################################################
# 
# Method: LDA
#
##########################################################################################################

# using R 3.6 or later
set.seed(5, sample.kind="Rounding")

train_lda <- train(train_x, as.factor(train_y), method="lda")
lda_preds <- predict(train_lda, test_x)

# Create a table to store accuracy and evaluation results
# FPR - False positive rate
# TPR - Sensitivity
evaluation_results <- bind_rows(evaluation_results,
                                tibble(Method = "LDA",
                                       Accuracy_on_training = train_lda$results["Accuracy"][[1]],
                                       Accuracy_on_test = mean(lda_preds == as.factor(test_y)),
                                       FPR = 1 - specificity(factor(lda_preds), as.factor(test_y)),
                                       TPR = sensitivity(factor(lda_preds), as.factor(test_y)),
                                       Recall = sensitivity(factor(lda_preds), as.factor(test_y)),
                                       Precision = precision(factor(lda_preds), as.factor(test_y)),
                                       F_1 = F_meas(factor(glm_preds), as.factor(test_y))))

evaluation_results

# plot the preditions and mark the incorrect values
preds_plot <- test_x %>% ggplot(aes(age, anaemia, color = as.factor(lda_preds), shape=ifelse(lda_preds == as.factor(test_y), "Correct","Incorrect"))) + 
  geom_jitter(size=3, width = 0.1, alpha = 0.8) + 
  ggtitle("Predict")
preds_plot

# show important variables in this model
varImp(train_lda)



##########################################################################################################
# 
# Method: Decision tree
#
##########################################################################################################

# using R 3.6 or later
set.seed(13, sample.kind="Rounding")

# tune the parameters
tuning_rpart <- data.frame(cp = seq(0, 0.1, 0.001))
train_rpart <- train(train_x, as.factor(train_y), method="rpart", tuneGrid = tuning_rpart)

# print the value of cp that maximizes accuracy
train_rpart$bestTune

rpart_preds <- predict(train_rpart, test_x)

# Create a table to store accuracy and evaluation results
# FPR - False positive rate
# TPR - Sensitivity
evaluation_results <- bind_rows(evaluation_results,
                                tibble(Method = "Decision tree",
                                       Accuracy_on_training = train_rpart$results$Accuracy[which(train_rpart$bestTune$cp == train_rpart$results$cp)],
                                       Accuracy_on_test = mean(rpart_preds == as.factor(test_y)),
                                       FPR = 1 - specificity(factor(rpart_preds), as.factor(test_y)),
                                       TPR = sensitivity(factor(rpart_preds), as.factor(test_y)),
                                       Recall = sensitivity(factor(rpart_preds), as.factor(test_y)),
                                       Precision = precision(factor(rpart_preds), as.factor(test_y)),
                                       F_1 = F_meas(factor(glm_preds), as.factor(test_y))))

evaluation_results

# plot the preditions and mark the incorrect values
preds_plot <- test_x %>% ggplot(aes(age, anaemia, color = as.factor(rpart_preds), shape=ifelse(rpart_preds == as.factor(test_y), "Correct","Incorrect"))) + 
  geom_jitter(size=3, width = 0.1, alpha = 0.8) + 
  ggtitle("Predict")
preds_plot

# plot to see the shape of the tree
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)

# show important variables in this model
varImp(train_rpart)



##########################################################################################################
# 
# Method: Random forest
#
##########################################################################################################

# using R 3.6 or later
set.seed(25, sample.kind="Rounding")

# tune the parameters
tuning_rf <- data.frame(mtry = seq(3, 11, 2))
train_rf <- train(train_x, as.factor(train_y), method="rf", ntree=100, tuneGrid=tuning_rf, importance=TRUE)

# print the value of mtry that maximizes accuracy
train_rf$bestTune

rf_preds <- predict(train_rf, test_x)

# Create a table to store accuracy and evaluation results
# FPR - False positive rate
# TPR - Sensitivity
evaluation_results <- bind_rows(evaluation_results,
                                tibble(Method = "Random forest",
                                       Accuracy_on_training = train_rf$results$Accuracy[which(train_rf$bestTune$mtry == train_rf$results$mtry)],
                                       Accuracy_on_test = mean(rf_preds == as.factor(test_y)),
                                       FPR = 1 - specificity(factor(rf_preds), as.factor(test_y)),
                                       TPR = sensitivity(factor(rf_preds), as.factor(test_y)),
                                       Recall = sensitivity(factor(rf_preds), as.factor(test_y)),
                                       Precision = precision(factor(rf_preds), as.factor(test_y)),
                                       F_1 = F_meas(factor(glm_preds), as.factor(test_y))))

evaluation_results

# plot the preditions and mark the incorrect values
preds_plot <- test_x %>% ggplot(aes(age, anaemia, color = as.factor(rf_preds), shape=ifelse(rf_preds == as.factor(test_y), "Correct","Incorrect"))) + 
  geom_jitter(size=3, width = 0.1, alpha = 0.8) + 
  ggtitle("Predict")
preds_plot

# plot to see how many features being selected that gives the highest accuracy
ggplot(train_rf)

# show important variables in this model
varImp(train_rf)



##########################################################################################################
# 
# Method: SVM Linear
#
##########################################################################################################

# using R 3.6 or later
set.seed(33, sample.kind="Rounding")

train_svm <- train(train_x, as.factor(train_y), method="svmLinear")
svm_preds <- predict(train_svm, test_x)

# Create a table to store accuracy and evaluation results
# FPR - False positive rate
# TPR - Sensitivity
evaluation_results <- bind_rows(evaluation_results,
                                tibble(Method = "SVM Linear",
                                       Accuracy_on_training = train_svm$results["Accuracy"][[1]],
                                       Accuracy_on_test = mean(svm_preds == as.factor(test_y)),
                                       FPR = 1 - specificity(factor(svm_preds), as.factor(test_y)),
                                       TPR = sensitivity(factor(svm_preds), as.factor(test_y)),
                                       Recall = sensitivity(factor(svm_preds), as.factor(test_y)),
                                       Precision = precision(factor(svm_preds), as.factor(test_y)),
                                       F_1 = F_meas(factor(glm_preds), as.factor(test_y))))

evaluation_results

# plot the preditions and mark the incorrect values
preds_plot <- test_x %>% ggplot(aes(age, anaemia, color = as.factor(svm_preds), shape=ifelse(svm_preds == as.factor(test_y), "Correct","Incorrect"))) + 
  geom_jitter(size=3, width = 0.1, alpha = 0.8) + 
  ggtitle("Predict")
preds_plot

# show important variables in this model
varImp(train_svm)



##########################################################################################################
# 
# Method: Ensemble
#
##########################################################################################################

# using R 3.6 or later
set.seed(15, sample.kind="Rounding")

# binding predictions from all the methods we used above
ensemble <- cbind(glm = glm_preds == 0, knn = knn_preds == 0, lda = lda_preds == 0, rpart = rpart_preds == 0, rf = rf_preds == 0, svmLinear = svm_preds == 0)

# calculate the average of all training set accuracy
avg_accuracy <- mean(evaluation_results$Accuracy_on_training)

# only voting on the methods that are above the training average
# ensemble draw the prediction based on the majority of the votes
ind <- evaluation_results$Accuracy_on_training >= avg_accuracy
votes <- rowMeans (ensemble[, ind] == TRUE)
ensemble_preds <- ifelse(votes >= 0.5, 0, 1)

# Create a table to store accuracy and evaluation results
# FPR - False positive rate
# TPR - Sensitivity
evaluation_results <- bind_rows(evaluation_results,
                                tibble(Method = "Ensemble",
                                       Accuracy_on_training = 0.00,
                                       Accuracy_on_test = mean(ensemble_preds == test_y),
                                       FPR = 1 - specificity(factor(ensemble_preds), as.factor(test_y)),
                                       TPR = sensitivity(factor(ensemble_preds), as.factor(test_y)),
                                       Recall = sensitivity(factor(ensemble_preds), as.factor(test_y)),
                                       Precision = precision(factor(ensemble_preds), as.factor(test_y)),
                                       F_1 = F_meas(factor(glm_preds), as.factor(test_y))))

evaluation_results

# plot the preditions and mark the incorrect values
preds_plot <- test_x %>% ggplot(aes(age, anaemia, color = as.factor(ensemble_preds), shape=ifelse(ensemble_preds == as.factor(test_y), "Correct","Incorrect"))) + 
  geom_jitter(size=3, width = 0.1, alpha = 0.8) + 
  ggtitle("Predict")
preds_plot



##########################################################################################################
# 
# Compare accuracy on test set
#
##########################################################################################################

avg_accuracy_test <- mean(evaluation_results$Accuracy_on_test)
avg_accuracy_test

# show the methods that are equal or above the average accuracy on test
ind <- evaluation_results$Accuracy_on_test >= avg_accuracy_test
evaluation_results$Method[ind]



##########################################################################################################
# 
# Analyse features to find some insights and see what can be done in the future
#
##########################################################################################################

# table to show correlation between different features and 'DEATH_EVENT'
feature_cor <- tibble(Feature = c("\'time\'", "\'ejection_fraction\'", "\'serum_creatinine\'","\'smoking\'", "\'diabetes\'", "\'high_blood_pressure\'"), 
                      Correlation = c(cor(heartFailures$time, heartFailures$DEATH_EVENT), cor(heartFailures$ejection_fraction, heartFailures$DEATH_EVENT), cor(heartFailures$serum_creatinine, heartFailures$DEATH_EVENT), cor(heartFailures$smoking, heartFailures$DEATH_EVENT), cor(heartFailures$diabetes, heartFailures$DEATH_EVENT), cor(heartFailures$high_blood_pressure, heartFailures$DEATH_EVENT)))

feature_cor

# scatter plot ejection_fraction and serum_creatinine
heartFailures %>%
  ggplot(aes(ejection_fraction, serum_creatinine, col=as.factor(DEATH_EVENT))) + 
  geom_point() +
  stat_ellipse()


