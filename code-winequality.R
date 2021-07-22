# prepare libraries

if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")

library(dplyr)
library(ggplot2)
library(randomForest)
library(class)
library(caret)
library(corrplot)

# load data

wine <- read.csv("https://raw.githubusercontent.com/ryanreid-code/capstone/main/winequality-red.csv")

head(wine)
summary(wine)

# Split into test and train sets

set.seed(123, sample.kind ="Rounding")

test_index <- createDataPartition(y = wine$quality, times = 1, p = 0.2, list = FALSE)
train_wines <- wine[-test_index,]
test_wines <- wine[test_index,]

# The dataset is relatively small (1,599 wines), so the following code shouldn't take long to run.

# KNN using 6 key features

set.seed(123, sample.kind ="Rounding")
train_x <- subset(train_wines, select= c(alcohol, sulphates, total.sulfur.dioxide, volatile.acidity, density, citric.acid))
test_x <- subset(test_wines, select= c(alcohol, sulphates, total.sulfur.dioxide, volatile.acidity, density, citric.acid))
train_y <- train_wines$quality
test_y <- test_wines$quality

knn3_pred <- knn(train_x, test_x, train_y, k=3)
knn3_cm <- table(knn3_pred, test_y)
knn3_cm

knn1_pred <- knn(train_x, test_x, train_y, k=1)
knn1_cm <- table(knn1_pred, test_y)
knn1_cm

knn3_cmstat <- confusionMatrix(knn3_cm)
knn1_cmstat <- confusionMatrix(knn1_cm)

knn3_cmstat
knn1_cmstat

# KNN with 6 features is around 60% accurate at predicting the correct wine quality score out of 10 (KNN1 60%, KNN3 55%)

# KNN with 8 features. Added residual sugar and pH

set.seed(123, sample.kind ="Rounding")
train8_x <- subset(train_wines, select= c(alcohol, sulphates, total.sulfur.dioxide, volatile.acidity, density, citric.acid, residual.sugar, pH))
test8_x <- subset(test_wines, select= c(alcohol, sulphates, total.sulfur.dioxide, volatile.acidity, density, citric.acid, residual.sugar, pH))
train_y <- train_wines$quality
test_y <- test_wines$quality

knn3_pred <- knn(train8_x, test8_x, train_y, k=3)
knn3_cm <- table(knn3_pred, test_y)
knn3_cm

knn1_pred <- knn(train8_x, test8_x, train_y, k=1)
knn1_cm <- table(knn1_pred, test_y)
knn1_cm

knn3_cmstat <- confusionMatrix(knn3_cm)
knn1_cmstat <- confusionMatrix(knn1_cm)

knn3_cmstat
knn1_cmstat

# With 8 features, KNN1 is 58% accurate and KNN3 is 55% accurate. Doesn't look like adding the extra features contributes much. Likely better to stick with 6 for model simplicity.

# Random Forests

set.seed(123, sample.kind ="Rounding")
wine_rf <- randomForest(as.factor(quality)~alcohol+ sulphates+ total.sulfur.dioxide+ volatile.acidity+ density+ citric.acid+ residual.sugar+ pH, data=train_wines, importance=TRUE, ntree = 500)
wine_rf

varImpPlot(wine_rf)

plot(wine_rf, ylim=c(0,0.36))
wine_rf$confusion

# The error rate declines as more trees are added to the forest (greater ntree), however the error rate is minimized with fewer than 100 trees.

set.seed(123, sample.kind ="Rounding")
final_prediction <- predict(wine_rf, test_wines)

confMat <- table(actual = test_wines$quality,predicted = final_prediction)
confMat

accuracy <- sum(diag(confMat))/sum(confMat)
accuracy

# The random forest model is 67% accurate, which is an improvement over KNN (KNN1 with 6 features - 60%)
