#################################################################################
############################### Bike Sharing Demand #############################
#################################################################################

#================================= Prepare data =================================
#================================================================================

data_hour = read.csv("hour.csv")
library(caret)

# Split data into training and validation set
set.seed(3)
split = createDataPartition(data_hour$casual, p=0.80, list=FALSE)       # Predict on casual users only
validation = data_hour[-split,]
dataset = data_hour[split,]
dataset = dataset[,c(-1,-2,-4)]                                         # remove instance id and date column

# View strucutre of data
str(dataset)
apply(dataset[,1:7],2,unique)


#=============================== 1. Explore data ================================
#================================================================================

summary(dataset)

#======================= Visualisations =======================

#=== Univariate charts ===

# Analyse frequency count
ggplot(dataset) + geom_bar(aes(x=season))
ggplot(dataset) + geom_bar(aes(x=mnth))
ggplot(dataset) + geom_bar(aes(x=hr))
ggplot(dataset) + geom_bar(aes(x=holiday))
ggplot(dataset) + geom_bar(aes(x=weekday))
ggplot(dataset) + geom_bar(aes(x=workingday))
ggplot(dataset) + geom_bar(aes(x=weathersit))
ggplot(dataset) + geom_histogram(aes(x=temp), binwidth = 0.01)
ggplot(dataset) + geom_histogram(aes(x=atemp), binwidth = 0.01)
ggplot(dataset) + geom_histogram(aes(x=hum), binwidth = 0.01)
ggplot(dataset) + geom_histogram(aes(x=windspeed), binwidth = 0.01)
ggplot(dataset) + geom_histogram(aes(x=casual), binwidth = 25)

#================ Check if Gaussian distribution ================

plot(density(dataset$season))
plot(density(dataset$mnth))
plot(density(dataset$hr))
plot(density(dataset$holiday))
plot(density(dataset$weekday))
plot(density(dataset$workingday))
plot(density(dataset$weathersit))
plot(density(dataset$temp))
plot(density(dataset$atemp))
plot(density(dataset$hum))
plot(density(dataset$windspeed))
plot(density(dataset$casual))

#====================== Calculate skewnwess =====================

library(e1071)
apply(dataset[,9:10],2,skewness)

#====================== Check for outliers ======================

boxplot(dataset$season)
boxplot(dataset$mnth)
boxplot(dataset$hr)
boxplot(dataset$holiday)
boxplot(dataset$weekday)
boxplot(dataset$workingday)
boxplot(dataset$weathersit)
boxplot(dataset$temp)
boxplot(dataset$atemp)
boxplot(dataset$hum)
boxplot(dataset$windspeed)
boxplot(dataset$casual)

ggplot(dataset) + geom_point(aes(x=hum, y=casual)) 
outlier_boundary = mean(dataset$hum) - 3*sd(dataset$hum)      # remove outlier in humidity
dataset$hum[dataset$hum <outlier_boundary] = NA


#======= Analyse relationships of independent variables ==========

pairs(dataset[,6:9])
ggplot(dataset) + geom_point(aes(x=temp, y=atemp), position ='jitter')  # strong correlation between air temp and feeling temp

#== Analyse correlations between independent variables
library(corrplot)
correlations = cor(dataset[,6:10])
corrplot(correlations, method = 'circle', type = 'upper')


#======= Analyse relationships of independent variables ==========
#==================  and dependent variable ======================

library(bestNormalize)
casual_transformed = predict(boxcox(dataset$casual+0.1))    # casual renters transformed to Gaussian distribution

summary(aov(casual_transformed ~ dataset$season))           # p = 0.000 indicates there is a chance that there is a relationship between season and casual renters
summary(aov(casual_transformed ~ dataset$mnth))             # p = 0.000 indicates there is a chance that there is a relationship between month and casual renters
summary(aov(casual_transformed ~ dataset$hr))               # p = 0.000 indicates there is a chance that there is a relationship between hour and casual renters
t.test(casual_transformed ~ dataset$holiday)                # p = 0.05458 indicates there is a chance that there is a relationship between holiday and casual renters
summary(aov(casual_transformed ~ dataset$weekday))          # p = 0.13 indicates there is no relationship between weekday and casual renters
t.test(casual_transformed ~ dataset$workingday)             # p = 0.000 indicates there is a chance that there is a relationship between workingday and casual renters
summary(aov(casual_transformed ~ dataset$weathersit))       # p = 0.000 indicates there is a chance that there is a relationship between weather situation and casual renters

temp_transformed = predict(boxcox(dataset$temp))            # independent variables transformed to Gaussian distribution
atemp_transformed = predict(boxcox(dataset$atemp + 0.1))
hum_transformed = predict(boxcox(dataset$hum + 0.1))
windspeed_transformed = predict(boxcox(dataset$windspeed + 0.1))

cor(casual_transformed,temp_transformed)                    # cor = 0.556   some positive relationship between air temperature and casual renters
cor(casual_transformed,atemp_transformed)                   # cor = 0.551   some positive relationship between feeling temperature and casual renters
cor(casual_transformed,hum_transformed)                     # cor = -0.38   little negative relationship between humidity and casual renters
cor(casual_transformed,windspeed_transformed)               # cor = 0.11    little positive relationship between windspeed and casual renters

# Try removing atemp (as highly correlated with temp)
# Try removing weekday (as little variance)
# Try removing humidity and windspeed as little correlation with casual renters
# Try removing humidity outlier


#========================== 2. Data Transformation ==============================
#================================================================================

library(bestNormalize)
dataset_selected = dataset[,c(-13,-14)]                                                # remove registered count and total count
for (i in 8:12) { dataset_selected[,i] = predict(boxcox(dataset_selected[,i] + 0.1))}  # transform numeric variables to Gaussian

str(dataset_selected)
summary(dataset_selected)                                                              # check transformation
par(mfrow = c(2,3))
for(i in 8:12){ plot(density(dataset_selected[,i])) }


#========================== 3. Spot-check algorithms ============================
#================================================================================

#===== Sampling and evaluation metric ===
control = trainControl(method = 'repeatedcv', number = 10, repeats =5)
metric = 'RMSE'
seed = 10

#============================== A. Linear Models==============================
library(doMC)
registerDoMC(cores=4)
library(elasticnet)
library(pls)

models = c('lm','rlm','pls','enet', 'glmnet','lars','glm')
# Linear Regression, Robust Linear Model, Partial Least Squares, Elastinet, Regularised Regresssion, Least Angle Regression, Logisitc Regression

results = list()
j = 0 
for (i in models) {
  set.seed(seed)
  fit = train(casual ~., data = dataset_selected, method = i, metric = metric, trControl = control)
  j = j +1
  results[[j]] = fit
}
names(results) = models
summary(resamples(results))

# RMSE 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# lm     0.6205147 0.6316855 0.6423318 0.6401356 0.6463703 0.6697148    0
# rlm    0.6200277 0.6317695 0.6422044 0.6401795 0.6462666 0.6699780    0
# pls    0.6526539 0.6677491 0.6759383 0.6767395 0.6848914 0.7051299    0
# enet   0.6205143 0.6316845 0.6423293 0.6401350 0.6463707 0.6697151    0
# glmnet 0.6205870 0.6314195 0.6424352 0.6400927 0.6461065 0.6693383    0
# lars   0.6205147 0.6316855 0.6423318 0.6401356 0.6463703 0.6697148    0
# glm    0.6205147 0.6316855 0.6423318 0.6401356 0.6463703 0.6697148    0
# 
# Rsquared 
#             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# lm     0.5578577 0.5814314 0.5876227 0.5905327 0.6014926 0.6170123    0
# rlm    0.5575358 0.5814331 0.5876020 0.5905123 0.6014319 0.6170584    0
# pls    0.5162683 0.5356691 0.5423933 0.5422432 0.5520507 0.5707123    0
# enet   0.5578587 0.5814310 0.5876236 0.5905335 0.6014911 0.6170120    0
# glmnet 0.5581722 0.5812457 0.5877517 0.5905955 0.6013143 0.6168766    0
# lars   0.5578577 0.5814314 0.5876227 0.5905327 0.6014926 0.6170123    0
# glm    0.5578577 0.5814314 0.5876227 0.5905327 0.6014926 0.6170123    0


#============================== B. Non - Linear Models==============================

for(i in 1:7){ dataset_selected[,i] = as.factor(dataset_selected[,i])}

#====== K-Nearest Neighbour ======
set.seed(seed)
tunegrid <- expand.grid(.k=1:10)
fit.knn <- train(casual~., data=dataset_selected, method="knn", metric=metric, tuneGrid=tunegrid, trControl=control)
print(fit.knn)

results = summary(resamples(list (knn=fit.knn)))

# k   RMSE       Rsquared   MAE      
# 1  0.4826590  0.7808613  0.3590346
# 2  0.4067970  0.8365955  0.3069159
# 3  0.3852938  0.8522808  0.2918803
# 4  0.3788274  0.8568738  0.2880943
# 5  0.3756947  0.8591567  0.2863068
# 6  0.3741351  0.8603516  0.2855527
# 7  0.3734981  0.8608633  0.2852455
# 8  0.3739151  0.8606030  0.2859017
# 9  0.3734302  0.8610713  0.2856066
# 10  0.3736403  0.8610181  0.2858233


#===== Support Vector Machine =====
#==== with Radial Basis Kernel ====

library(kernlab)
fit.svm <- train(casual~., data=dataset_selected, method="svmRadial", metric=metric, trControl=control)
print(fit.svm)

#   C     RMSE       Rsquared   MAE      
# 0.25  0.3722934  0.8626243  0.2833549
# 0.50  0.3643021  0.8679325  0.2771636
# 1.00  0.3606249  0.8703321  0.2745464


#============================== C. Regression Trees ==============================

for(i in 1:7){ dataset_selected[,i] = as.factor(dataset_selected[,i])}

#====== CART ======

library(rpart)
set.seed(seed)
fit.cart <- rpart(casual ~., data = dataset_selected, control=rpart.control(minsplit=5))
predictions <- predict(fit.cart, dataset_selected[,1:11])

rmse = sqrt(mean((dataset_selected$casual - predictions)^2))
mse <- mean((dataset_selected$casual - predictions)^2)
accuracy =1 - (sum((dataset_selected$casual - predictions)^2)/sum((dataset_selected$casual - mean(dataset_selected$casual))^2))

# rmse: 0.5126
# mse: 0.2631488
# accuracy : 0.736832


#============================== D. Ensembles ==============================

#====== Random Forest ======

library(randomForest)
set.seed(seed)
fit.rf <- train(casual~., data=dataset_selected, method="rf", ntree = 20, metric=metric, trControl=control)
print(fit.rf)

#   mtry  RMSE       Rsquared   MAE      
# 2    0.3855593  0.8557302  0.2981464
# 6    0.3522343  0.8759948  0.2689868
# 11    0.3581051  0.8718824  0.2729314

#====== Bagged CART ======

set.seed(seed)
fit.treebag <- train(casual~., data=dataset_selected, method="treebag", metric=metric, trControl=control)
print(fit.treebag)

#   RMSE       Rsquared   MAE      
# 0.6558802  0.5739196  0.5103539


#===== Stochastic Gradient =====
#=========== Boosting ==========

library(gbm)
set.seed(seed)
fit.gbm <- train(casual~., data=dataset_selected, method="gbm", metric=metric, trControl=control, verbose=FALSE)
summary(resamples(list(gbm=fit.gbm)))
print(fit.gbm)

# interaction.depth  n.trees  RMSE       Rsquared   MAE      
# 1                   50      0.5537760  0.7289003  0.4316063
# 1                  100      0.4872379  0.7756629  0.3784144
# 1                  150      0.4612512  0.7928375  0.3575615
# 2                   50      0.4680050  0.7927259  0.3663238
# 2                  100      0.4219690  0.8239476  0.3299666
# 2                  150      0.4087938  0.8335235  0.3191725
# 3                   50      0.4323040  0.8192817  0.3385821
# 3                  100      0.3978507  0.8425489  0.3109249
# 3                  150      0.3869089  0.8505277  0.3015611

#===== Cubist (Boosting) =====

set.seed(seed)
fit.cubist <- train(casual~., data=dataset_selected, method="cubist", metric=metric, trControl=control)
print(fit.cubist)

# committees  neighbors  RMSE       Rsquared   MAE      
# 1          0          0.3733197  0.8611250  0.2847539
# 1          5          0.3569087  0.8732976  0.2701966
# 1          9          0.3539680  0.8750875  0.2692229
# 10          0          0.3585599  0.8722863  0.2739154
# 10          5          0.3435647  0.8822317  0.2607029
# 10          9          0.3405624  0.8841298  0.2597004
# 20          0          0.3570507  0.8734531  0.2726967
# 20          5          0.3423507  0.8830392  0.2598420
# 20          9          0.3392689  0.8849999  0.2587405


#============================ 4. Evaluate algorithms ============================
#================================================================================

# 3 models came up with the highest fit accuracies (R-squared),
# Support Vector Machine, Random Forest and Cubist

results_selected = resamples(list (svm=fit.svm, rf=fit.rf, cb=fit.cubist))


# MAE 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# svm 0.3294626 0.3482873 0.3522409 0.3510266 0.3560496 0.3658624    0
# rf  0.2574890 0.2640059 0.2681741 0.2683164 0.2717670 0.2794298    0
# cb  0.2469333 0.2562039 0.2584200 0.2587405 0.2623868 0.2673745    0
# 
# RMSE 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# svm 0.4370527 0.4605037 0.4661086 0.4649663 0.4725580 0.4863254    0
# rf  0.3347577 0.3460214 0.3503749 0.3517964 0.3575403 0.3725355    0
# cb  0.3255104 0.3334554 0.3396623 0.3392689 0.3446640 0.3525490    0
# 
# Rsquared 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# svm 0.7616147 0.7786509 0.7837098 0.7854709 0.7909592 0.8137928    0
# rf  0.8592843 0.8726937 0.8766974 0.8763138 0.8798131 0.8859579    0
# cb  0.8740415 0.8825455 0.8846767 0.8849999 0.8879373 0.8949615    0
         
#=== Box & Whiskers ===
scales <- list(x = list(relation='free'), y=list(relation='free'))
bwplot(results_selected, scales = scales)

#=== Density Plots ===
densityplot(results_selected, scales=scales, pch ='|')

#=== Dot Plots ===
dotplot(results_selected, scales=scales)

#=== Parallel Plots ===
parallelplot(results_selected)

#=== Scatterplot matrix ===
splom(results_selected)  


#==================== 5. Improve Accuracy and Tune Models =======================
#================================================================================

# Here I will select Random Forest Model to tune and evaluate

#==================== Tune Random Forest Model =======================

trainControl <- trainControl(method="cv", number=10, search="grid")
metric = 'RMSE'
set.seed(78)

#=== Find the best mtry - the number of variables randomly sampled ===
tunegrid <- expand.grid(.mtry=c(1:11))
fit.rf <- train(casual~., data=dataset_selected, method="rf", metric=metric, tuneGrid=tunegrid, trControl=trainControl, importance= TRUE, nodesize =5, ntree = 5)
print(fit.rf)

# mtry  RMSE       Rsquared   MAE      
# 1    0.5816264  0.7182689  0.4548942
# 2    0.4127578  0.8308755  0.3196869
# 3    0.3917680  0.8464831  0.3004894
# 4    0.3794804  0.8560830  0.2900777
# 5    0.3782278  0.8571747  0.2889529
# 6    0.3778317  0.8576047  0.2873271
# 7    0.3787190  0.8570529  0.2871619
# 8    0.3799261  0.8560793  0.2883088
# 9    0.3769633  0.8584830  0.2861518
# 10    0.3795418  0.8565119  0.2892823
# 11    0.3837596  0.8534970  0.2917720


#=== Find maximum number of terminal nodes ===

store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = 9)

for (maxnodes in c(1:300)){
set.seed(14)
fit.rf <- train(casual~., data=dataset_selected, method="rf", metric=metric, tuneGrid=tunegrid, trControl=trainControl, importance= TRUE, nodesize =5, maxnodes = maxnodes, ntree = 5)
current_iteration <- toString(maxnodes)
store_maxnode[[current_iteration]] <- fit.rf
}

results_maxnode <- resamples(store_maxnode)
summary(results_maxnode)

# Rsquared 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# 253 0.8405315 0.8508707 0.8524152 0.8540588 0.8577132 0.8690271    0


#=== Find the optimal number of trees ===

store_maxtrees <- list()
tuneGrid <- expand.grid(.mtry = 9)

for (ntree in c(30, 50)){
  set.seed(45)
  fit.rf <- train(casual~., data=dataset_selected, method="rf", metric=metric, tuneGrid=tunegrid, trControl=trainControl, importance= TRUE, nodesize =5, maxnodes = 253, ntree = ntree)
  current_iteration <- toString(ntree)
  store_maxtrees[[current_iteration]] <- fit.rf
}

results_ntree <- resamples(store_maxtrees)
summary(results_ntree)

#       rmse        R-squared median
# 5     0.38466       0.8516399
# 20    0.3765255     0.8585559
# 30    0.3762        0.8596
# 50    0.375055      0.8593


#=== Find the optimal node size ===
# minimal number of observations 
# in a node in order for a split to be performed

store_nodesize <- list()
tuneGrid <- expand.grid(.mtry = 9)

for (nodesize in c(100)){
  set.seed(5)
  fit.rf <- train(casual~., data=dataset_selected, method="rf", metric=metric, tuneGrid=tunegrid, trControl=trainControl, importance= TRUE, nodesize = nodesize, maxnodes = 253, ntree = 20)
  current_iteration <- toString(nodesize)
  store_nodesize[[current_iteration]] <- fit.rf
}

results_nodesize <- resamples(store_nodesize)
summary(results_nodesize)

# RMSE: 0.3859
# R-squared: 0.865


#==================== Train Random Forest Model =======================

#== Train model using optimal parameters ==

#== Process validation data set ==

library(bestNormalize)
validation = validation[,c(-1,-2,-4)]                                                        # remove instance id and date column
validation_selected = validation[,c(-13,-14)]                                                # remove registered count and total count
for (i in 8:12) { validation_selected[,i] = predict(boxcox(validation_selected[,i] + 0.1))}  # transform numeric variables to Gaussian


#======= Train and evaluate model on optimal parameters ====

tuneGrid <- expand.grid(.mtry = 9)

evaluate = function(dataset_selected, validation_selected){
  fit.rf <- train(casual~., data=dataset_selected, method="rf", metric=metric, tuneGrid=tuneGrid, trControl=trainControl, importance= TRUE, nodesize =100, maxnodes = 253, ntree = 20)
  prediction <-predict(fit.rf, validation_selected)
  
  # Evaluation metric
  prediction_actual = data.frame("prediction" = prediction, 'actual' = validation_selected$casual)
  prediction_actual$sq_errors = (prediction_actual$prediction - prediction_actual$actual)**2
  prediction_actual$errors_mean = (prediction_actual$actual - mean(prediction_actual$actual))**2
  
  rmse = sqrt(mean(prediction_actual$sq_errors))
  accuracy = 1 - sum(prediction_actual$sq_errors) / sum(prediction_actual$errors_mean)          # Nash-Sutcliffe normalised RMSE
  return(c(rmse, accuracy))
}

evaluate(dataset_selected, validation_selected)

# RMSE: 0.3663
# Accuracy: 86.6% 


#===== Try to improve accuracy by removing some features =====

#=== Remove atemp ===
dataset_selected_x_atemp = subset(dataset_selected, select = -c(atemp))
validation_selected_x_atemp = subset(validation_selected, select = -c(atemp))
evaluate(dataset_selected_x_atemp, validation_selected_x_atemp)

# RMSE: 0.367
# Accuracy: 86.5% 

#=== Remove weekday ===
dataset_selected_x_weekday = subset(dataset_selected, select = -c(weekday))
validation_selected_x_weekday = subset(validation_selected, select = -c(weekday))
evaluate(dataset_selected_x_weekday, validation_selected_x_weekday)

# RMSE: 0.37
# Accuracy: 0.863

#=== Remove humidity ===
dataset_selected_x_humidity = subset(dataset_selected, select = -c(hum))
validation_selected_x_humidity = subset(validation_selected, select = -c(hum))
evaluate(dataset_selected_x_humidity, validation_selected_x_humidity)

# RMSE: 0.372
# Accuracy: 0.861

#=== Remove windspeed ===
dataset_selected_x_windspeed = subset(dataset_selected, select = -c(windspeed))
validation_selected_x_windspeed = subset(validation_selected, select = -c(windspeed))
evaluate(dataset_selected_x_windspeed, validation_selected_x_windspeed )

# RMSE: 0.367
# Accuracy: 0.865

#=== Remove holiday ===
dataset_selected_x_holiday = subset(dataset_selected, select = -c(holiday))
validation_selected_x_holiday = subset(validation_selected, select = -c(holiday))
evaluate(dataset_selected_x_holiday, validation_selected_x_holiday )

# RMSE: 0.367
# Accuracy: 0.865



#=============================== 6. Save final model ============================
#================================================================================

#== Final model ==
fit.rf <- train(casual~., data=dataset_selected, method="rf", metric="RMSE", tuneGrid=data.frame(mtry = 9), importance= TRUE, nodesize =100, maxnodes = 253, ntree = 20)
saveRDS(fit.rf, "./finalModel.rds")


#=== View feature importance ===

varimp <- varImp(fit.rf)
plot(varimp, main="Variable Importance")

# Overall
# hr         100.000
# workingday  23.838
# hum         11.496
# weathersit   9.300
# temp         8.204
# season       6.210
# mnth         4.224
# atemp        1.455
# windspeed    1.401
# weekday      0.332
# holiday      0.000
