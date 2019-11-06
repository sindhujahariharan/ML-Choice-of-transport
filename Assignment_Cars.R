
library(caret)
library(dlookr)## Please load package "caret" before "mlr"- explaination follows
library(e1071)
library(rpart)
?impute
library(rpart.plot)
library(mlr)

data<-setwd("C:\\Users\\kausik\\Documents\\Sindhu\\PGP BABI\\Machine learning\\Group Assignment")

#Read Data

data<- read.csv("Cars.csv")

## Split the data in Dev & Hold Out sample (70:30)

?createDataPartition
Train.rows<- createDataPartition(y= data$Transport, p=0.7, list = FALSE)
Train<- data[Train.rows,] # 70% data goes in here
table(Train$Transport)

Test<- data[-Train.rows,] # 30% data goes in here
table(Test$Transport)


## To check for any missing values and outliers.

summarizeColumns(Train)
summarizeColumns(Test)
summarizeColumns(imp_train)
summarizeColumns(imp_test)


## EDA

  #Converting variables into factors

Train$Engineer<-as.factor(Train$Engineer)
Train$MBA<-as.factor(Train$MBA)
Train$license<-as.factor(Train$license)

Test$Engineer<-as.factor(Test$Engineer)
Test$MBA<-as.factor(Test$MBA)
Test$license<-as.factor(Test$license)


  #impute missing values by mean and mode

imp <- impute(obj = Train, classes = list(factor = imputeMode(), integer = imputeMean()))
imp1 <- impute(obj = Test, classes = list(factor = imputeMode(), integer = imputeMean()))

imp_train <- imp$data
imp_test <- imp1$data


### Going fwd, we need to use imp_test & imp_train data set since this contains imputed data too


  #Analyzing & Removing outliers

boxplot(imp_train$Salary)
boxplot(imp_train$Work.Exp)

imp_train$Salary<-imputate_outlier(imp_train, Salary, method = "capping")
summary(imp_train$Salary)
imp_train$Work.Exp<-imputate_outlier(imp_train, Work.Exp, method = "capping")
summary(imp_train$Work.Exp)


  #Observing skewed data and standardising them.


skewness(imp_train$Age)
skewness(imp_train$Work.Exp)##
skewness(imp_train$Salary)##
skewness(imp_train$Distance)

hist(imp_train$Age, breaks = 300, main = "Age Chart",xlab = "Age") #----> little bit positively skewed
hist(imp_train$Work.Exp, breaks = 300, main = "Wrok Exp Chart",xlab = "Work Exp") #----> little bit positively skewed
hist(imp_train$Salary, breaks = 300, main = "Salary Chart",xlab = "Salary")  #----> positively skewed
hist(imp_train$Distance, breaks = 300, main = "Distance Chart",xlab = "Distance")

imp_train$Work.Exp<-as.numeric(imp_train$Work.Exp)
imp_train$Salary<-as.numeric(imp_train$Salary)
imp_test$Work.Exp<-as.numeric(imp_test$Work.Exp)
imp_test$Salary<-as.numeric(imp_test$Salary)

?transform
imp_train$Work.Exp<-transform(imp_train$Work.Exp, method = "log+1")
imp_train$Salary<-transform(imp_train$Salary, method = "log+1")
imp_test$Work.Exp<-transform(imp_test$Work.Exp, method = "log+1")
imp_test$Salary<-transform(imp_test$Salary, method = "log+1")




##scaling to remove skewness in Train and Test

?scale
imp_train$Work.Exp<-scale(imp_train$Work.Exp,center =T,scale =T)
imp_train$Salary<-scale(imp_train$Salary,center =T,scale =T)

imp_test$Work.Exp<-scale(imp_test$Work.Exp,center =T,scale =T)
imp_test$Salary<-scale(imp_test$Salary,center =T,scale =T)



### 

## Essential Step in mlr package. You have to ensure that response variable is identified correctly

trainTask = makeClassifTask(data = imp_train,target = "Transport")
testTask = makeClassifTask(data = imp_test, target = "Transport")

#create a separate datset for test prediction

trainTask ## To check the details

str(getTaskData(trainTask)) ## To check details

#===================================================================================================================

## Method 1- Naive Bayes

## SOP
## 1. Make Learner
## 2. Train Learner with Task
## 3. Predict



nb.learner = makeLearner("classif.naiveBayes")

nb.model = train(nb.learner, trainTask)
nb.predict=predict(nb.model,testTask)

preddata<-read.csv("testprediction.csv")
summarizeColumns(preddata)
preddata$Engineer<-as.factor(preddata$Engineer)
preddata$MBA<-as.factor(preddata$MBA)
preddata$license<-as.factor(preddata$license)

newdata.pred = predict(nb.model,newdata=preddata)
newdata.pred
##table(nbpredict$data$truth,nbpredict$data$response)
confusionMatrix(nb.predict$data$truth,nb.predict$data$response,positive="1")


#===================================================================================================================


## Method 4: CART

getParamSet("classif.rpart") ## Gets you tunable parameters
cart.learner = makeLearner("classif.rpart", predict.type = "response")
cart.model = train(cart.learner, trainTask)

cartModel=getLearnerModel(cart.model) ## In case you need to plot tree

prp(cartModel,extra=2, roundint=FALSE)## For plotting tree, you may need rpart.plot

#make predictions
cart.predict = predict(cart.model, testTask)


summarizeColumns(preddata)
preddata$Engineer<-as.factor(preddata$Engineer)
preddata$MBA<-as.factor(preddata$MBA)
preddata$license<-as.factor(preddata$license)

newdata.pred = predict(cart.model,newdata=preddata)
newdata.pred

##table(tpmodel$data$truth,tpmodel$data$response)
confusionMatrix(cart.predict$data$truth,cart.predict$data$response,positive="1")



#====================================================================================================================



## Method: 6 SVM
getParamSet("classif.ksvm") #do install kernlab package 
ksvm.learner = makeLearner("classif.ksvm", predict.type = "response")

ksvm.model = train(ksvm.learner, trainTask)

ksvm.predict = predict(ksvm.model, testTask)

newdata.pred = predict(ksvm.model,newdata=preddata)
newdata.pred

## table(ksvm.predict$data$truth,ksvm.predict$data$response)
confusionMatrix(ksvm.predict$data$truth,ksvm.predict$data$response)



#=====================================================================================================================



#Method 9: knn

getParamSet("classif.knn")

knn.learner=makeLearner("classif.knn",predict.type = "response")
knn.model=train(knn.learner,trainTask)
knn.predict=predict(knn.model, testTask)

confusionMatrix(knn.predict$data$truth,knn.predict$data$response)



#=====================================================================================================================
