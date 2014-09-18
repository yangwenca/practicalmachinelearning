---
output: html_document
---
# Practical Machine Learning Project

## Data Processing

Load the data both training set and testing set. There are 19622 rows and 160 columns in the data.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pmltrain <- read.csv("pml-training.csv")
pmltest <- read.csv("pml-testing.csv")
nrow(pmltrain)
```

```
## [1] 19622
```

```r
ncol(pmltrain)
```

```
## [1] 160
```

It is a large dataset, removing unrelated parameters is important. The index, user_names, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp are removed, because the correct fashion is indenpendent on time, user_name, window and index. 


```r
set.seed(111)
pmltrain <- pmltrain[,c(-1,-2,-3,-4,-5, -6)]
pmltest <- pmltest[,c(-1,-2,-3,-4,-5, -6)]
```

There are lots of NA values in the dataset. I remove columns if all of values in this column is NA. Columns with near zero variance is removed in order to increase speed. Now, the number of columns is reduced from 160 to 54. At the beginning, I do not remove columns with near zero variance. But, functions like "gbm", "rf" can't generate an algorthim due to large size of dataset. 

```r
allNA <- apply(pmltrain, 2, function(x){
    sum(is.na(x))
})

allNA2 <- apply(pmltest, 2, function(x){
    sum(is.na(x))
})
pmltrain<-pmltrain[,which(allNA==0)]
pmltest<-pmltest[,which(allNA2==0)]
novar <- nearZeroVar(pmltrain)
pmltrain <- pmltrain[, -novar]
ncol(pmltrain)
```

```
## [1] 54
```

This is a medium size data set. 60% of data is using as training, and 40% of data is using of cross-validation.


```r
inTrain <- createDataPartition(y=pmltrain$classe, p=0.6, list=FALSE)
training <- pmltrain[inTrain,]
cross <- pmltrain[-inTrain,]
dim(training); dim(cross)
```

```
## [1] 11776    54
```

```
## [1] 7846   54
```

## Build a machine learning algorithm

I use gbm as the method to train the data, since it has high accurancy and is built from simple models. Gbm is generalized boosted regression models. It includes regression methods for least squares, absolute loss, t-distribution,etc. The advantages of gbm are that it uses several weak predictors and builds a strong predictor from weak predictors. The accurancy of gbm is relatively high. But it might take long time to run.



```r
if(!file.exists("m.rda")){
    myControl <- trainControl(method="cv", number=4, allowParallel=TRUE)
    model1 <- train(training$classe ~ ., data=training, method="gbm", trControl=myControl)
    save(model1, file="m.rda")
}else{
    load("m.rda")
}
```

## Cross-Validation
In order to evaulate the accurancy of the model, I use the cross data set. From the results, the accuracy is 98.5%.

```r
pred <- predict(model1, cross)
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
## Loading required package: plyr
```

```r
cross$predRight <- pred == cross$classe
confusionMatrix(pred, cross$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2224   25    0    0    0
##          B    8 1469   11    9    4
##          C    0   24 1353   16    5
##          D    0    0    3 1260   10
##          E    0    0    1    1 1423
## 
## Overall Statistics
##                                         
##                Accuracy : 0.985         
##                  95% CI : (0.982, 0.988)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.981         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.968    0.989    0.980    0.987
## Specificity             0.996    0.995    0.993    0.998    1.000
## Pos Pred Value          0.989    0.979    0.968    0.990    0.999
## Neg Pred Value          0.999    0.992    0.998    0.996    0.997
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.283    0.187    0.172    0.161    0.181
## Detection Prevalence    0.287    0.191    0.178    0.162    0.182
## Balanced Accuracy       0.996    0.981    0.991    0.989    0.993
```

## Out of Sample Error

From the above, the out of sample error is 1.5%.

## Save the prediction for testing data

All of my prediction are correct since it passes all test cases. It indicates the high accurancy of the model I build.

```r
results <- predict(model1, pmltest)
length(results)
```

```
## [1] 20
```

```r
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(results)
```
