####################
# Import libraries #
####################
library(caret) # Stratified split
library(mice) # Multiple imputation
library(ROSE) # ROSE
library(DMwR) # SMOTE
library(corrplot) # Correlation plot
library(readr)
library(dplyr)
library(ggraph)
library(igraph)
library(nnet) 
library(randomForest)
library(e1071)
library(rpart)
library(adabag)
library(themis)
library(naivebayes)

# Read the data
enron <- read.csv("enron.csv")
# Change POI attribute type from character to factor
enron$poi <- as.factor(enron$poi)
# Change NaN to NA
is.nan.data.frame <- function(x)
do.call(cbind, lapply(x, is.nan))
enron[is.nan(enron)] <- NA
# Remove records with 18 or more missing values
enron = enron[rowSums(is.na(enron)) < 18,] # remove LOCKHART EUGENE
# Remove record of travel agency in the park
enron = enron[enron$name != 'THE TRAVEL AGENCY IN THE PARK',] 
# Use stratified split to get training and test set indices
set.seed(1)
index <- createDataPartition(enron$poi,p=0.7,list = FALSE)


###############################################################################
#############################
# Exploratory data analysis #
#############################

# Summary statistics
summary(enron)
# Attribute information
str(enron)
# Row and column numbers
dim(enron)
# Class labels ratio
table(enron$poi)

# Create 2 new datasets based on class labels
### Only used for EDA
enron_false <- enron[enron$poi=="False",]
enron_true <- enron[enron$poi=="True",]

### Pairwise boxplot comparisons between POI and non-POI
enron.boxplot <- function(x) {
  par(mfrow=c(1,2))
  enron_tru <- enron_true[,-c(1,17,21)]
  enron_fals <- enron_false[,-c(1,17,21)]
  boxplot(enron_tru[,x],xlab=paste(colnames(enron_tru)[x],"for POI"))
  boxplot(enron_fals[,x],xlab=paste(colnames(enron_fals)[x],"for non-POI"))
}

### enron.boxplot(i) for i=1,2,...,19 for 19 pairs of boxplot for each numerical feature,
### left boxplot uses only POI data, right boxplot uses only non-POI data
# Example for salary:
enron.boxplot(1)


### Outlier analysis 

# Boxplots of all numeric features
par(mfrow=c(1,1),mar=c(9,5,2,2),cex.axis=0.8)
boxplot(enron[,-c(1,17,21)],las=2,main="Boxplots of numerical features")
### One VERY FAR point for total payments, potential outlier ###

# Extract outlier: record with the largest total payment
outlier <- max(boxplot.stats(enron$total_payments)$out)
View(enron[which(enron$total_payments==outlier),])
### This record belongs to one of the POI, should not remove


### Missing data 

# Check missing values
sum(is.na(enron))
# Number of missing values for each feature
for (i in 1:ncol(enron)){
  cat(colnames(enron[i]),":",sum(is.na(enron[,i])),"\n")
}
# Proportion of missing values for each feature
for (i in 1:ncol(enron)){
  cat(colnames(enron[i]),":",round(sum(is.na(enron[,i]))/nrow(enron),2),"\n")
}
### Restricted stock deferred 88%, loan advances 98%, director fees 89%
### Too many missing values, may be justified to remove
### For each of those 3 features, check the records which HAVE values
View(enron[which(!is.na(enron$restricted_stock_deferred)),])
### All non-POI
View(enron[which(!is.na(enron$loan_advances)),])
### Only 3 records, not much can be said
View(enron[which(!is.na(enron$director_fees)),])
### LOTS of missing values for almost every other feature
###############################################################################

######################
# Data preprocessing #
######################

# Remove names, email address and features with too many missing values
enron <- enron[,-c(1,10,13,18,21)]

### Multiple imputation 
# Indices of test values
ignore_index <- rep(FALSE,143)
ignore_index[-index]<-TRUE

# Mean imputation on Enron data, using only training data
imp<-mice(enron, ignore = ignore_index,method = "mean", seed = 1)
enron_imp<-complete(imp)

# Create training and test sets using imputed data
enron_imp_train <- enron_imp[index,]
enron_imp_test <- enron_imp[-index,]


############
# Sampling #
############

# ROSE
enron_rose <- ROSE(poi ~ ., data = enron_imp_train, seed = 1)$data

# SMOTE
set.seed(1)
enron_smote <- SMOTE(poi~.,enron_imp_train)



###############################################################################
###                              Data modelling                             ###
###############################################################################

###############################################################################
###                        Without cross validation                         ###
###############################################################################

##################
# Decision Tree  #
##################
# Training set = ROSE
# Training set = SMOTE

# ROSE
TrainSet = enron_rose
TestSet = enron_imp_test
# SMOTE
TrainSet = enron_smote 
TestSet = enron_imp_test

# Decision tree model
set.seed(1)
tree <- rpart(poi ~., data = TrainSet, method='class')
plotcp(tree)
prune_tree <- prune(tree, cp = 0.061) # 0.061 for ROSE, 0.042 for SMOTE
prediction_tree <- predict(prune_tree, TestSet, type='class')
conf_tree <-confusionMatrix(prediction_tree, TestSet$poi, positive='True')
conf_tree
conf_tree$byClass[c(5,6)]


##################
# Neural Network #
##################
# Training set = Standardized ROSE 
# Training set = Standardized SMOTE

# Standardized ROSE
TrainSet = enron_rose[,-14]
TestSet = enron_imp_test[,-14]
train_para = preProcess(TrainSet,method = c("center","scale"))
TrainSet = predict(train_para,TrainSet)
TestSet <- predict(train_para,TestSet)
TrainSet$poi <- enron_rose$poi
TestSet$poi <- enron_imp_test$poi

# Standardized SMOTE
TrainSet = enron_smote[,-14]
TestSet = enron_imp_test[,-14]
train_para = preProcess(TrainSet,method = c("center","scale"))
TrainSet = predict(train_para,TrainSet)
TestSet <- predict(train_para,TestSet)
TrainSet$poi <- enron_smote$poi
TestSet$poi <- enron_imp_test$poi

# Neural network
err11=0
err12=0
n_tr=dim(TrainSet)[1]
n_te=dim(TestSet)[1]
for(i in seq(1, 301, 50))
{
  set.seed(1)
  model=nnet(poi ~ ., data=TrainSet,maxit=i,size=6,decay = 0.01, MaxNWts = 100000)
  err11[i]=sum(predict(model,TrainSet,type='class')!=TrainSet[,17])/n_tr
  err12[i]=sum(predict(model,TestSet,type='class')!=TestSet[,17])/n_te
}
error_1 = na.omit(err11)
error_2 = na.omit(err12)
plot(seq(1, 301, 50),error_1,col=1,type="b",ylab="Error",xlab="Epoch",ylim=c(min(min(error_1),min(error_2)),max(max(error_1),max(error_2))))
lines(seq(1, 301, 50),error_2,col=2,type="b")
legend("topleft",pch=c(15,15),legend=c("Train","Test"),col=c(1,2),bty="n")

set.seed(1)
ann_best=nnet(poi ~ ., data=TrainSet,maxit=50,size=6,decay = 0.01, MaxNWts = 10000)
prediction_ann = predict(ann_best,TestSet,type="class")
table = table(prediction_ann, TestSet$poi)
conf_nn <- confusionMatrix(table,positive='True')
conf_nn
conf_nn$byClass[c(5,6)]


##########################
# Support vector machine #
##########################
# Training set = Standardized ROSE
# Training set = Standardized SMOTE

# Standardized ROSE
TrainSet = enron_rose[,-14]
TestSet = enron_imp_test[,-14]
train_para = preProcess(TrainSet)
TrainSet = predict(train_para,TrainSet)
TestSet <- predict(train_para,TestSet)
TrainSet$poi <- enron_rose$poi
TestSet$poi <- enron_imp_test$poi

# Standardized SMOTE
TrainSet = enron_smote[,-14]
TestSet = enron_imp_test[,-14]
train_para = preProcess(TrainSet)
TrainSet = predict(train_para,TrainSet)
TestSet <- predict(train_para,TestSet)
TrainSet$poi <- enron_smote$poi
TestSet$poi <- enron_imp_test$poi

# SVM
set.seed(1)
svm_tune <- tune.svm(poi ~., data=TrainSet, gamma = c(0.25,0.5,1,2,4,8,16,32,64,128), 
                     cost = c(0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12))
svm_tune$best.parameters


model_svm <- svm(poi ~., data=TrainSet, kernel = "radial", gamma = svm_tune$best.parameters$gamma,
                 cost = svm_tune$best.parameters$cost)
predict_svm <- predict(model_svm, TestSet)
conf_svm <- confusionMatrix(predict_svm, TestSet$poi, positive = 'True')
conf_svm
conf_svm$byClass[c(5,6)]


###############
# Naive Bayes #
###############
# Training set = ROSE
# Training set = SMOTE

# ROSE
TrainSet = enron_rose
TestSet = enron_imp_test
# SMOTE
TrainSet = enron_smote
TestSet = enron_imp_test


# NB model
model_nb <- naive_bayes(poi ~ ., data = TrainSet)

# Training error
trainpred_nb <- predict(model_nb, TrainSet)
table(trainpred_nb, TrainSet$poi)

# Test error
prediction_nb <- predict(model_nb, TestSet)
conf_nb <- confusionMatrix(prediction_nb,TestSet$poi, positive='True')
conf_nb
conf_nb$byClass[c(5,6)]


##################
#  Random Forest #
##################
# Training set = ROSE
# Training set = SMOTE

# ROSE
TrainSet = enron_rose
TestSet = enron_imp_test

# SMOTE
TrainSet = enron_smote
TestSet = enron_imp_test

# RF model
set.seed(1)
model_rf = randomForest(poi ~., data=TrainSet, ntree=500, importance = TRUE)           
prediction_rf = predict(model_rf, TestSet)
conf_rf <- confusionMatrix(prediction_rf, TestSet$poi, positive='True')
conf_rf
conf_rf$byClass[c(5,6)]


############
# Boosting #
############
# Training set = ROSE
# Training set = SMOTE

# ROSE
TrainSet = enron_rose
TestSet = enron_imp_test

# SMOTE
TrainSet = enron_smote
TestSet = enron_imp_test

# AdaBoost model
set.seed(1)
model_boosting = boosting(poi~., data=TrainSet, mfinal=100, importance = TRUE)
model_boosting$importance
prediction_boosting = predict(model_boosting,TestSet)
prediction_boosting


###############################################################################
###                        5-fold cross validation                          ###
###############################################################################


#################
# Decision tree #
#################

# 5-fold CV with ROSE
train_control <- trainControl(method = "cv", number = 5, sampling = "rose",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

# 5-fold CV with SMOTE
train_control <- trainControl(method = "cv", number = 5, sampling = "smote",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

set.seed(1)
model_cvtree <- train(poi ~., data = enron_imp_train,
                       method = "rpart",
                       trControl = train_control, metric = "ROC")
prediction_cvtree = predict(model_cvtree, enron_imp_test)
conf_cvtree <- confusionMatrix(prediction_cvtree, enron_imp_test$poi, positive='True')
conf_cvtree
conf_cvtree$byClass[c(5,6)]


##################
# Neural network #
##################
tgrid <- expand.grid(
  .size = 6,
  .decay = 0.01
)

# Standardized training and test sets
TrainSet = enron_imp_train[,-14]
TestSet = enron_imp_test[,-14]
train_para = preProcess(enron_imp_train)
TrainSet = predict(train_para,TrainSet)
TestSet <- predict(train_para,TestSet)
TrainSet$poi <- enron_imp_train$poi
TestSet$poi <- enron_imp_test$poi


# 5-fold CV with ROSE
train_control <- trainControl(method = "cv", number = 5, sampling = "rose",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

# 5-fold CV with SMOTE
train_control <- trainControl(method = "cv", number = 5, sampling = "smote",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

set.seed(1)
model_cvnn <- train(poi ~., data = TrainSet,
                       method = "nnet",
                       trControl = train_control, metric = "ROC",
                    tuneGrid = tgrid, maxit=50)
prediction_cvnn = predict(model_cvnn, TestSet)
conf_cvnn <- confusionMatrix(prediction_cvnn, TestSet$poi, positive='True')
conf_cvnn
conf_cvnn$byClass[c(5,6)]


##########################
# Support vector machine #
##########################

# Standardized training and test sets
TrainSet = enron_imp_train[,-14]
TestSet = enron_imp_test[,-14]
train_para = preProcess(enron_imp_train)
TrainSet = predict(train_para,TrainSet)
TestSet <- predict(train_para,TestSet)
TrainSet$poi <- enron_imp_train$poi
TestSet$poi <- enron_imp_test$poi

# 5-fold CV with ROSE
train_control <- trainControl(method = "cv", number = 5, sampling = "rose",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

# 5-fold CV with SMOTE
train_control <- trainControl(method = "cv", number = 5, sampling = "smote",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

set.seed(1)
model_cvsvm <- train(poi ~., data = TrainSet,
               method = "svmLinear2",
               trControl = train_control, metric = "ROC")
prediction_cvsvm = predict(model_cvsvm, TestSet)
conf_cvsvm <- confusionMatrix(prediction_cvsvm, TestSet$poi, positive='True')
conf_cvsvm
conf_cvsvm$byClass[c(5,6)]


###############
# Naive Bayes #
###############

# 5-fold CV with ROSE
train_control <- trainControl(method = "cv", number = 5, sampling = "rose",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

# 5-fold CV with SMOTE
train_control <- trainControl(method = "cv", number = 5, sampling = "smote",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

set.seed(1)
model_cvnb <- train(poi ~., data = enron_imp_train,
                    method = "naive_bayes",
                    trControl = train_control, metric = "ROC")


prediction_cvnb <- predict(model_cvnb, enron_imp_test)
conf_cvnb <- confusionMatrix(prediction_cvnb,enron_imp_test$poi, positive='True')
conf_cvnb
conf_cvnb$byClass[c(5,6)]


#################
# Random Forest #
#################
tgrid <- expand.grid(
  .mtry = 5
)

# 5-fold CV with ROSE
train_control <- trainControl(method = "cv", number = 5, sampling = "rose",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

# 5-fold CV with SMOTE
train_control <- trainControl(method = "cv", number = 5, sampling = "smote",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

set.seed(1)
start <- Sys.time()
model_cvrf <- train(poi ~., data = enron_imp_train,
               method = "rf",
               trControl = train_control, metric = "ROC"
               , ntree=500, importance = TRUE, tuneGrid = tgrid)
end <- Sys.time()
end-start
prediction_cvrf = predict(model_cvrf, enron_imp_test)
conf_cvrf <- confusionMatrix(prediction_cvrf, enron_imp_test$poi, positive='True')
conf_cvrf
conf_cvrf$byClass[c(5,6)]
rf <- model_cvrf$finalModel
importance(rf)
varImpPlot(rf)
####Credit https://www.bbsmax.com/A/kPzOXYjoJx/ #####
tree_func <- function(final_model,
                      tree_num) {
  # get tree by index
  tree <- randomForest::getTree(final_model,
                                k = tree_num,
                                labelVar = TRUE) %>%
    tibble::rownames_to_column() %>%
    # make leaf split points to NA, so the 0s won't get plotted
    mutate(`split point` = ifelse(is.na(prediction), `split point`, NA))
  # prepare data frame for graph
  graph_frame <- data.frame(from = rep(tree$rowname, 2),
                            to = c(tree$`left daughter`, tree$`right daughter`))
  # convert to graph and delete the last node that we don't want to plot
  graph <- graph_from_data_frame(graph_frame) %>%
    delete_vertices("0")
  # set node labels
  V(graph)$node_label <- gsub("_", " ", as.character(tree$`split var`))
  V(graph)$leaf_label <- as.character(tree$prediction)
  V(graph)$split <- as.character(round(tree$`split point`, digits = 2))
  # plot
  plot <- ggraph(graph, 'dendrogram') +
    theme_bw() +
    geom_edge_link() +
    geom_node_point() +
    geom_node_text(aes(label = node_label), na.rm = TRUE, repel = TRUE) +
    geom_node_label(aes(label = split), vjust = 2.5, na.rm = TRUE, fill = "white") +
    geom_node_label(aes(label = leaf_label, fill = leaf_label), na.rm = TRUE,
                    repel = TRUE, colour = "white", fontface = "bold", show.legend = FALSE) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_blank(),
          panel.background = element_blank(),
          plot.background = element_rect(fill = "white"),
          panel.border = element_blank(),
          axis.line = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          plot.title = element_text(size = 18))
  print(plot)
}
tree_func(model_cvrf$finalModel, 1)


############
# Boosting #
############
tgrid <- expand.grid(
  .mfinal = 100,
  .maxdepth = 3,
  .coeflearn = "Breiman"
)

# 5-fold CV with ROSE
train_control <- trainControl(method = "cv", number = 5, sampling = "rose",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

# 5-fold CV with SMOTE
train_control <- trainControl(method = "cv", number = 5, sampling = "smote",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)

set.seed(1)
start <- Sys.time()
model_cvboost <- train(poi ~., data = enron_imp_train,
                    method = "AdaBoost.M1",
                    trControl = train_control, metric = "ROC",
                    tuneGrid = tgrid)
end <- Sys.time()
end-start
prediction_cvboost = predict(model_cvboost, enron_imp_test)
conf_cvboost <- confusionMatrix(prediction_cvboost, enron_imp_test$poi, positive='True')
conf_cvboost
conf_cvboost$byClass[c(5,6)]

