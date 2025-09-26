# LOGIT MODEL - 1

# Packages
library(writexl)
library(readxl)
library(dplyr)
library(purrr)
library(fixest)
library(ggplot2)

setwd("C:/Users/saksh/Downloads") # Set file directory

pdata_final <- read_excel("./final-panel-ashley.xlsx", sheet = "Sheet1",
                          col_names = TRUE, na = "NaN")

# List of all variables
all_vars <- names(pdata_final)

# Variables you want to exclude from regression
vars_to_exclude <- c("systemic_crisis", "country_clean", "year",
                     "general_govt_net_lendingborrowing_._of_gdp",
                     "general_govt_revenue_._of_gdp", 
                     "general_govt_total_expenditure_._of_gdp",
                     "income_group", "domestic_debt_in_default")

# Create list of independent variables
independent_vars <- setdiff(all_vars, vars_to_exclude)

# Build the regression formula
formula <- as.formula(
  paste("systemic_crisis ~", paste(independent_vars, collapse = " + "))
)

# Run the logistic regression with fixed effects(panel structure) and clustered SEs
logit_panel <- feglm(
  formula,
  data = pdata_final,
  family = binomial(link = "logit"),
  cluster = ~country_clean,
  fixef = "year" # <- Cluster SEs at the country level
)

# View summary
summary(logit_panel)


# RANDOM FOREST BASIC - 2

# Load required packages
library(readxl)
library(dplyr)
library(randomForest)
library(caret)
library(pROC)   # for ROC-AUC curve

# 1. Set working directory and read data
setwd("C:/Users/saksh/Downloads") 
pdata_final <- read_excel("final-panel-ashley.xlsx", sheet = "Sheet1",
                          col_names = TRUE, na = "NaN")

# 2. Preprocessing
# Remove unnecessary columns
pdata_rf <- pdata_final %>%
  select(-country_clean, -year, -income_group)  # Drop ID, Year, Income group

# Make sure 'systemic_crisis' is a factor (classification target)
pdata_rf$systemic_crisis <- as.factor(pdata_rf$systemic_crisis)

# 3. Split data into Train and Test
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(pdata_rf$systemic_crisis, p = 0.8, list = FALSE)
train_data <- pdata_rf[trainIndex, ]
test_data <- pdata_rf[-trainIndex, ]

# 4. Train Random Forest Model
rf_model <- randomForest(systemic_crisis ~ ., 
                         data = train_data,
                         ntree = 500,        # number of trees
                         mtry = 4,           # number of variables randomly sampled at each split
                         importance = TRUE)  # get variable importance

# 5. View Model Summary
print(rf_model)

# 6. Predict on Test Set
rf_preds <- predict(rf_model, newdata = test_data)

# 7. Confusion Matrix
confusionMatrix(rf_preds, test_data$systemic_crisis)

# 8. ROC Curve and AUC
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[,2]  # get probabilities for class '1'

roc_curve <- roc(test_data$systemic_crisis, rf_probs)
plot(roc_curve, main = "ROC Curve - Random Forest")
auc(roc_curve)  # print AUC value

# 9. Variable Importance Plot
varImpPlot(rf_model, main = "Variable Importance (Random Forest)")


# RANDOM FOREST TUNED WITH CARET (USING 5-FOLD CLASSIFICATION)

# Load required packages
library(readxl)
library(dplyr)
library(caret)
library(randomForest)
library(pROC)

# 1. Set working directory and read data
setwd("C:/Users/saksh/Downloads")
pdata_final <- read_excel("final-panel-ashley.xlsx", sheet = "Sheet1",
                          col_names = TRUE, na = "NaN")

# 2. Preprocessing
pdata_rf <- pdata_final %>%
  select(-country_clean, -year, -income_group)

# Correct factor levels
pdata_rf$systemic_crisis <- factor(pdata_rf$systemic_crisis, 
                                   levels = c(0, 1),
                                   labels = c("No", "Yes"))

# 3. Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(pdata_rf$systemic_crisis, p = 0.8, list = FALSE)
train_data <- pdata_rf[trainIndex, ]
test_data <- pdata_rf[-trainIndex, ]

# 4. Define Training Control
control <- trainControl(method = "cv",           # Cross-validation
                        number = 5,               # 5-fold CV
                        search = "grid",          # Grid search
                        classProbs = TRUE,        # Needed for AUC
                        summaryFunction = twoClassSummary)  # Optimize for ROC

# 5. Define Tuning Grid
tuneGrid <- expand.grid(mtry = seq(2, 10, by = 1))  # Try different values for mtry

# 6. Train Random Forest Model with Tuning
set.seed(123)
rf_tuned <- train(systemic_crisis ~ ., 
                  data = train_data, 
                  method = "rf",
                  metric = "ROC",       # optimize based on AUC (not accuracy)
                  tuneGrid = tuneGrid,
                  trControl = control,
                  ntree = 500)

# 7. View Best Model
print(rf_tuned)
plot(rf_tuned)  # Plot performance vs mtry

# 8. Best hyperparameter
rf_tuned$bestTune

# 9. Predict on Test Set
rf_preds_tuned <- predict(rf_tuned, newdata = test_data)
confusionMatrix(rf_preds_tuned, test_data$systemic_crisis)

# 10. ROC Curve and AUC for Tuned Model
rf_probs_tuned <- predict(rf_tuned, newdata = test_data, type = "prob")[,2]

roc_curve_tuned <- roc(test_data$systemic_crisis, rf_probs_tuned)
plot(roc_curve_tuned, main = "ROC Curve - Tuned Random Forest")
auc(roc_curve_tuned)



# NEURAL NETWORK

# Install if needed
# install.packages("nnet")

# Load libraries
library(readxl)
library(dplyr)
library(caret)
library(nnet)
library(pROC)

# 1. Read Data
setwd("C:/Users/saksh/Downloads")
pdata_final <- read_excel("final-panel-ashley.xlsx", sheet = "Sheet1",
                          col_names = TRUE, na = "NaN")

# 2. Preprocessing
pdata_nn <- pdata_final %>%
  select(-country_clean, -year, -income_group)

# Correct factor levels
pdata_nn$systemic_crisis <- factor(pdata_nn$systemic_crisis,
                                   levels = c(0, 1),
                                   labels = c("No", "Yes"))

# 3. Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(pdata_nn$systemic_crisis, p = 0.8, list = FALSE)
train_data <- pdata_nn[trainIndex, ]
test_data <- pdata_nn[-trainIndex, ]

# 4. Train Neural Network Model
set.seed(123)
nn_model <- nnet(systemic_crisis ~ ., 
                 data = train_data,
                 size = 5,            # 5 neurons in the hidden layer
                 decay = 0.01,        # weight decay to prevent overfitting
                 maxit = 500,         # maximum number of iterations
                 trace = FALSE)       # do not print steps while training

# 5. Predict on Test Set
nn_preds <- predict(nn_model, newdata = test_data, type = "class")

# Important: Convert predictions to factor with correct levels
nn_preds <- factor(nn_preds, levels = c("No", "Yes"))

# 6. Confusion Matrix
confusionMatrix(nn_preds, test_data$systemic_crisis)

# 7. ROC Curve and AUC
nn_probs <- predict(nn_model, newdata = test_data, type = "raw")  # probabilities

roc_curve_nn <- roc(test_data$systemic_crisis, as.numeric(nn_probs))
plot(roc_curve_nn, main = "ROC Curve - Neural Network")
auc(roc_curve_nn)




# SVM (Support Vector Machine)

install.packages("kernlab")

# Load libraries
library(readxl)
library(dplyr)
library(caret)
library(kernlab)   # Correct package for SVM with caret
library(pROC)

# 1. Read Data
setwd("C:/Users/saksh/Downloads")
pdata_final <- read_excel("final-panel-ashley.xlsx", sheet = "Sheet1",
                          col_names = TRUE, na = "NaN")

# 2. Preprocessing
pdata_svm <- pdata_final %>%
  select(-country_clean, -year, -income_group)

# Correct factor levels
pdata_svm$systemic_crisis <- factor(pdata_svm$systemic_crisis,
                                    levels = c(0, 1),
                                    labels = c("No", "Yes"))

# 3. Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(pdata_svm$systemic_crisis, p = 0.8, list = FALSE)
train_data <- pdata_svm[trainIndex, ]
test_data <- pdata_svm[-trainIndex, ]

# 4. Train SVM Model (Radial Kernel)
set.seed(123)

# Define training control
control <- trainControl(method = "cv", 
                        number = 5,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)

# Train model
svm_model <- train(systemic_crisis ~ ., 
                   data = train_data,
                   method = "svmRadial",   # SVM with RBF kernel
                   trControl = control,
                   metric = "ROC",
                   tuneLength = 5)  # Try 5 different hyperparameter sets

# 5. View Model Summary
print(svm_model)
plot(svm_model)

# 6. Predict on Test Set
svm_preds <- predict(svm_model, newdata = test_data)

# Important: Ensure predictions are factor
svm_preds <- factor(svm_preds, levels = c("No", "Yes"))

# Confusion Matrix
confusionMatrix(svm_preds, test_data$systemic_crisis)

# 7. ROC Curve and AUC
svm_probs <- predict(svm_model, newdata = test_data, type = "prob")[,2]

roc_curve_svm <- roc(test_data$systemic_crisis, svm_probs)
plot(roc_curve_svm, main = "ROC Curve - SVM")
auc(roc_curve_svm)


# XGBoost

# Load libraries
library(readxl)
library(dplyr)
library(caret)
library(xgboost)
library(pROC)

# 1. Read Data
setwd("C:/Users/saksh/Downloads")
pdata_final <- read_excel("final-panel-ashley.xlsx", sheet = "Sheet1",
                          col_names = TRUE, na = "NaN")

# 2. Preprocessing
pdata_xgb <- pdata_final %>%
  select(-country_clean, -year, -income_group)

# Correct factor levels
pdata_xgb$systemic_crisis <- factor(pdata_xgb$systemic_crisis,
                                    levels = c(0, 1),
                                    labels = c("No", "Yes"))

# 3. Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(pdata_xgb$systemic_crisis, p = 0.8, list = FALSE)
train_data <- pdata_xgb[trainIndex, ]
test_data <- pdata_xgb[-trainIndex, ]

# 4. Train XGBoost Model
set.seed(123)

# Define training control
control <- trainControl(method = "cv", 
                        number = 5,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)

# Train model
xgb_model <- train(systemic_crisis ~ ., 
                   data = train_data,
                   method = "xgbTree",       # XGBoost Tree method
                   trControl = control,
                   metric = "ROC",            # Optimize based on AUC
                   tuneLength = 5)            # Try 5 different hyperparameter sets

# 5. View Model Summary
print(xgb_model)
plot(xgb_model)

# 6. Predict on Test Set
xgb_preds <- predict(xgb_model, newdata = test_data)

# Important: Ensure predictions are factor
xgb_preds <- factor(xgb_preds, levels = c("No", "Yes"))

# Confusion Matrix
confusionMatrix(xgb_preds, test_data$systemic_crisis)

# 7. ROC Curve and AUC
xgb_probs <- predict(xgb_model, newdata = test_data, type = "prob")[,2]

roc_curve_xgb <- roc(test_data$systemic_crisis, xgb_probs)
plot(roc_curve_xgb, main = "ROC Curve - XGBoost")
auc(roc_curve_xgb)
