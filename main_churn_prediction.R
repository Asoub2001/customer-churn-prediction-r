# === 01 Install Required Packages ===
install.packages(c(
  "ggplot2", "dplyr", "e1071", "randomForest", "xgboost", "Matrix",
  "MLmetrics", "caret", "glmnet", "pROC", "readr"
))

# === 02 Load Libraries ===
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(randomForest)
library(xgboost)
library(Matrix)
library(MLmetrics)
library(pROC)

# === 03 Load and Preprocess Data ===
telco_data <- read_csv("/bin/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove rows with missing TotalCharges
telco_data <- telco_data[!is.na(telco_data$TotalCharges), ]

# Convert character columns to factors
telco_data <- telco_data %>%
  mutate_if(is.character, as.factor)
telco_data$SeniorCitizen <- as.factor(telco_data$SeniorCitizen)

# Plot class distribution
ggplot(telco_data, aes(x = Churn, fill = Churn)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Churn Class Distribution", x = "Churn", y = "Count")

# === 04 Split Dataset ===
set.seed(123)
train_index <- createDataPartition(telco_data$Churn, p = 0.8, list = FALSE)
train_data <- telco_data[train_index, ]
test_data <- telco_data[-train_index, ]

# === 05 Logistic Regression (with Cross-Validation) ===
x_train <- model.matrix(Churn ~ . -1, data = train_data)
y_train <- ifelse(train_data$Churn == "Yes", 1, 0)

cv_model <- cv.glmnet(x_train, y_train, family = "binomial")
best_lambda <- cv_model$lambda.min
lognet <- glmnet(x_train, y_train, family = "binomial")
x_test <- model.matrix(Churn ~ . -1, data = test_data)
log_preds <- predict(lognet, newx = x_test, s = best_lambda, type = "response")
log_preds_class <- factor(ifelse(log_preds > 0.5, "Yes", "No"), levels = levels(test_data$Churn))

# === 06 Random Forest ===
# Drop high-cardinality factor columns (if any)
factor_cols <- sapply(train_data, is.factor)
for (col_name in names(train_data)[factor_cols]) {
  if (nlevels(train_data[[col_name]]) > 53) {
    train_data <- train_data %>% select(-one_of(col_name))
    test_data <- test_data %>% select(-one_of(col_name))
  }
}
rf_model <- randomForest(Churn ~ ., data = train_data, ntree = 100)
rf_preds <- predict(rf_model, newdata = test_data)

# === 07 XGBoost ===
train_matrix <- model.matrix(Churn ~ . -1, data = train_data)
test_matrix <- model.matrix(Churn ~ . -1, data = test_data)
train_label <- ifelse(train_data$Churn == "Yes", 1, 0)
test_label <- ifelse(test_data$Churn == "Yes", 1, 0)

xgb_model <- xgboost(data = train_matrix, label = train_label, nrounds = 100,
                     objective = "binary:logistic", verbose = 0)
xgb_preds <- predict(xgb_model, newdata = test_matrix)
xgb_preds_class <- factor(ifelse(xgb_preds > 0.5, "Yes", "No"), levels = levels(test_data$Churn))

# === 08 Evaluation: Confusion Matrices ===
cm_log <- confusionMatrix(log_preds_class, test_data$Churn, positive = "Yes")
cm_rf  <- confusionMatrix(rf_preds, test_data$Churn, positive = "Yes")
cm_xgb <- confusionMatrix(xgb_preds_class, test_data$Churn, positive = "Yes")

# === 09 Metrics: F1, Accuracy, Precision, Recall ===
f1_log <- F1_Score(log_preds_class, test_data$Churn, positive = "Yes")
f1_rf  <- F1_Score(rf_preds, test_data$Churn, positive = "Yes")
f1_xgb <- F1_Score(xgb_preds_class, test_data$Churn, positive = "Yes")

acc_log <- mean(log_preds_class == test_data$Churn)
acc_rf  <- mean(rf_preds == test_data$Churn)
acc_xgb <- mean(xgb_preds_class == test_data$Churn)

precision_log <- Precision(log_preds_class, test_data$Churn, positive = "Yes")
precision_rf  <- Precision(rf_preds, test_data$Churn, positive = "Yes")
precision_xgb <- Precision(xgb_preds_class, test_data$Churn, positive = "Yes")

recall_log <- Recall(log_preds_class, test_data$Churn, positive = "Yes")
recall_rf  <- Recall(rf_preds, test_data$Churn, positive = "Yes")
recall_xgb <- Recall(xgb_preds_class, test_data$Churn, positive = "Yes")

# === 10 Summary Table ===
comparison_df <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  Accuracy = c(acc_log, acc_rf, acc_xgb),
  F1_Score = c(f1_log, f1_rf, f1_xgb),
  Precision = c(precision_log, precision_rf, precision_xgb),
  Recall = c(recall_log, recall_rf, recall_xgb)
)
print(comparison_df)

# === 11 Confusion Matrix Plots ===
plot_conf_matrix <- function(cm, title) {
  df <- as.data.frame(as.table(cm$table))
  colnames(df) <- c("Prediction", "Reference", "Freq")
  ggplot(df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "black") +
    geom_text(aes(label = Freq), size = 5) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal()
}
plot_conf_matrix(cm_log, "Confusion Matrix - Logistic Regression")
plot_conf_matrix(cm_rf,  "Confusion Matrix - Random Forest")
plot_conf_matrix(cm_xgb, "Confusion Matrix - XGBoost")

# === 12 F1 Score Comparison Plot ===
results <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  F1_Score = c(f1_log, f1_rf, f1_xgb)
)
ggplot(results, aes(x = Model, y = F1_Score, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "F1 Score Comparison", y = "F1 Score", x = "Model")

# === 13 ROC and AUC ===
actual_binary <- ifelse(test_data$Churn == "Yes", 1, 0)
roc_log <- roc(actual_binary, log_preds)
roc_xgb <- roc(actual_binary, xgb_preds)

auc_log <- auc(roc_log)
auc_xgb <- auc(roc_xgb)

plot(roc_log, col = "blue", main = "ROC Curves")
lines(roc_xgb, col = "red")
legend("bottomright", legend = c("Logistic", "XGBoost"), col = c("blue", "red"), lwd = 2)
cat("AUC Logistic Regression:", auc_log, "\n")
cat("AUC XGBoost:", auc_xgb, "\n")
