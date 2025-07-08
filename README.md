# Customer Churn Prediction in R

This project demonstrates how to predict customer churn using three machine learning models:
- Logistic Regression (via glmnet)
- Random Forest
- XGBoost

It uses the Telco Customer Churn dataset from Kaggle and is structured for clarity and reproducibility.

## 📁 Files

- `main_churn_prediction.R`: Complete R script containing data loading, preprocessing, model training, evaluation, and ROC plots.

## 📊 Dataset

Download the dataset from Kaggle:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV file (e.g., `WA_Fn-UseC_-Telco-Customer-Churn.csv`) in your working directory and adjust the path inside the R script if needed.

## ▶️ How to Run

1. Open RStudio.
2. Install required packages (see below).
3. Run the script: `main_churn_prediction.R`.

## 📦 Requirements

All required R packages are listed in `requirements.txt`. Install them with:

```r
install.packages(readLines("requirements.txt"))
```

## 📈 Output

- Confusion matrices (printed and visualized)
- Accuracy, F1 score, precision, and recall
- ROC curves and AUC for Logistic Regression and XGBoost

## 🧠 License

MIT License – use freely with credit.
