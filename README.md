# Telco Customer Churn Prediction

This Model analyzes the **Telco Customer Churn dataset** to predict whether a customer will churn (leave the service) or not. The notebook walks through preprocessing, model building, tuning, and evaluation.

---

## 📂 Dataset
- **Source**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Contains customer demographics, account information, service usage, and churn label.
- Target variable: **Churn** (Yes/No)

---

## ⚙️ Workflow
1. **Data Exploration & Preprocessing**
   - Load dataset and inspect shape, missing values, and summary statistics.
   - Encode categorical features.
   - Scale numerical features using `StandardScaler`.
   - Handle class imbalance with **RandomOverSampler**.

2. **Modeling**
   - Train a baseline **Random Forest Classifier**.
   - Hyperparameter tuning with **BayesSearchCV** for optimized Random Forest.

3. **Evaluation**
   - Metrics: **Accuracy, Precision, Recall, F1-score, ROC-AUC**.
   - Confusion matrix visualization.
   - Comparison between baseline and tuned Random Forest models.

---

## 📊 Results
- Baseline Random Forest trained and evaluated.
- Tuned Random Forest achieved improved performance using **Bayesian optimization**.
- Model comparison showed significant improvements in accuracy and F1-score after tuning.

---

## 🚀 Requirements
Install the required dependencies before running the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn scikit-optimize
