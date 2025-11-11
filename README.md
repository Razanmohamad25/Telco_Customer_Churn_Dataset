# Telco Customer Churn Prediction
This repository contains a machine learning solution designed to predict customer churn in the telecommunications industry, leveraging the Telco Customer Churn dataset. Churn prediction helps identify at-risk customers, aiding retention efforts and improving business profitability.

# Overview
The project utilizes advanced preprocessing, feature, and model selection techniques to analyze customer demographics, account information, service usage, and churn outcomes. Both baseline and tuned Random Forest models are evaluated to maximize predictive performance and business relevance.

# Dataset
Source:
```WA_Fn-UseC_-Telco-Customer-Churn.csv```

Contains customer ID, demographics, account data, service usage details, and churn label.

Target Variable: Churn (Yes/No)

# Workflow
Data Exploration & Preprocessing

Inspect data shape, missing values, and summary statistics.

Encode categorical features.

Scale numerical features using StandardScaler.

Address class imbalance via RandomOverSampler.

# Modeling

Train baseline Random Forest Classifier.

Tune hyperparameters with Bayesian optimization (BayesSearchCV).

# Evaluation

Assess metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

Confusion matrix visualization.

Compare baseline and optimized models for improvement.

# Project Structure
```
Telco_Customer_Churn_Dataset/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── churn_notebook.ipynb                   # Main analysis and modeling notebook
├── README.md                              # Project documentation
WA_Fn-UseC_-Telco-Customer-Churn.csv: Source data file for model training and evaluation.
```
churn_notebook.ipynb: Contains complete  workflow—EDA, preprocessing, modeling, evaluation, and visualization.

README.md: Overview, instructions, project details.

# Installation
Make sure you have the required dependencies installed:
```
bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn scikit-optimize
```
# Usage
Clone the repository.

Open and run churn_notebook.ipynb in Jupyter Notebook.

Follow the workflow from data loading to model evaluation; all code and visualizations are included.

