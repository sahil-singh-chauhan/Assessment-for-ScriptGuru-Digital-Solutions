# Churn Prediction Project

## Overview
This project implements a **Customer Churn Prediction** system for a telecom company. The goal is to identify customers who are likely to churn, enabling targeted retention strategies. The analysis leverages **Logistic Regression** (baseline and hyperparameter-tuned) and **Random Forest** models for comparison.

---

## Dataset
- **Source:** Telco Customer Churn dataset  
- **Rows:** 7,043  
- **Columns:** 21  
- **Key Columns:** `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `InternetService`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`

---

## Steps Performed

### 1. Data Cleaning & Preprocessing
- Converted `TotalCharges` to numeric and imputed missing values.  
- Removed rows with zero `tenure` as they are non-informative for churn modeling.  
- Encoded categorical features using label encoding.  
- Checked feature distributions and correlations with churn.

### 2. Exploratory Data Analysis (EDA)
- **Numeric distributions:** Tenure, MonthlyCharges, TotalCharges  
- **Categorical distributions:** Gender, Contract type, Internet service, etc.  
- **Churn analysis:** Identified features contributing most to churn.  
- Visualizations: histograms, bar plots, boxplots, heatmaps.

### 3. Baseline Model: Logistic Regression V1
- Simple logistic regression as a baseline.  
- **Performance:**  
  - AUC: 0.83  
  - Precision (Churn): Moderate  
  - Recall (Churn): Moderate  
- **Top Features:** PhoneService, Contract, PaperlessBilling, OnlineSecurity, TechSupport

### 4. Hyperparameter-Tuned Model: Logistic Regression V2
- Applied **GridSearchCV** to tune:
  - `penalty`: l1, l2  
  - `solver`: liblinear  
  - `class_weight`: None, balanced  
- Increased `max_iter` to ensure convergence.  
- Handled class imbalance more effectively.  
- **Performance:**  
  - AUC: 0.8348  
  - Precision (Churn): Moderate  
  - Recall (Churn): Moderate  

### 5. Random Forest Comparison
- Evaluated as a more complex model for benchmark.  
- Slightly higher precision for churn, but comparable AUC and recall.  

---

## Key Insights
- Customers with **higher monthly charges** and **paperless billing** are more likely to churn.  
- Long-tenure customers are less likely to churn, but short-tenure customers require close monitoring.  
- Hyperparameter tuning improved convergence and model stability but did not drastically change performance metrics.  
- **Recall is critical** for churn prediction since missing actual churners can result in revenue loss.

---

## Usage
1. Load dataset into `df`.  
2. Preprocess data as per notebook steps (convert, encode, remove zero-tenure).  
3. Train models:
   - Baseline Logistic Regression  
   - Hyperparameter-tuned Logistic Regression  
   - Random Forest (optional)  
4. Evaluate using AUC, precision, recall, F1-score, confusion matrix, ROC curve.  
5. Use `churn_probability` column for targeted retention campaigns.

---

## Dependencies
- Python >= 3.8  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  

---

## Authors
- Sahil Singh Chauhan  

---
