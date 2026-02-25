# Loan Approval Prediction — End-to-End Machine Learning Project

## Problem Statement

Financial institutions must evaluate loan applications efficiently and accurately.  
The goal of this project is to build a supervised machine learning model that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and demographic features.

This is a **binary classification problem**.

---

## Dataset Overview

The dataset contains 4,269 loan application records with the following features:

- no_of_dependents  
- education  
- self_employed  
- income_annum  
- loan_amount  
- loan_term  
- cibil_score  
- residential_assets_value  
- commercial_assets_value  
- luxury_assets_value  
- bank_asset_value  
- loan_status (Target Variable)

Target Variable:
- Approved
- Rejected

---

## Project Workflow

### 1. Data Cleaning
- Checked for missing values
- Removed irrelevant column (`loan_id`)
- Encoded categorical variables using One-Hot Encoding

### 2. Model Training
Trained and compared multiple classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### 3. Model Evaluation
Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- 5-Fold Cross Validation

Generalization gap formula:

Training Accuracy − Testing Accuracy

---

## Best Model

Random Forest achieved the highest performance:

- Testing Accuracy ≈ 98.1%
- Cross-Validation Score ≈ 98.1%
- Training Accuracy = 1.0

The small generalization gap (~1.9%) indicates strong model stability and minimal overfitting.

---

## Feature Importance Analysis

The Random Forest model shows that `cibil_score` is the most influential feature (~81% importance).

This indicates that credit score plays a dominant role in loan approval decisions.

---

## Limitations

- The model depends on historical approval decisions.
- Potential bias may exist in training data.
- Strong reliance on credit score may reduce robustness if policies change.

---

## Future Improvements

- Hyperparameter tuning (GridSearchCV)
- ROC-AUC evaluation
- Threshold optimization
- Model deployment using Flask or Streamlit

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Repository Structure

```text
loan_approval_prediction/
│
├── loan_approval_prediction_end_to_end.ipynb
├── loan_approval_dataset.csv
└── README.md
```
---

## Key Takeaways

- Built a complete ML pipeline from data cleaning to model evaluation.
- Compared multiple classification algorithms.
- Applied cross-validation to validate generalization.
- Interpreted model using feature importance analysis.

---

## Author

Habib Shah
