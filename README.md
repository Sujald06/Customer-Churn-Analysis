🚀 Customer Churn Analysis — EDA & Predictive Modeling
📘 Overview

Customer churn (or customer attrition) occurs when customers stop doing business with a company. Understanding and predicting churn is crucial for retaining valuable customers and improving business profitability.

This project focuses on Exploratory Data Analysis (EDA) and machine learning modeling to uncover insights and predict customer churn. Through visualization, feature engineering, and advanced classification algorithms, we identify key factors driving churn and evaluate models to achieve optimal prediction performance.

📊 Dataset

Source: Kaggle - Telco Customer Churn Dataset

Key Features:

Feature	Description
customerID	Unique customer identifier
gender	Gender of the customer
SeniorCitizen	Indicates if the customer is a senior citizen
Partner, Dependents	Family-related attributes
tenure	Number of months the customer has stayed with the company
PhoneService, MultipleLines	Phone service details
InternetService, OnlineSecurity, DeviceProtection, etc.	Internet service details
Contract, PaymentMethod, PaperlessBilling	Customer’s contract and billing preferences
MonthlyCharges, TotalCharges	Financial data
Churn	Target variable — whether the customer left (Yes/No)

The target variable for this analysis is Churn.

🧩 Project Workflow

Data Cleaning & Preprocessing

Handled missing values and data inconsistencies

Encoded categorical variables

Engineered new features (e.g., ratios, interaction terms)

Exploratory Data Analysis (EDA)

Visualized churn distribution and correlations

Analyzed demographic and service-based churn patterns

Feature Scaling

Applied StandardScaler to normalize numerical features

Model Development

Implemented multiple classification algorithms:

Logistic Regression

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

XGBoost

Gradient Boosting

Model Evaluation

Evaluated performance using Accuracy, Recall, F1-Score, and ROC-AUC

Addressed class imbalance with SMOTE (Synthetic Minority Oversampling Technique)

⚙️ Requirements

Make sure to install the following dependencies before running the notebook:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

📈 Results
Model	Accuracy	Recall	F1 Score	ROC AUC
Logistic Regression	0.7037	0.6832	0.4730	0.7641
Random Forest	0.8620	0.4144	0.5390	0.8524
K-Nearest Neighbors	0.7523	0.6678	0.5121	0.7766
Support Vector Machine	0.7857	0.6627	0.5462	0.8225
XGBoost	0.8330	0.6096	0.5870	0.8418
Gradient Boosting	0.8170	0.7003	0.5984	0.8598
🏆 Model Insights

Gradient Boosting achieved the best balance between precision and recall with the highest F1 and ROC AUC scores.

XGBoost followed closely, demonstrating strong predictive power.

Random Forest achieved high accuracy but struggled with recall, indicating bias toward the majority class.

💡 Key Insights

Senior citizens and customers with month-to-month contracts exhibit higher churn rates.

Customers with fiber optic internet and high monthly charges are more likely to leave.

Long-tenure customers and those with automatic payment methods have higher retention.

🧠 Learnings

Handling imbalanced datasets with SMOTE significantly improves minority class detection.

Ensemble models (XGBoost, Gradient Boosting) outperform traditional classifiers for churn prediction.

Feature engineering and careful preprocessing are key to improving model performance.

📚 Tools & Technologies

Programming: Python

Libraries: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, XGBoost, Imbalanced-learn

Techniques: EDA, Feature Engineering, Model Evaluation, SMOTE, ROC Analysis

🏁 Conclusion

The project successfully built and evaluated several machine learning models to predict customer churn.
Gradient Boosting proved to be the most effective, delivering strong performance metrics and actionable insights to guide customer retention strategies.

🧩 Author

👤 Sujal D.
💼 Data Science Enthusiast | Machine Learning Practitioner
🔗 Portfolio
 • LinkedIn
 • GitHub