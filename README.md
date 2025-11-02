<h1 align="center">🚀 Customer Churn Analysis — EDA & Predictive Modeling</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=yellow" />
  <img src="https://img.shields.io/badge/ML-ScikitLearn-orange.svg?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Visualization-Seaborn%20|%20Matplotlib-blueviolet.svg?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-Kaggle-lightgrey.svg?style=for-the-badge&logo=kaggle" />
</p>

<p align="center">
  📊 Predicting customer churn using data-driven insights, EDA, and advanced machine learning models.
</p>

---

## 📘 Overview

Customer churn (or attrition) refers to the loss of customers over time.  
Understanding and predicting churn is crucial for improving **retention strategies**, **profitability**, and **customer satisfaction**.

This project performs **Exploratory Data Analysis (EDA)** and builds **Machine Learning models** on the **Telco Customer Churn Dataset** to identify the key drivers of churn and predict at-risk customers.

---

## 📊 Dataset

**Source:** [Kaggle – Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Key Features

| Feature | Description |
|----------|-------------|
| `customerID` | Unique customer identifier |
| `gender` | Gender of the customer |
| `SeniorCitizen` | Indicates if the customer is a senior citizen |
| `Partner`, `Dependents` | Family-related attributes |
| `tenure` | Duration (in months) of customer relationship |
| `PhoneService`, `MultipleLines` | Phone service details |
| `InternetService`, `OnlineSecurity`, `DeviceProtection`, etc. | Internet service details |
| `Contract`, `PaymentMethod`, `PaperlessBilling` | Contract and billing preferences |
| `MonthlyCharges`, `TotalCharges` | Financial data |
| `Churn` | Target variable (Yes = Churned, No = Active) |

🎯 **Target Variable:** `Churn`

---

## 🧩 Project Workflow

### 🔹 1. Data Cleaning & Preprocessing
- Handled missing and inconsistent data (`TotalCharges`)
- Encoded categorical variables using label encoding
- Removed irrelevant columns (`customerID`)
- Scaled numerical features using **StandardScaler**

### 🔹 2. Exploratory Data Analysis (EDA)
- Visualized churn distribution and correlations
- Analyzed patterns across demographics, services, and contract types

### 🔹 3. Model Development
Implemented and compared the following classification models:
- Logistic Regression  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- XGBoost  
- Gradient Boosting  

### 🔹 4. Model Evaluation
- Metrics: **Accuracy**, **Recall**, **F1-Score**, **ROC-AUC**
- Addressed class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**

---

## ⚙️ Requirements


Before running the notebook, make sure you have the following dependencies installed:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

📈 Results
Model	Accuracy	Recall	F1 Score	ROC AUC
Logistic Regression	0.7037	0.6832	0.4730	0.7641
Random Forest	0.8620	0.4144	0.5390	0.8524
K-Nearest Neighbors	0.7523	0.6678	0.5121	0.7766
Support Vector Machine	0.7857	0.6627	0.5462	0.8225
XGBoost	0.8330	0.6096	0.5870	0.8418
🏆 Gradient Boosting	0.8170	0.7003	0.5984	0.8598
🏆 Model Insights

🌟 Gradient Boosting delivered the best balance between precision and recall.

⚡ XGBoost closely followed with strong predictive performance.

🌲 Random Forest achieved high accuracy but lower recall — indicating bias toward the majority class.

💡 Key Insights

👴 Senior citizens and customers with month-to-month contracts exhibit higher churn rates.

🌐 Customers using fiber optic internet and paying higher monthly charges are more likely to leave.

💳 Long-tenure customers with automatic payment methods tend to remain loyal.

🧠 Learnings

🧩 SMOTE significantly improved detection of minority (churn) cases.

🚀 Ensemble models like Gradient Boosting and XGBoost outperformed traditional classifiers.

⚖️ Proper feature scaling and encoding enhanced model stability and performance.

📚 Tools & Technologies
Category	Tools
Language	Python 🐍
Libraries	Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, XGBoost, Imbalanced-learn
Techniques	EDA, Feature Engineering, Model Tuning, SMOTE, ROC-AUC Analysis
🏁 Conclusion

This project demonstrates how machine learning and data analytics can predict customer churn and provide actionable insights to enhance customer retention.

💬 Gradient Boosting emerged as the top-performing model, offering both accuracy and interpretability.

👤 Author

Sujal D.
💼 Data Science Enthusiast | Machine Learning Practitioner

<p align="center"> <a href="#">🌐 Portfolio</a> • <a href="#">💼 LinkedIn</a> • <a href="#">💻 GitHub</a> </p> <p align="center"> ⭐ If you found this project useful, consider giving it a star! </p>
