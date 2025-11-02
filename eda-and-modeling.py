#!/usr/bin/env python
# coding: utf-8
# ============================================
# 📊 Telco Customer Churn Analysis & Modeling
# ============================================

# ============== IMPORTS =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

import warnings
warnings.filterwarnings('ignore')

# ============================================
# 🧮 LOAD DATA
# ============================================
df = pd.read_csv('Telco-Customer-Churn.csv')
print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

# ============================================
# 🧹 DATA CLEANING
# ============================================
# Clean 'TotalCharges' column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop customerID (not useful for prediction)
df = df.drop(columns=['customerID'])

# ============================================
# 📊 EXPLORATORY DATA ANALYSIS
# ============================================
plt.figure(figsize=(4,4))
churn_counts = df['Churn'].value_counts()
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Customer Churn Distribution')
plt.show()

# ============================================
# ⚙️ ENCODING CATEGORICAL FEATURES
# ============================================
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ============================================
# ✂️ TRAIN TEST SPLIT
# ============================================
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ============================================
# 🔄 SCALING NUMERIC FEATURES
# ============================================
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ============================================
# 🤖 MODEL TRAINING
# ============================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'K-Nearest Neighbors': make_pipeline_imb(SMOTE(random_state=42), KNeighborsClassifier()),
    'Support Vector Machine': make_pipeline_imb(SMOTE(random_state=42), SVC(probability=True, random_state=42)),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gradient Boosting': make_pipeline_imb(SMOTE(random_state=42), GradientBoostingClassifier(random_state=42))
}

results = []

lb = LabelBinarizer().fit(y_train)

for name, model in models.items():
    print(f"\n🔹 Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(lb.transform(y_test), model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else np.nan

    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 3),
        'Recall': round(recall, 3),
        'F1 Score': round(f1, 3),
        'ROC AUC': round(roc_auc, 3)
    })

# ============================================
# 📈 RESULTS COMPARISON
# ============================================
results_df = pd.DataFrame(results)
print("\n===== Model Performance Summary =====\n")
print(results_df.sort_values(by='ROC AUC', ascending=False))

plt.figure(figsize=(10,5))
sns.barplot(data=results_df.melt(id_vars='Model', value_vars=['Accuracy','Recall','F1 Score','ROC AUC']),
            x='Model', y='value', hue='variable')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.show()

# ============================================
# ✅ BEST MODEL SELECTION
# ============================================
best_model = results_df.sort_values(by='ROC AUC', ascending=False).iloc[0]
print(f"\n🏆 Best Model: {best_model['Model']} (ROC AUC: {best_model['ROC AUC']})")
