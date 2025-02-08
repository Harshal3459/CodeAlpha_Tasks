# -*- coding: utf-8 -*-
"""Credit Score Prediction using Random Forest in Google Colab

This script loads a cleaned credit score dataset, processes categorical variables,
trains a Random Forest Classifier, evaluates its performance, and defines a function
to predict creditworthiness based on user input.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from google.colab import drive

# Mount Google Drive to access dataset
drive.mount('/content/drive')

# Define dataset path
dataset_path = "/content/drive/My Drive/Datasets/Credit_Score_Clean.csv"

# Load dataset
df = pd.read_csv(dataset_path)
print(df.head())  # Display first few rows

# Check for missing values and drop them
df = df.dropna()

# Encode categorical variables into numerical values
df['Occupation'] = df['Occupation'].astype('category').cat.codes
# Convert categorical columns into numerical categories
df['Credit_Mix'] = df['Credit_Mix'].astype('category').cat.codes
df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].astype('category').cat.codes
df['Payment_Behaviour'] = df['Payment_Behaviour'].astype('category').cat.codes

# Map target variable 'Credit_Score' to numerical values
credit_score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
df['Credit_Score'] = df['Credit_Score'].map(credit_score_mapping)

# Separate features and target variable
X = df.drop(columns=['Credit_Score'])  # Features
y = df['Credit_Score']  # Target variable

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate model performance
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot()
plt.show()

# Define a function to predict creditworthiness based on user input
def predict_credit_worthiness(age, occupation, annual_income, num_bank_accounts, num_credit_card,
                              interest_rate, num_of_loan, delay_from_due_date, num_of_delayed_payment,
                              changed_credit_limit, num_credit_inquiries, credit_mix, outstanding_debt,
                              credit_utilization_ratio, payment_of_min_amount, total_emi_per_month,
                              amount_invested_monthly, payment_behaviour, monthly_balance, credit_history_age_months,
                              model=best_model):
    """
    Predicts the creditworthiness score based on user input features.
    """
    # Create a DataFrame with user input
    input_data = pd.DataFrame([[
        age, occupation, annual_income, num_bank_accounts, num_credit_card,
        interest_rate, num_of_loan, delay_from_due_date, num_of_delayed_payment,
        changed_credit_limit, num_credit_inquiries, credit_mix, outstanding_debt,
        credit_utilization_ratio, payment_of_min_amount, total_emi_per_month,
        amount_invested_monthly, payment_behaviour, monthly_balance, credit_history_age_months
    ]], columns=X.columns)

    # Predict credit score
    credit_score = model.predict(input_data)[0]
    return credit_score

# Test the function with sample input
predicted_score = predict_credit_worthiness(
    age=30, occupation=1, annual_income=50000, num_bank_accounts=2, num_credit_card=3,
    interest_rate=5, num_of_loan=2, delay_from_due_date=1, num_of_delayed_payment=0,
    changed_credit_limit=10, num_credit_inquiries=2, credit_mix=1, outstanding_debt=10000,
    credit_utilization_ratio=0.3, payment_of_min_amount=0, total_emi_per_month=200,
    amount_invested_monthly=300, payment_behaviour=2, monthly_balance=1000, credit_history_age_months=120
)

print(f"Predicted Credit Worthiness Score: {predicted_score}")
