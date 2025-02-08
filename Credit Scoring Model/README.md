# Credit Score Prediction using Random Forest

## Overview
This project implements a credit score prediction model using a Random Forest Classifier. The model is trained on a cleaned credit score dataset and evaluates various customer financial features to predict creditworthiness.

## Dataset
The dataset used for training the model is sourced from Kaggle: [Credit Score Classification Clean Data](https://www.kaggle.com/datasets/eneskosar19/credit-score-classification-clean-data).

## Features
The dataset contains various financial indicators such as:
- Age
- Occupation
- Annual Income
- Number of Bank Accounts
- Number of Credit Cards
- Interest Rate
- Number of Loans
- Payment Behavior
- Outstanding Debt
- Credit Utilization Ratio
- Credit Mix
- Monthly Balance
- Credit History Age

## Model
- **Algorithm:** Random Forest Classifier
- **Performance Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

## Installation
To run this project in Google Colab:
1. Mount Google Drive to access the dataset.
2. Install required libraries if not already installed.
3. Run the script to train and evaluate the model.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage
The script includes a function to predict creditworthiness based on user input:
```python
def predict_credit_worthiness(age, occupation, annual_income, num_bank_accounts, ...):
    # Predict credit score
    return model.predict(input_data)[0]
```

Example:
```python
predicted_score = predict_credit_worthiness(
    age=30, occupation=1, annual_income=50000, num_bank_accounts=2, num_credit_card=3,
    interest_rate=5, num_of_loan=2, delay_from_due_date=1, num_of_delayed_payment=0,
    changed_credit_limit=10, num_credit_inquiries=2, credit_mix=1, outstanding_debt=10000,
    credit_utilization_ratio=0.3, payment_of_min_amount=0, total_emi_per_month=200,
    amount_invested_monthly=300, payment_behaviour=2, monthly_balance=1000, credit_history_age_months=120
)
print(f"Predicted Credit Worthiness Score: {predicted_score}")
```

## License
This project is open-source and free to use. Check the dataset's license on Kaggle before usage.

## Author
Developed in Google Colab using Python, Pandas, Scikit-learn, and Matplotlib.
