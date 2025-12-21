# Bank Customer Retention Risk Scoring

This project focuses on predicting whether a bank customer is likely to leave the bank based on their demographic and account-related information. The goal is to demonstrate a complete machine learning inference pipeline, from data preprocessing to model prediction, wrapped inside an interactive Streamlit application.

---

## Project Overview

Customer retention is a major concern in the banking sector, as acquiring new customers is significantly more expensive than retaining existing ones. This project uses a trained Artificial Neural Network (ANN) model to estimate the probability of customer churn.

The application allows users to input customer details through a simple web interface and receive a prediction indicating whether the customer is likely to stay or leave the bank.

The emphasis of this project is on:
- Correct preprocessing at inference time
- Consistency between training and prediction pipelines
- Practical deployment using Streamlit

---

## Model Description

- Model type: Artificial Neural Network (ANN)
- Framework: TensorFlow / Keras
- Problem type: Binary classification
- Output: Probability score indicating customer churn risk

The model was trained using historical customer data and standard preprocessing techniques such as encoding and feature scaling.

---

## Technologies Used

- Python 3.10
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit

---

## Input Features

The model uses the following features for prediction:

- Credit Score  
- Geography (One-Hot Encoded)  
- Gender (Label Encoded)  
- Age  
- Tenure  
- Account Balance  
- Number of Products  
- Credit Card Availability  
- Active Membership Status  
- Estimated Salary  

---

## Application Interface

The Streamlit application provides a form-based interface where users can enter customer details and receive a churn prediction in real time.

### Screenshots / Demo Images 


<img width="953" height="773" alt="Screenshot 2025-12-21 201013" src="https://github.com/user-attachments/assets/dabf6ca3-491a-46f3-8e64-7707352c2081" />

<img width="1051" height="576" alt="Screenshot 2025-12-21 201029" src="https://github.com/user-attachments/assets/63e79d7d-ab86-48af-bf38-9ef7f3055049" />

