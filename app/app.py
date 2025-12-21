## streamlit app
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


## loading the trained model
model = tf.keras.models.load_model('../retention_risk_scoring_model.keras', compile=False)

with open("../label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("../onehot_encoder_geography.pkl", "rb") as f:
    onehot_encoder_geography = pickle.load(f)

with open("../scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Bank Customer Retention Risk Scoring Model")

## user inputs
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=1)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


## preparing the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]})


## converting categorical features into numerical
geo_encoded = onehot_encoder_geography.transform(input_data[["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(["Geography"])) 

## combining the encoded features with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data = input_data.drop("Geography", axis=1)
input_data = input_data.reindex(columns=scaler.feature_names_in_)
## scaling the features
input_data_scaled = scaler.transform(input_data)

## making predictions
predictions = model.predict(input_data_scaled)
predictions_probability = predictions[0][0]

st.write(f"Churn Probability: {predictions_probability:.2f}")
if predictions_probability > 0.5:
    st.write("The customer is likely to leave the bank.")
else:
    st.write("The customer is likely to stay with the bank.")

