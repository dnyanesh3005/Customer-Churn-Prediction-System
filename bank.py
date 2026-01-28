import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
model=tf.keras.models.load_model('churn.h5')
preprocessor = joblib.load("preprocessor.pkl")
# Load the pre-trained model


st.title("Bank Churn Prediction")
st.write("Welcome to the Bank Churn Prediction Application. This app helps you predict whether a customer will churn based on their banking data.")
st.write("Please provide the following details about the customer:")        

col1, col2,col3 = st.columns(3)
with col1:
    CreditScore = st.number_input("Credit Score", min_value=200, max_value=1000, value=600)
    Geography = st.selectbox("Geography", options=["France", "Spain", "Germany"])
    Gender = st.selectbox("Gender", options=["Male", "Female"])
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=3)
with col2:
    Balance = st.number_input("Account Balance", min_value=0.0, value=1000.0)
    NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    HasCrCard = st.selectbox("Has Credit Card", options=[0, 1])
    IsActiveMember = st.selectbox("Is Active Member", options=[0, 1])
    EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
with col3:
    Complain = st.selectbox("Has Complaints", options=[0, 1])
    Satisfaction_Score = st.number_input("Satisfaction Level (0-1)", min_value=0.0, max_value=1.0, value=0.5)
    Card_Type = st.selectbox("Card Type", options=["Diamond", "gold", "silver", "Platinum"])
    Point_Earned = st.number_input("Points Earned", min_value=0, value=1000)
    
input_df = pd.DataFrame({
    "CreditScore": [CreditScore],
    "Geography": [Geography],
    "Gender": [Gender],
    "Age": [Age],
    "Tenure": [Tenure],
    "Balance": [Balance],
    "NumOfProducts": [NumOfProducts],
    "HasCrCard": [HasCrCard],
    "IsActiveMember": [IsActiveMember],
    "EstimatedSalary": [EstimatedSalary],
    "Complain": [Complain],
    "Satisfaction_Score": [Satisfaction_Score],
    "Card_Type": [Card_Type],
    "Point_Earned": [Point_Earned]
})
# Preprocess input data
X_input = preprocessor.transform(input_df)
# Make prediction
prediction = model.predict(X_input)
churn_probability = prediction[0][0]
# Display result
if st.button("Predict Churn"):
    if churn_probability > 0.5:
        st.error(f"The customer is likely to churn with a probability of {churn_probability:.2f}")
    else:
        st.success(f"The customer is unlikely to churn with a probability of {churn_probability:.2f}")
