Customer Churn Prediction System
🔍 Project Overview

Customer churn refers to when customers stop using a company’s services. Predicting churn in advance helps businesses take preventive actions to retain valuable customers.

This project builds a machine learning / neural network–based churn prediction system that analyzes customer information and predicts whether a customer is likely to exit (churn) or stay.

🎯 Objective

To predict whether a bank customer will churn (Exited = 1) or not (Exited = 0) based on demographic, financial, and behavioral data.

🧾 Dataset Description

Target Column

Exited → 1 = Customer Churned, 0 = Customer Retained

Input Features

Feature	Description
CreditScore	Customer credit score
Geography	Customer location
Gender	Male / Female
Age	Customer age
Tenure	Number of years with bank
Balance	Account balance
NumOfProducts	Number of bank products used
HasCrCard	Credit card ownership
IsActiveMember	Active status
EstimatedSalary	Estimated annual salary
Complain	Complaint status
Satisfaction Score	Customer satisfaction level
Card Type	Type of card owned
Point Earned	Reward points earned
🧠 Model Used

Neural Network (ANN)

Input layer

Hidden layers with ReLU activation

Output layer with Sigmoid activation

Binary Classification Problem

⚙️ Technologies & Libraries

Python 🐍

NumPy

Pandas

Scikit-Learn

TensorFlow / Keras

Joblib

Gradio (for UI deployment)

🧪 Data Preprocessing

Label Encoding / One-Hot Encoding for categorical features

Standard Scaling for numerical features

Train-Test Split

Saved encoders and scalers using joblib for deployment consistency

📊 Model Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Loss & Validation Loss (for neural network)

🖥️ Web Application (Gradio)

A user-friendly Gradio web interface allows users to:

Enter customer details

Get real-time churn prediction

View results instantly

💾 Model & Preprocessor Saving
model.h5
scaler.pkl
encoder.pkl


Saved using joblib and Keras for reuse during deployment.

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install numpy pandas scikit-learn tensorflow gradio joblib

2️⃣ Train the Model
python train_model.py

3️⃣ Run the Gradio App
python app.py

📌 Output Example

Churn Prediction: Exited  / Not Exited

Probability Score: Confidence of churn

📈 Future Improvements

Hyperparameter tuning

Feature importance analysis

Model explainability (SHAP / LIME)

Cloud deployment (Gradio / streamlit)

👨‍💻 Author

Dnyaneshwar Kale
Data Analyst | Machine Learning Enthusiast
