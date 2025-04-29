# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Import libraries
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('save_model/heart_disease_model.joblib')  # Make sure this path is correct

# App Title
st.title('Heart Disease Prediction App')

# User Input Form
st.header('Patient Data')
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
cp = st.selectbox('Chest Pain Type (cp)', options=[0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=50, max_value=250, value=120)
chol = st.number_input('Serum Cholesterol in mg/dl (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', options=[0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=50, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', options=[0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', value=1.0)
slope = st.selectbox('Slope of Peak Exercise ST Segment (slope)', options=[0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', options=[0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (thal)', options=[0, 1, 2, 3])

# Prediction Button
if st.button('Predict'):
    # Create feature array for prediction
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Output
    if prediction[0] == 1:
        st.error('⚠️ The patient is likely to have heart disease.')
    else:
        st.success('✅ The patient is unlikely to have heart disease.')

