import streamlit as st
import yaml
import pandas as pd
import numpy as np
from src.heart_disease.prediction import predict_with_preprocessing  # Import your prediction function

# Load parameters from params.yaml
pred = yaml.safe_load(open("params.yaml"))["prediction"]
MODEL_PATH = pred["model"]
TRAINING_DATA_PATH = pred["data"]

# Load training data for feature detection (once, when app starts)
training_data_sample = pd.read_csv(TRAINING_DATA_PATH)

# App title and author
st.title("Heart Disease Prediction")
st.subheader("By Aradhya Patel")
st.write("Provide the required details to predict the likelihood of heart disease.")

# Define precise ranges for sliders (adjust as needed based on dataset or domain knowledge)
slider_ranges = {
    "age": (29, 77),
    "trestbps": (94, 200),
    "chol": (126, 564),
    "thalach": (71, 202),
    "oldpeak": (0.0, 6.2),
}

# User input fields
age = st.slider("Enter Age", min_value=29, max_value=77, value=50)
sex = st.selectbox("Select Gender", options=["Male", "Female"], index=0)
sex = 1 if sex == "Male" else 0  # Male -> 1, Female -> 0
cp = st.slider("Enter Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
trestbps = st.slider("Enter Resting Blood Pressure", min_value=94, max_value=200, value=120)
chol = st.slider("Enter Cholesterol Level", min_value=126, max_value=564, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], index=0)
restecg = st.slider("Resting ECG Results", min_value=0, max_value=2, value=0)
thalach = st.slider("Enter Maximum Heart Rate Achieved", min_value=71, max_value=202, value=150)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", options=[0, 1], index=0)
oldpeak = st.slider("Enter ST Depression Value", min_value=0.0, max_value=6.2, value=1.0)
slope = st.slider("Enter Slope of Peak Exercise ST Segment", min_value=0, max_value=2, value=1)
ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
thal = st.selectbox("Enter Thal Value (3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect)", options=[3, 6, 7], index=0)

# Convert user inputs into a NumPy array
user_inputs_array = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,0])

# Check the shape and the length of the input array
# st.write(f"Input feature array length: {len(user_inputs_array)}")
# st.write(f"Input feature array: {user_inputs_array}")

# Prediction button
if st.button("Predict"):
    try:
        # Check if the input has the correct number of features
        expected_num_features = 14  # Adjust based on the actual model's expected number of features
        if len(user_inputs_array) != expected_num_features:
            st.error(f"Error: Expected {expected_num_features} features, but received {len(user_inputs_array)}.")
        else:
            # Call prediction function
            prediction = predict_with_preprocessing(user_inputs_array, MODEL_PATH, training_data_sample)

            # Friendly message based on prediction
            if prediction == 0:
                st.success("The prediction is: **No Heart Disease**")
            elif prediction == 1:
                st.error("The prediction is: **Heart Disease**")
            else:
                st.warning(f"The prediction returned: {prediction} (unexpected result)")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
