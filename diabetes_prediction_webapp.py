#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:33:35 2025

@author: yashkumar
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('/Users/yashkumar/Downloads/trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)  # Convert to float

    # Reshape for prediction (1 sample, multiple features)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_data_reshaped)

    # Return result instead of printing
    return "The person is Diabetic" if prediction[0] == 1 else "The person is not Diabetic"

def main():
    # Title
    st.title("Diabetes Prediction Web App")

    # Getting the input data (using number inputs instead of text input)
    Pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, value=0)
    Glucose = st.number_input("Glucose Level", min_value=0.0, step=0.1, value=0.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1, value=0.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1, value=0.0)
    Insulin = st.number_input("Insulin Level", min_value=0.0, step=0.1, value=0.0)
    BMI = st.number_input("BMI", min_value=0.0, step=0.1, value=0.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01, value=0.0)
    Age = st.number_input("Age", min_value=1, step=1, value=1)

    # Diagnosis result
    diagnosis = ""

    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                         Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)  # Display result

if __name__ == '__main__':
    main()