# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 23:10:07 2025

@author: merof
"""

import numpy as np
import pickle 
import streamlit as st
from PIL import Image


# Loading the saved model
loaded_model = pickle.load(open('C:/Users/merof/Documents/Data Science & ML/Deploying Diabetes ML Model/trained_model.sav', 'rb'))

# Loading the saved scaler
loaded_scaler = pickle.load(open('C:/Users/merof/Documents/Data Science & ML/Deploying Diabetes ML Model/scaler.sav', 'rb'))


# creating a function for prediction

def diabetes_prediction(input_data):
    
    input_as_np = np.asarray(input_data)
    input_reshaped = input_as_np.reshape(1, -1)
    # because in the training we've scaled the data
    input_scaled = loaded_scaler.transform(input_reshaped)

    pred = loaded_model.predict(input_scaled)
    print(pred)

    if(pred[0] == 0):
        return 'The person is Non-Diabetic'
    else:
        return 'The person is Diabetic'

      


def main():
    
    
    image = Image.open('C:/Users/merof/Documents/Data Science & ML/Deploying Diabetes ML Model/diabetes_pic.png.png')  
    st.image(image, use_container_width=True)  # Adjust width as needed
    
    # Giving a title 
    st.title('Diabetes Prediction Web App')
    
    
    # Getting the input data from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness (in mm)')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI (Body Mass Index)')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')
    
    
    # The code for Prediction
    diagnosis = ''
    
    # Creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    
    st.success(diagnosis)
    
    # Adding developer credit
    st.markdown("---")

    st.markdown("### Developed by Fares Yassen 👨‍💻") 
    
    
if __name__ == '__main__':
    main()
    
    