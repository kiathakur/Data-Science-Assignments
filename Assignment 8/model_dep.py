#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load trained model and scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
std_sca = pickle.load(open("scaler.pkl", "rb"))


# In[3]:


st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):

    if glucose == 0 or bmi == 0:
        st.warning("Glucose and BMI must be greater than 0")
    else:
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = std_sca.transform(input_data)

        pred_prob = model.predict_proba(input_scaled)
        diabetes_prob = pred_prob[0][1]

        st.write(f" Diabetes Probability: **{diabetes_prob:.2%}**")

        if diabetes_prob >= 0.5:
            st.error(" High Risk of Diabetes")
        else:
            st.success("Low Risk of Diabetes")


# In[ ]:




