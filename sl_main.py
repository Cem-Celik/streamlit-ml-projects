#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore")
from statistics import mode
import joblib

# In[2]:


import mobile as mob
import car
import telco
import ibm
import carCustomer
import titanic
import whatsapp
#import sys
#sys.path.insert(0, 'C:/Users/cemce/Desktop/data science/project_2/stream/car_price')
#import car



st.set_option('deprecation.showPyplotGlobalUse', False)
def main():
    car_df = car.read_data()
    mobile = mob.read_data()
    telco_df = telco.read_data()
    ibm_df = ibm.read_data()
    car_customer_df = carCustomer.read_data()
    titanic_df = titanic.read_data()
    whatsapp_df = whatsapp.read_data()
    
    page_main = st.sidebar.selectbox("Choose a project", ["Homepage", "Mobile Price Classification","Car Price Regression","Telco Churn Classification","IBM Employee Attrition Classification","Car Customer Segmentation","Titanic Survive Classification"])
    if page_main == "Homepage":    
        set_image()
        st.title("ML projects. You can click left to select a project!")
    elif page_main =="Mobile Price Classification":
        mob.mobilePricing(mobile)
    elif page_main == "Car Price Regression":
        car.carPricing(car_df)
    elif page_main == "Telco Churn Classification":
        telco.telcoChurn(telco_df)
    elif page_main == "IBM Employee Attrition Classification":
        ibm.ibmAttr(ibm_df)
    elif page_main == "Car Customer Segmentation":
        carCustomer.carCustomer(car_customer_df)
    elif page_main == "Titanic Survive Classification":
        titanic.titanic_surv(titanic_df)
    elif page_main == "Web Whatsapp Status Emotion Classification":
        whatsapp.wh(whatsapp_df)
        
        
def set_image():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542744173-05336fcc7ad4?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1291&q=80");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

if __name__ == "__main__":
    main()