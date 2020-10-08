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
#import sys
#sys.path.insert(0, 'C:/Users/cemce/Desktop/data science/project_2/stream/car_price')
#import car



st.set_option('deprecation.showPyplotGlobalUse', False)
def main():
    car_df = car.read_data()
    mobile = mob.read_data()

    page_main = st.sidebar.selectbox("Choose a project", ["Homepage", "Mobile Price Classification","Car Price Regression"])
    if page_main == "Homepage":    
        set_image()
        st.title("HELLO WORLD!")
    elif page_main =="Mobile Price Classification":
        mob.mobilePricing(mobile)
    elif page_main == "Car Price Regression":
        car.carPricing(car_df)
        
        
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