#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


def read_data():
    df = pd.read_csv("mobile_price/data/mobile_price.csv")
    return df

def read_lrmodel():
    lr_model = joblib.load("mobile_price/joblib/model/lr_pfone.joblib")
    return lr_model

def read_knnmodel():
    knn = joblib.load("mobile_price/joblib/model/knn.joblib")
    return knn

def read_decision_tree():
    decision_tree = joblib.load("mobile_price/joblib/model/decision_tree.joblib")
    return decision_tree

def read_random_forest():
    random_forest = joblib.load("mobile_price/joblib/model/random_forest.joblib")
    return random_forest

def read_svc():
    svc_model = joblib.load("mobile_price/joblib/model/svc.joblib")
    return svc_model

def read_xgb_model():
    xgb_model = joblib.load("mobile_price/joblib/model/xgb_model.joblib")
    return xgb_model
#@st.cache
#def read_pca():
 #   prepro = joblib.load("joblib/prepro/prepro.joblib")
  #  return prepro


#prepro = read_pca()
st.set_option('deprecation.showPyplotGlobalUse', False)
#def main():
 #   mobile=read_data()
  #  mobilePricing(mobile)




def mobilePricing(df):
    st.title("Mobile Pricing Project")
    page_sub = st.sidebar.selectbox("Do something on the dataset", ["Prediction", "Exploration"])
    if page_sub == "Prediction":
        st.header("This is your prediction page.")
        st.write("Enter your values as you want.")
        page_model = st.sidebar.selectbox("Choose a Data Exploration", ["K-nearest Neighbour","Decision Tree", "Random Forest","SVC","XGBoost"])
        if page_model == "K-nearest Neighbour":
            model = read_knnmodel()
            lst = prediction(model)
        elif page_model == "Decision Tree":
            model = read_decision_tree()
            lst = prediction(model)
            feature_importance(model,lst)
        elif page_model == "Random Forest":
            model = read_random_forest()
            lst = prediction(model)
            feature_importance(model,lst)
        elif page_model == "SVC":
            model = read_svc()
            lst = prediction(model)
        elif page_model == "XGBoost":
            model = read_xgb_model()
            lst = prediction(model)
            feature_importance(model,lst)
    elif page_sub == "Exploration":
        dataExploration(df)


def dataExploration(df):
    st.title("Data Exploration")
    page_exp = st.sidebar.selectbox("Choose a Data Exploration", ["Visulation", "Statistics"])
    if page_exp == "Visulation":
        visualize_data(df)
    elif page_exp == "Statistics":
        print(0)
    
    
    
def visualize_data(df):
    page_plot = st.sidebar.selectbox("Choose your plot", ["Scatter", "Plot","Countplot"])
    
    if page_plot == "Scatter":
        x_axis = st.selectbox("Choose a variable for the x-axis", df.columns, index=3)
        y_axis = st.selectbox("Choose a variable for the y-axis", df.columns, index=4)
        if st.checkbox('Show scatter with hue'):
            hue = st.selectbox("Choose a variable for the hue", df.columns, index=5)
            sns.scatterplot(df[x_axis],df[y_axis],hue=df[hue],palette="rocket_r")
            st.pyplot()
        else:
            sns.scatterplot(df[x_axis],df[y_axis],palette="rocket_r")
            st.pyplot()
    elif page_plot == "Plot":
        x_axis = st.selectbox("Choose a variable for the x-axis", df.columns, index=6)
        y_axis = st.selectbox("Choose a variable for the y-axis", df.columns, index=7)
        plt.plot(df[x_axis],df[y_axis])
        st.pyplot()
    elif page_plot == "Countplot":
        x_axis = st.selectbox("Choose a variable for the x-axis", df.columns, index=8)
        if st.checkbox('Show countplot with hue'):
            hue = st.selectbox("Choose a variable for the hue", df.columns, index=10)
            sns.countplot(df[x_axis],hue=df[hue],palette="rocket_r")
            st.pyplot()
        else:
            sns.countplot(df[x_axis],palette="rocket_r")
            st.pyplot()
            

def prediction(model):
    dummy = {
        "battery_power" : [st.sidebar.number_input("Battery Power (mAh)",300,4000,value = 1253)],
        "blue" : [st.sidebar.selectbox("Bluetooth",[0,1],index  = 1)],
        "clock_speed" : [st.sidebar.number_input("Clock Speed at which microprocessor executes instructions",0.50,5.00,value = 0.5000)],
        "dual_sim" : [st.sidebar.selectbox("Dual Sim?",[0,1],index = 1)],
        "fc" : [st.sidebar.number_input("Front Camera",max_value = 60,value = 5)],
        "four_g" : [st.sidebar.selectbox("Has 4G?",[0,1],index  = 1)],
        "int_memory" : [st.sidebar.number_input("Internal Memory in Gigabytes",2,256,value = 5)],
        "m_dep" : [st.sidebar.number_input("Mobile Depth in cm",0.1,1.5,value = 0.2000)],
        "mobile_wt" : [st.sidebar.number_input("Weight of mobile phone",50,300,value = 152)],
        "n_cores" : [st.sidebar.number_input("How much microprocessor have core?",1,16,value = 2)],
        "pc" : [st.sidebar.number_input("Primary Camera",max_value = 60,value = 19)],
        "px_height" : [st.sidebar.number_input("Pixel Resolution Height",1,1960,value = 685)],
        "px_width" : [st.sidebar.number_input("Pixel Resolution Width",1,2000,value = 714)],
        "ram" : [st.sidebar.number_input("Random Access Memory in Mega Bytes",256,16000,value = 1878)] ,
        "sc_h": [st.sidebar.number_input("Screen Height of mobile in cm",3,25,value = 15)], 
        "sc_w" : [st.sidebar.number_input("Screen Width of mobile in cm",max_value = 18,value = 0)],
        "talk_time" : [st.sidebar.number_input("longest time that a single battery charge will last when you are",2,20,value = 4)],
        "three_g":[st.sidebar.selectbox("Has 3G or not",[0,1],index = 1)],
        "touch_screen" : [st.sidebar.selectbox("Has touch screen or not",[0,1],index = 1)],
        "wifi" : [st.sidebar.selectbox("Has wifi or not",[0,1],index = 0)]}
    
    test_data = pd.DataFrame.from_dict(dummy)
    test_dt_pf = prepro(test_data)
    #test_dt_pca = prepro_pca(test_dt_pf)
    try:
        st.write(read_data())
        st.write("Prob is : ",model.predict_proba(test_dt_pf).max())      
        st.write("Our predict is ------> :")
        if model.predict(test_dt_pf) == 0:
            st.write(test_data)
            st.write("------------Cheapest class(Class-0)------------")
        elif model.predict(test_dt_pf) == 1:
            st.write(test_data)
            st.write("------------Cheaper class(Class-1)----------------")
        elif model.predict(test_dt_pf) == 2:
            st.write(test_data)
            st.write("--------------------Expensive class(Class-2)---------------")
        elif model.predict(test_dt_pf) == 3:
            st.write(test_data)
            st.write("------------------------Most Expensive Class(Class-3)--------------------")
        
        return test_dt_pf
    except:
        st.write("An exception occurred")
    
    
    
def set_image():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542744173-05336fcc7ad4?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1291&q=80");
    background-size: auto;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
def prepro(X):
    from sklearn.preprocessing import PolynomialFeatures
    import pandas as pd
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X)
    X_pf_df = pd.DataFrame(X_poly, columns = poly.get_feature_names(X.columns))
    X_pf_df.drop(columns = "1", inplace=True)
    X_final = X_pf_df[['ram',
 'ram^2',
 'battery_power ram',
 'px_width ram',
 'mobile_wt ram',
 'ram sc_h',
 'ram talk_time',
 'n_cores ram',
 'px_height ram',
 'int_memory ram',
 'clock_speed ram',
 'ram three_g',
 'm_dep ram',
 'pc ram',
 'ram sc_w',
 'four_g ram',
 'dual_sim ram',
 'ram wifi',
 'blue ram']]
    return X_final

def feature_importance(model,lst):
    importance_level = pd.Series(data=model.feature_importances_,
                        index= lst.columns)

    importance_level_sorted = importance_level.sort_values()
    plt.figure(figsize=(10,5))
    importance_level_sorted.plot(kind='barh', color='darkred')
    st.pyplot()
    plt.title('Importance Level of the Features')
    plt.show()


def prepro_pca(X_pf_df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import pandas as pd
    X_Sta = StandardScaler().fit_transform(X_pf_df)
    pca = PCA(n_components=16)
    principal_components = pca.fit_transform(X_Sta)
    X_pf_pca = pd.DataFrame(principal_components)
    return X_pf_pca

