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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

def read_data():
    df = pd.read_csv("car_price/data/car.csv")
    return df

#def read_lrmodel():
    #lr_model = joblib.load("mobile_price/joblib/model/lr_pfone.joblib")
    #return lr_model

def read_knnmodel():
    knn = joblib.load("car_price/joblib/model/knn.joblib")
    return knn

def read_decision_tree():
    decision_tree = joblib.load("car_price/joblib/model/dt_reg.joblib")
    return decision_tree

def read_random_forest():
    random_forest = joblib.load("car_price/joblib/model/random_forest.joblib")
    return random_forest

#def read_svc():
    #svc_model = joblib.load("car_price/joblib/model/svc.joblib")
    #return svc_model

def read_xgb_model():
    xgb_model = joblib.load("car_price/joblib/model/xgb.joblib")
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




def carPricing(df):
    st.title("Car Pricing Project")
    page_sub = st.sidebar.selectbox("Do something on the dataset", ["Prediction", "Exploration"])
    if page_sub == "Prediction":
        st.header("This is your prediction page.")
        st.write("Enter your values as you want.")
        page_model = st.sidebar.selectbox("Choose a Data Exploration", ["Decision Tree", "Random Forest","XGBoost"])
        if page_model == "Decision Tree":
            model = read_decision_tree()
            test_data = pd.DataFrame.from_dict(get_dummy())
            test_custom_pf = first_pro(test_data)
            prediction(model,test_custom_pf)
            feature_importance(model,test_custom_pf)
        elif page_model == "Random Forest":
            model = read_random_forest()
            test_data = pd.DataFrame.from_dict(get_dummy())
            test_dt_pf = prepro_pf(test_data)
            prediction(model,test_dt_pf)
            feature_importance(model,test_dt_pf)
        elif page_model == "XGBoost":
            model = read_xgb_model()
            test_data = pd.DataFrame.from_dict(get_dummy())
            test_dt_pf = prepro_pf(test_data)
            prediction(model,test_dt_pf)
            feature_importance(model,test_dt_pf)
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
            
def get_dummy():
    dummy = {
        "Name" : [st.sidebar.text_input("What is your cars model? Example : Hyundai Accent",value="Audi A6 2.7 TDI")],
        "Location" : [st.sidebar.selectbox('What is your Location to sell?',('Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Hyderabad', 'Jaipur',
       'Kochi', 'Kolkata', 'Delhi', 'Bangalore', 'Ahmedabad'),index = 7)],
        "Year" : [st.sidebar.number_input("What is the production of your car?",1950,2021,value = 2010)],
        "Kilometers_Driven" : [st.sidebar.number_input("Front Camera",value = 35000)],
        "Fuel_Type" : [st.sidebar.selectbox('What is fuel type of your car?',('CNG', 'Diesel', 'Petrol', 'LPG'),index=1)],
        "Transmission" : [st.sidebar.selectbox('What is transmission type of your car?',('Manual', 'Automatic'),index=1)],
        "Owner_Type" : [st.sidebar.selectbox('How much people used this car before?',('First', 'Second', 'Third', 'Fourth & Above'),index =0)],
        "Mileage" : [st.sidebar.number_input("What is the standard mileage offered by the car company in kmpl or km/kg?",value = 12.4)],
        "Engine" : [st.sidebar.number_input("What is the displacement volume of the engine in CC",150,5000,value = 2698)],
        "Power" : [st.sidebar.number_input("How much microprocessor have core?",value = 179.5)],
        "Seats" : [st.sidebar.number_input("What is the number of seats in the car.",max_value = 11,value = 5)]}
    return dummy
    
def prediction(model,X):
    st.write(X)
    #test_dt_pca = prepro_pca(test_dt_pf)
    #try:
    st.write(read_data())
        #st.write("Prob is : ",model.predict_proba(X).max())
    
    st.write("Our predict is ------> :")
    st.write(model.predict(X))
    #except:
    st.write("An exception occurred")
    return X
    
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

def split(df):
    lst = []
    for x in range(df.shape[0]):
        txt = df.iloc[x]
        number = txt.split(" ")
        lst.append(number[0])
    return lst

def first_pro(X):
    X["Year"] = np.abs(2020- X["Year"]) 
    for i in ['Automatic', 'Manual']:
        if i == X.Transmission.iloc[0]:
            lst= []
            lst.append(1)
            X[i] = [1]
        else:
            X[i] = [0]

    lst = []
    for i in range(X.shape[0]):
        if X.Owner_Type.iloc[i] == "First":
            lst.append(1)
        elif X.Owner_Type.iloc[i] == "Second":
            lst.append(2)
        elif X.Owner_Type.iloc[i] == "Third":
            lst.append(3)
        else:
            lst.append(4)
    X["OwnerType"] = lst 
    
    lp = ['CNG', 'Diesel',"Petrol","LPG","Electric"]
    for i in lp:
        lst = []
        if i == X.Fuel_Type.iloc[0]:
            lst.append(1)
            X["is_"+i] = lst
        else:
            lst.append(0)
            X["is_"+i]=lst

    X["Brand"] = split(X.Name)
    lp = ['Ambassador', 'Audi', 'BMW', 'Bentley', 'Chevrolet', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'ISUZU', 'Isuzu', 'Jaguar', 'Jeep', 'Lamborghini', 'Land', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'Porsche', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']
    for i in lp:
        lst = []
        if i == X.Brand.iloc[0]:
            lst.append(1)
            X[i] = lst
        else:
            lst.append(0)
            X[i]=lst
    X = X.select_dtypes(exclude = "object")
    X.reset_index(inplace = True)
    X.drop(columns = "index", inplace = True)
    st.write(X.shape[1])
    return X
    
def prepro_custom_pf(X):
    X_Sta = first_pro(X)

    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X_Sta)
    X_pf_df = pd.DataFrame(X_poly, columns = poly.get_feature_names(X_Sta.columns))
    X_pf_df.drop(columns = "1", inplace=True)
    st.write(X_pf_df)
    best_cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats',
       'Automatic', 'Manual', 'OwnerType', 'Year^2', 'Year Kilometers_Driven',
       'Year Mileage', 'Year Engine', 'Year Power', 'Year Seats',
       'Year Automatic', 'Year Manual', 'Year OwnerType',
       'Kilometers_Driven^2', 'Kilometers_Driven Mileage',
       'Kilometers_Driven Engine', 'Kilometers_Driven Power',
       'Kilometers_Driven Seats', 'Kilometers_Driven Automatic',
       'Kilometers_Driven Manual', 'Kilometers_Driven OwnerType', 'Mileage^2',
       'Mileage Engine', 'Mileage Power', 'Mileage Seats', 'Mileage Manual',
       'Mileage OwnerType', 'Engine^2', 'Engine Seats', 'Engine Manual',
       'Engine OwnerType', 'Power^2', 'Power Seats', 'Power Automatic',
       'Seats^2', 'Seats Manual', 'Seats OwnerType', 'Automatic^2',
       'Automatic OwnerType', 'Manual^2', 'OwnerType^2']
    X_custom_pf = X_pf_df[best_cols]
    return X_custom_pf

def prepro_pf(X):
    
    st.write(X)
    X_Sta = first_pro(X)
   

    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X_Sta)
    X_pf_df = pd.DataFrame(X_poly, columns = poly.get_feature_names(X_Sta.columns))
    X_pf_df.drop(columns = "1", inplace=True)
    return X_pf_df

def feature_importance(model,lst):
    importance_level = pd.Series(data=model.feature_importances_,
                        index= lst.columns)

    importance_level_sorted = importance_level.sort_values(ascending=False)[:10]
    plt.figure(figsize=(10,5))
    importance_level_sorted.plot(kind='barh', color='darkred')
    st.pyplot()
    plt.title('Importance Level of the Features')
    plt.show()



