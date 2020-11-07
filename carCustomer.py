import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


import warnings
warnings.filterwarnings(action="ignore")
import joblib

def read_data():
    df = pd.read_csv("ibm/ibm.csv")
    return df

def read_lrmodel():
    log_reg = joblib.load("car_customer/joblib/log_reg.joblib")
    return log_reg

def read_knnmodel():    
    knn = joblib.load("car_customer/joblib/knn.joblib")
    return knn

def read_decision_tree():
    dt = joblib.load("car_customer/joblib/dt.joblib")
    return dt

def read_random_forest():
    rf = joblib.load("car_customer/joblib/rf.joblib")
    return rf

def read_xgb_model():
    xgb = joblib.load("car_customer/joblib/xgb.joblib")
    return xgb


def carCustomer(df):
    st.title("Customer Segmentation for a Car Project")
    page_sub = st.sidebar.selectbox("Do something on the dataset", ["Prediction", "Exploration"])
    if page_sub == "Prediction":
        st.header("This is your prediction page.")
        st.write("Enter your values as you want.")
        page_model = st.sidebar.selectbox("Choose a Data Exploration", ["Logistic Regression","KNN","Decision Tree", "Random Forest","XGBoost"])
        if page_model == "Decision Tree":
            model = read_decision_tree()
            test_data = pd.DataFrame.from_dict(get_dummy())
            test_prepro = prepro(test_data)
            prediction(model,test_prepro)
            feature_importance(model,test_prepro)
        elif page_model == "Random Forest":
            model = read_random_forest()
            test_data = pd.DataFrame.from_dict(get_dummy())
            test_prepro = prepro(test_data)
            prediction(model,test_prepro)
            feature_importance(model,test_prepro)
        elif page_model == "Logistic Regression":
            model = read_lrmodel()
            test_data = pd.DataFrame.from_dict(get_dummy())
            test_prepro = prepro(test_data)
            prediction(model,test_prepro)
        elif page_model == "KNN":
            model = read_knnmodel()
            test_data = pd.DataFrame.from_dict(get_dummy())
            test_prepro = prepro(test_data)
            prediction(model,test_prepro)
        elif page_model == "XGBoost":
            model = read_xgb_model()
            test_data = pd.DataFrame.from_dict(get_dummy())
            test_prepro = prepro(test_data)
            prediction(model,test_prepro)
            feature_importance(model,test_prepro)
    elif page_sub == "Exploration":
        dataExploration(df)

def get_dummy():
    dummy = {
        "ID" : [st.sidebar.number_input("What is the customer id? ",value=1)],
        "Gender" : [st.sidebar.selectbox("Gender",("Female", "Male"),index = 0)],
        "Ever_Married" : [st.sidebar.selectbox("Have costumer ever married with someone? ",("Yes", "No"),index = 0)],
        "Age" : [st.sidebar.number_input("What is the customer's age? ",value=41)],
        "Graduated" : [st.sidebar.selectbox("Have costumer ever graduated from somewhere? ",("Yes", "No"),index = 0)],
        "Profession" : [st.sidebar.selectbox("What is customer's profession?",("Artist","Healthcare","Entertainment","Doctor","Engineer","Executive","Lawyer","Marketing","Homemaker"),index = 1)],
        "Work_Experience" : [st.sidebar.number_input("Work experience in year",value = 1)],
        "Spending_Score" : [st.sidebar.selectbox("Customer general spending score? ",("Low","Average", "High"),index = 0)],
        "Family_Size" : [st.sidebar.number_input("Family Size",value = 1)],
        "Var_1" : [st.sidebar.selectbox("Anonymised Category for the customer.?",("Cat_1","Cat_2","Cat_3","Cat_4","Cat_5","Cat_6","Cat_7"),index = 1)],
        }
    return dummy


def prepro(carCustomerDf):
        
    carCustomerDf.drop(columns = ["ID"],inplace = True)
    
    if carCustomerDf.Gender.iloc[0] == "Female":
        carCustomerDf["Gender"] = [0]
    elif carCustomerDf.Gender.iloc[0]== "Male":
        carCustomerDf["Gender"] = [1]
    
    if carCustomerDf.Ever_Married.iloc[0] == "Yes":
        carCustomerDf["Ever_Married"] = [1]
    elif carCustomerDf.Ever_Married.iloc[0]== "No":
        carCustomerDf["Ever_Married"] = [0]
    
        
    if carCustomerDf.Graduated.iloc[0] == "Yes":
        carCustomerDf["Graduated"] = [1]
    elif carCustomerDf.Graduated.iloc[0]== "No":
        carCustomerDf["Graduated"] = [0]
    

    if carCustomerDf.Profession.iloc[0] == "Artist":
        carCustomerDf["Artist"] = [1]
        carCustomerDf["Doctor"] = [0]
        carCustomerDf["Engineer"] = [0]
        carCustomerDf["Entertainment"] = [0]
        carCustomerDf["Executive"] = [0]
        carCustomerDf["Healthcare"] = [0]
        carCustomerDf["Homemaker"] = [0]
        carCustomerDf["Lawyer"] = [0]
        carCustomerDf["Marketing"] = [0]
    elif carCustomerDf.Profession.iloc[0] == "Doctor":
        carCustomerDf["Artist"] = [0]
        carCustomerDf["Doctor"] = [1]
        carCustomerDf["Engineer"] = [0]
        carCustomerDf["Entertainment"] = [0]
        carCustomerDf["Executive"] = [0]
        carCustomerDf["Healthcare"] = [0]
        carCustomerDf["Homemaker"] = [0]
        carCustomerDf["Lawyer"] = [0]
        carCustomerDf["Marketing"] = [0]
    elif carCustomerDf.Profession.iloc[0] == "Engineer":
        carCustomerDf["Artist"] = [0]
        carCustomerDf["Doctor"] = [0]
        carCustomerDf["Engineer"] = [1]
        carCustomerDf["Entertainment"] = [0]
        carCustomerDf["Executive"] = [0]
        carCustomerDf["Healthcare"] = [0]
        carCustomerDf["Homemaker"] = [0]
        carCustomerDf["Lawyer"] = [0]
        carCustomerDf["Marketing"] = [0]
    elif carCustomerDf.Profession.iloc[0] == "Entertainment":
        carCustomerDf["human_resources"] = [0]
        carCustomerDf["Doctor"] = [0]
        carCustomerDf["Engineer"] = [0]
        carCustomerDf["Entertainment"] = [1]
        carCustomerDf["Executive"] = [0]
        carCustomerDf["Healthcare"] = [0]
        carCustomerDf["Homemaker"] = [0]
        carCustomerDf["Lawyer"] = [0]
        carCustomerDf["Marketing"] = [0]
    elif carCustomerDf.Profession.iloc[0] == "Executive":
        carCustomerDf["Artist"] = [0]
        carCustomerDf["Doctor"] = [0]
        carCustomerDf["Engineer"] = [0]
        carCustomerDf["Entertainment"] = [0]
        carCustomerDf["Executive"] = [1]
        carCustomerDf["Healthcare"] = [0]
        carCustomerDf["Homemaker"] = [0]
        carCustomerDf["Lawyer"] = [0]
        carCustomerDf["Marketing"] = [0]
    elif carCustomerDf.Profession.iloc[0] == "Healthcare":
        carCustomerDf["Artist"] = [0]
        carCustomerDf["Doctor"] = [0]
        carCustomerDf["Engineer"] = [0]
        carCustomerDf["Entertainment"] = [0]
        carCustomerDf["Executive"] = [0]
        carCustomerDf["Healthcare"] = [1]
        carCustomerDf["Homemaker"] = [0]
        carCustomerDf["Lawyer"] = [0]
        carCustomerDf["Marketing"] = [0]
    elif carCustomerDf.Profession.iloc[0] == "Homemaker":
        carCustomerDf["Artist"] = [0]
        carCustomerDf["Doctor"] = [0]
        carCustomerDf["Engineer"] = [0]
        carCustomerDf["Entertainment"] = [0]
        carCustomerDf["Executive"] = [0]
        carCustomerDf["Healthcare"] = [0]
        carCustomerDf["Homemaker"] = [1]
        carCustomerDf["Lawyer"] = [0]
        carCustomerDf["Marketing"] = [0]
    elif carCustomerDf.Profession.iloc[0] == "Lawyer":
        carCustomerDf["Artist"] = [0]
        carCustomerDf["Doctor"] = [0]
        carCustomerDf["Engineer"] = [0]
        carCustomerDf["Entertainment"] = [0]
        carCustomerDf["Executive"] = [0]
        carCustomerDf["Healthcare"] = [0]
        carCustomerDf["Homemaker"] = [0]
        carCustomerDf["Lawyer"] = [1]
        carCustomerDf["Marketing"] = [0]
    elif carCustomerDf.Profession.iloc[0] == "Marketing":
        carCustomerDf["Artist"] = [0]
        carCustomerDf["Doctor"] = [0]
        carCustomerDf["Engineer"] = [0]
        carCustomerDf["Entertainment"] = [0]
        carCustomerDf["Executive"] = [0]
        carCustomerDf["Healthcare"] = [0]
        carCustomerDf["Homemaker"] = [0]
        carCustomerDf["Lawyer"] = [0]
        carCustomerDf["Marketing"] = [1]
        
    carCustomerDf.drop(columns = ["Profession"],inplace = True)

    
    if carCustomerDf.Spending_Score.iloc[0] == "Average":
        carCustomerDf["Spending_Score"] = [1]

    elif carCustomerDf.Spending_Score.iloc[0] == "High":
        carCustomerDf["Spending_Score"] = [2]
    elif carCustomerDf.Spending_Score.iloc[0] == "Low":
        carCustomerDf["Spending_Score"] = [0]
    
    if carCustomerDf.Var_1.iloc[0] == "Cat_1":
        carCustomerDf["Cat_1"] = [1]
        carCustomerDf["Cat_2"] = [0]
        carCustomerDf["Cat_3"] = [0]
        carCustomerDf["Cat_4"] = [0]
        carCustomerDf["Cat_5"] = [0]
        carCustomerDf["Cat_6"] = [0]
        carCustomerDf["Cat_7"] = [0]
       
    elif carCustomerDf.Var_1.iloc[0] == "Cat_2":
        carCustomerDf["Cat_1"] = [0]
        carCustomerDf["Cat_2"] = [1]
        carCustomerDf["Cat_3"] = [0]
        carCustomerDf["Cat_4"] = [0]
        carCustomerDf["Cat_5"] = [0]
        carCustomerDf["Cat_6"] = [0]
        carCustomerDf["Cat_7"] = [0]
        
    elif carCustomerDf.Var_1.iloc[0] == "Cat_3":
        carCustomerDf["Cat_1"] = [0]
        carCustomerDf["Cat_2"] = [0]
        carCustomerDf["Cat_3"] = [1]
        carCustomerDf["Cat_4"] = [0]
        carCustomerDf["Cat_5"] = [0]
        carCustomerDf["Cat_6"] = [0]
        carCustomerDf["Cat_7"] = [0]
        
    elif carCustomerDf.Var_1.iloc[0] == "Cat_4":
        carCustomerDf["Cat_1"] = [0]
        carCustomerDf["Cat_2"] = [0]
        carCustomerDf["Cat_3"] = [0]
        carCustomerDf["Cat_4"] = [1]
        carCustomerDf["Cat_5"] = [0]
        carCustomerDf["Cat_6"] = [0]
        carCustomerDf["Cat_7"] = [0]
    elif carCustomerDf.Var_1.iloc[0] == "Cat_5":
        carCustomerDf["Cat_1"] = [0]
        carCustomerDf["Cat_2"] = [0]
        carCustomerDf["Cat_3"] = [0]
        carCustomerDf["Cat_4"] = [0]
        carCustomerDf["Cat_5"] = [1]
        carCustomerDf["Cat_6"] = [0]
        carCustomerDf["Cat_7"] = [0]
    elif carCustomerDf.Var_1.iloc[0] == "Cat_6":
        carCustomerDf["Cat_1"] = [0]
        carCustomerDf["Cat_2"] = [0]
        carCustomerDf["Cat_3"] = [0]
        carCustomerDf["Cat_4"] = [0]
        carCustomerDf["Cat_5"] = [0]
        carCustomerDf["Cat_6"] = [1]
        carCustomerDf["Cat_7"] = [0]
    elif carCustomerDf.Var_1.iloc[0] == "Cat_7":
        carCustomerDf["Cat_1"] = [0]
        carCustomerDf["Cat_2"] = [0]
        carCustomerDf["Cat_3"] = [0]
        carCustomerDf["Cat_4"] = [0]
        carCustomerDf["Cat_5"] = [0]
        carCustomerDf["Cat_6"] = [0]
        carCustomerDf["Cat_7"] = [1]
    
    carCustomerDf.drop(columns = ["Var_1"],inplace = True)
    return carCustomerDf


def prediction(model,X):
    #test_dt_pca = prepro_pca(test_dt_pf)
    #try:
    
    st.write("Our data: ")
    st.write(read_data())
    st.write(X.columns)
    st.write("Our predict is ------> :")
    if model.predict(X) == 0:
        st.write("------ > Class A < ------")
    elif model.predict(X) == 1:
        st.write("------ >  Class B < ------")
    elif model.predict(X) == 2:
        st.write("------ >  Class C < ------")
    elif model.predict(X) == 3:
        st.write("------ >  Class D < ------")
        
    st.write("Our predict probability is ------> :")
    st.write(model.predict_proba(X).max())
    #except:
    #st.write("An exception occurred")
    st.write("Importance level of features")
    return X



def feature_importance(model,lst):
    importance_level = pd.Series(data=model.feature_importances_,
                        index= lst.columns)

    importance_level_sorted = importance_level.sort_values(ascending=False)[:10]
    plt.figure(figsize=(10,5))
    importance_level_sorted.plot(kind='barh', color='darkred')
    st.pyplot()
    plt.title('Importance Level of the Features')
    plt.show()