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
    log_reg = joblib.load("ibm/joblib/log_reg.joblib")
    return log_reg

def read_knnmodel():
    knn = joblib.load("ibm/joblib/knn.joblib")
    return knn

def read_decision_tree():
    dt = joblib.load("ibm/joblib/dt.joblib")
    return dt

def read_random_forest():
    rf = joblib.load("ibm/joblib/rf.joblib")
    return rf

def read_xgb_model():
    xgb = joblib.load("ibm/joblib/xgb.joblib")
    return xgb


def ibmAttr(df):
    st.title("IBM HR Employee Attrition Project")
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
        "Age" : [st.sidebar.number_input("What is the customer's age? ",value=41)],
        "BusinessTravel" : [st.sidebar.selectbox('Do you have business travel?',('Travel Rarely', 'Travel Frequently',"Non-Travel"),index = 0)],
        "DailyRate" : [st.sidebar.number_input("Daily performance rate (0,1500)",value = 1102)],
        "Department" : [st.sidebar.selectbox('Do you have business travel?',('Research & Development', 'Sales',"Human Resources"),index = 1)],
        "DistanceFromHome" : [st.sidebar.number_input("Distance from home (1,30)",value = 1)],
        "Education" : [st.sidebar.number_input("Which level of education does the person have ? (1,5)",value = 2)],
        "EducationField" : [st.sidebar.selectbox("Which education field does the person have ?  (Life Sciences, Other, Medical, Marketing,Technical Degree, Human Resources)",('Life Sciences', 'Other', 'Medical', 'Marketing','Technical Degree', 'Human Resources'),index = 0)],
        "EmployeeCount" : 1,
        "EmployeeNumber" : [st.sidebar.number_input("How many employee works for the company?",value = 1)],
        "EnvironmentSatisfaction" : [st.sidebar.number_input("Does employees satisfy with their job environment (1,4)?",value = 2)],
        "Gender" : [st.sidebar.selectbox("Whether the customer has online security or not ",("Female", "Male"),index = 0)],
        "HourlyRate": [st.sidebar.number_input("Hourly Wages? (30,100)?",value = 94)],
        "JobInvolvement": [st.sidebar.number_input("Employee's job involvement level? (1,4)?",value = 3)],
        "JobLevel": [st.sidebar.number_input("What is employee's job level? (1,5)?",value = 2)],
        "JobRole" : [st.sidebar.selectbox("What is employee's job role",('Sales Executive', 'Research Scientist', 'Laboratory Technician','Manufacturing Director', 'Healthcare Representative', 'Manager','Sales Representative', 'Research Director', 'Human Resources'),index = 0)],
        "JobSatisfaction": [st.sidebar.number_input("Does employees satisfy with their job (1,4)?",value = 4)],
        "MaritalStatus": [st.sidebar.selectbox("What is employee's marital status? ",("Single", "Married","Divorced"),index = 0)],
        "MonthlyIncome": [st.sidebar.number_input("What is employee's wage (1000,20000)?",value = 5993)],
        "MonthlyRate" : [st.sidebar.number_input("What is employee's monthly rate (2000,28000)?",value = 19479)],
        "NumCompaniesWorked" : [st.sidebar.number_input("How many companies did employee worked? ",value = 8)],
        "Over18": ["Y"],
        "OverTime": [st.sidebar.selectbox("Does have employee work on over time ? ",("Yes", "No"),index = 0)],
        "PercentSalaryHike" : [st.sidebar.number_input("What is Percentage of Salary Hike? ",value = 11)],
        "PerformanceRating" : [st.sidebar.number_input("What is your employee's performance rating? ",value = 3)],
        "RelationshipSatisfaction" : [st.sidebar.number_input("What is your employee's relationship staticfaction? ",value = 1)],
        "StandardHours" : [st.sidebar.number_input("How many hours do employees work? ",value = 80)],
        "StockOptionLevel" : [st.sidebar.number_input("What is stockp option level? ",value = 0)],
        "TotalWorkingYears" : [st.sidebar.number_input("What is employee's total working years? ",value = 8)],
        "TrainingTimesLastYear" : [st.sidebar.number_input("What is training times last year? ",value = 0)],
        "WorkLifeBalance" : [st.sidebar.number_input("Work life balanca point ? ",value = 1)],
        "YearsAtCompany" : [st.sidebar.number_input("How many years employee work? ",value = 6)],
        "YearsInCurrentRole" : [st.sidebar.number_input("How many years did employee work for the same role? ",value = 4)],
        "YearsSinceLastPromotion" : [st.sidebar.number_input("How long did employee worked from last promotion? ",value = 0)],
        "YearsWithCurrManager" : [st.sidebar.number_input("How many years did employee worked with current manager? ",value = 5)],
        }
    return dummy


def prepro(ibm):
         
    if ibm.BusinessTravel.iloc[0] == "Travel Rarely":
        ibm["BusinessTravel"] = [1]
    elif ibm.BusinessTravel.iloc[0]== "Travel Frequently":
        ibm["BusinessTravel"] = [2]
    elif ibm.BusinessTravel.iloc[0] == "Non-Travel":
        ibm["BusinessTravel"] = [0] 
    
    if ibm.Department.iloc[0] == "Human Resources":
        ibm["HR"] = [1]
        ibm["R&D"] = [0]
        ibm["Sales"] = [0]
    elif ibm.Department.iloc[0] == "Research & Development":
        ibm["HR"] = [0]
        ibm["R&D"] = [1]
        ibm["Sales"] = [0]
    elif ibm.Department.iloc[0] == "Sales":
        ibm["HR"] = [0]
        ibm["R&D"] = [0]
        ibm["Sales"] = [1] 
    ibm.drop(columns = ["Department"],inplace = True)

    if ibm.EducationField.iloc[0] == "Life Sciences":
        ibm["human_resources"] = [0]
        ibm["life_sciences"] = [1]
        ibm["marketing"] = [0]
        ibm["medical"] = [0]
        ibm["other"] = [0]
        ibm["technical"] = [0]
    elif ibm.EducationField.iloc[0] == "Medical":
        ibm["human_resources"] = [0]
        ibm["life_sciences"] = [0]
        ibm["marketing"] = [0]
        ibm["medical"] = [1]
        ibm["other"] = [0]
        ibm["technical"] = [0]
    elif ibm.EducationField.iloc[0] == "Marketing":
        ibm["human_resources"] = [0]
        ibm["life_sciences"] = [0]
        ibm["marketing"] = [1]
        ibm["medical"] = [0]
        ibm["other"] = [0]
        ibm["technical"] = [0]
    elif ibm.EducationField.iloc[0] == "Technical Degree":
        ibm["human_resources"] = [0]
        ibm["life_sciences"] = [0]
        ibm["marketing"] = [0]
        ibm["medical"] = [0]
        ibm["other"] = [0]
        ibm["technical"] = [1]
    elif ibm.EducationField.iloc[0] == "Human Resources":
        ibm["human_resources"] = [1]
        ibm["life_sciences"] = [0]
        ibm["marketing"] = [0]
        ibm["medical"] = [0]
        ibm["other"] = [0]
        ibm["technical"] = [0]
    elif ibm.EducationField.iloc[0] == "Other":
        ibm["human_resources"] = [0]
        ibm["life_sciences"] = [0]
        ibm["marketing"] = [0]
        ibm["medical"] = [0]
        ibm["other"] = [1]
        ibm["technical"] = [0]
    ibm.drop(columns = ["EducationField"],inplace = True)
    
    if ibm.Gender.iloc[0] == "Female":
        ibm["Female"] = [1]
        ibm["Male"] = [0]
    elif ibm.Gender.iloc[0]== "Male":
        ibm["Female"] = [0]
        ibm["Male"] = [1]
    ibm.drop(columns = ["Gender"],inplace = True)
    

    
    if ibm.JobRole.iloc[0] == "Healthcare Representative":
        ibm['Healthcare Representative'] = [1]
        ibm['Human Resources'] = [0]
        ibm['Manager'] = [0]
        ibm['Manufacturing Director'] = [0]
        ibm['Research Director'] = [0]
        ibm['Research Scientist'] = [0]
        ibm['Sales Executive'] = [0]
        ibm['Sales Representative'] = [0]
        ibm['Laboratory Technician'] = [0]
    elif ibm.JobRole.iloc[0] == "Human Resources":
        ibm['Healthcare Representative'] = [0]
        ibm['Human Resources'] = [1]
        ibm['Manager'] = [0]
        ibm['Manufacturing Director'] = [0]
        ibm['Research Director'] = [0]
        ibm['Research Scientist'] = [0]
        ibm['Sales Executive'] = [0]
        ibm['Sales Representative'] = [0]
        ibm['Laboratory Technician'] = [0]
    elif ibm.JobRole.iloc[0] == 'Sales Executive':
        ibm['Healthcare Representative'] = [0]
        ibm['Human Resources'] = [0]
        ibm['Manager'] = [0]
        ibm['Manufacturing Director'] = [0]
        ibm['Research Director'] = [0]
        ibm['Research Scientist'] = [0]
        ibm['Sales Executive'] = [1]
        ibm['Sales Representative'] = [0]
        ibm['Laboratory Technician'] = [0]
    elif ibm.JobRole.iloc[0] == 'Manufacturing Director':
        ibm['Healthcare Representative'] = [0]
        ibm['Human Resources'] = [0]
        ibm['Manager'] = [0]
        ibm['Manufacturing Director'] = [1]
        ibm['Sales Executive'] = [0]
        ibm['Sales Representative'] = [0]
        ibm['Research Director'] = [0]
        ibm['Research Scientist'] = [0]
        ibm['Laboratory Technician'] = [0]
    elif ibm.JobRole.iloc[0] == "Healthcare Representative":
        ibm['Healthcare Representative'] = [1]
        ibm['Human Resources'] = [0]
        ibm['Manager'] = [0]
        ibm['Manufacturing Director'] = [0]
        ibm['Research Director'] = [0]
        ibm['Research Scientist'] = [0]
        ibm['Sales Executive'] = [0]
        ibm['Sales Representative'] = [0]
        ibm['Laboratory Technician'] = [0]
    elif ibm.JobRole.iloc[0] == "Manager":
        ibm['Healthcare Representative'] = [0]
        ibm['Human Resources'] = [0]
        ibm['Manager'] = [1]
        ibm['Manufacturing Director'] = [0]
        ibm['Research Director'] = [0]
        ibm['Research Scientist'] = [0]
        ibm['Sales Executive'] = [0]
        ibm['Sales Representative'] = [0]
        ibm['Laboratory Technician'] = [0]
    elif ibm.JobRole.iloc[0] == "Sales Representative":
        ibm['Healthcare Representative'] = [0]
        ibm['Human Resources'] = [0]
        ibm['Manager'] = [0]
        ibm['Manufacturing Director'] = [0]
        ibm['Research Director'] = [0]
        ibm['Research Scientist'] = [0]
        ibm['Sales Executive'] = [0]
        ibm['Sales Representative'] = [1]
        ibm['Laboratory Technician'] = [0]
    elif ibm.JobRole.iloc[0] == "Research Director":
        ibm['Healthcare Representative'] = [0]
        ibm['Human Resources'] = [0]
        ibm['Manager'] = [0]
        ibm['Manufacturing Director'] = [0]
        ibm['Research Director'] = [1]
        ibm['Research Scientist'] = [0]
        ibm['Sales Executive'] = [0]
        ibm['Sales Representative'] = [0]
        ibm['Laboratory Technician'] = [0]
    elif ibm.JobRole.iloc[0] == "Laboratory Technician":
        ibm['Healthcare Representative'] = [0]
        ibm['Human Resources'] = [0]
        ibm['Manager'] = [0]
        ibm['Manufacturing Director'] = [0]
        ibm['Research Director'] = [0]
        ibm['Research Scientist'] = [0]
        ibm['Sales Executive'] = [0]
        ibm['Sales Representative'] = [0]
        ibm['Laboratory Technician'] = [1]
    
    ibm.drop(columns = ["JobRole"],inplace = True)
    
    if ibm.MaritalStatus.iloc[0] == "Single":
        ibm["Divorced"] = [0]
        ibm["Married"] = [0]
        ibm["Single"] = [1]
    elif ibm.MaritalStatus.iloc[0] == "Married":
        ibm["Divorced"] = [0]
        ibm["Married"] = [1]
        ibm["Single"] = [0]
    elif ibm.MaritalStatus.iloc[0] == "Divorced":
        ibm["Divorced"] = [1] 
        ibm["Married"] = [0]
        ibm["Single"] = [0]
    ibm.drop(columns = ["MaritalStatus"],inplace = True)
    
    if ibm.OverTime.iloc[0] == "Yes":
        ibm["OverTime"] = [1]
    elif ibm.OverTime.iloc[0] == "No":
        ibm["OverTime"] = [0]
    
    ibm.drop(columns = ["Over18"],inplace = True)
    return ibm


def prediction(model,X):
    #test_dt_pca = prepro_pca(test_dt_pf)
    #try:
    
    st.write("Our data: ")
    st.write(read_data())
    st.write(X.columns)
    st.write("Our predict is ------> :")
    if model.predict(X) == 0:
        st.write("------ > No Attrition < ------")
    elif model.predict(X) == 1:
        st.write("------ > Has Attrition! < ------")
        
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