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
    df = pd.read_csv("telco/data/telco.csv")
    return df

def read_lrmodel():
    log_reg = joblib.load("telco/joblib/log_reg.joblib")
    return log_reg

def read_knnmodel():
    knn = joblib.load("telco/joblib/knn.joblib")
    return knn

def read_decision_tree():
    dt = joblib.load("telco/joblib/dt.joblib")
    return dt

def read_random_forest():
    rf = joblib.load("telco/joblib/rf.joblib")
    return rf

def read_xgb_model():
    xgb = joblib.load("telco/joblib/xgb.joblib")
    return xgb




def telcoChurn(df):
    st.title("Telco Project")
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
        "customerID" : [st.sidebar.text_input("What is the customer id? ",value="7590-VHVEG")],
        "gender" : [st.sidebar.selectbox('What is customer\'s gender?',('female', 'male'),index = 0)],
        "SeniorCitizen" : [st.sidebar.selectbox('Is customer a Senior Citizen?',("Yes","No"),index = 1)],
        "Partner" : [st.sidebar.selectbox('Whether the customer has a partner or not (Yes, No)?',("Yes","No"),index = 0)],
        "Dependents" : [st.sidebar.selectbox("Whether the customer has dependents or not (Yes, No)",("Yes","No"),index = 1)],
        "tenure" : [st.sidebar.number_input("How much months the customer has stayed with the company?",value = 1)],
        "PhoneService" : [st.sidebar.selectbox('Whether the customer has a phone service or not (Yes, No)?',("Yes","No"),index = 1)],
        "MultipleLines" : [st.sidebar.selectbox("Whether the customer has multiple lines or not (Yes, No, No phone service)",("Yes", "No", "No phone service"),index = 2)],
        "InternetService" : [st.sidebar.selectbox("Customer’s internet service provider (DSL, Fiber optic, No)",("DSL", "Fiber optic", "No"),index = 0)],
        "OnlineSecurity" : [st.sidebar.selectbox("Whether the customer has online security or not (Yes, No, No internet service)",("Yes", "No", "No internet service"),index = 1)],
        "OnlineBackup" : [st.sidebar.selectbox("Whether the customer has online backup or not (Yes, No, No internet service)",("Yes", "No", "No internet service"),index = 0)],
        "DeviceProtection" : [st.sidebar.selectbox("Whether the customer has device protection or not (Yes, No, No internet service)",("Yes", "No", "No internet service"),index = 1)],
        "TechSupport" : [st.sidebar.selectbox("Whether the customer has tech support or not (Yes, No, No internet service)",("Yes", "No", "No internet service"),index = 1)],
        "StreamingTV" : [st.sidebar.selectbox("Whether the customer has streaming TV or not (Yes, No, No internet service)",("Yes", "No", "No internet service"),index = 1)],
        "StreamingMovies" : [st.sidebar.selectbox("Whether the customer has streaming movies or not (Yes, No, No internet service)",("Yes", "No", "No internet service"),index = 1)],
        "Contract" : [st.sidebar.selectbox("The contract term of the customer (Month-to-month, One year, Two year)",("Month-to-month", "One year", "Two year"),index = 0)],
        "PaperlessBilling" : [st.sidebar.selectbox("Whether the customer has paperless billing or not (Yes, No)",("Yes", "No"),index = 0)],
        "PaymentMethod" : [st.sidebar.selectbox("The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))",("Electronic check", "Mailed check","Bank transfer (automatic)","Credit card (automatic)"),index = 0)],
        "MonthlyCharges" : [st.sidebar.number_input("The amount charged to the customer monthly",value = 29.85)],
        "TotalCharges" : [st.sidebar.number_input("The total amount charged to the customer",value = 29.85)]
        }
    return dummy

def prepro(telco):
    le = LabelEncoder()
    ohe = OneHotEncoder()
    telco.drop(columns = ["customerID"],inplace = True)
    #telco.gender = le.fit_transform(telco.gender)
    #telco.Partner = le.fit_transform(telco.Partner)
    #telco.Dependents = le.fit_transform(telco.Dependents)
    #telco.SeniorCitizen = le.fit_transform(telco.SeniorCitizen)
    #telco.PhoneService = le.fit_transform(telco.PhoneService)
    #telco["PaperlessBilling"] = le.fit_transform(telco.PaperlessBilling)
    #telco["Churn"] = le.fit_transform(telco.Churn)
    
    for i in ["gender","Partner","Dependents","SeniorCitizen","PhoneService","PaperlessBilling"]:
        if telco[i].iloc[0] == "Yes" or telco[i].iloc[0] == "yes" or telco[i].iloc[0] == "male":
            telco[i] = [1]
        elif telco[i].iloc[0] == "No" or telco[i].iloc[0] == "no" or telco[i].iloc[0] == "female": 
            telco[i] = [0]
            
    if telco.MultipleLines.iloc[0] == "No phone service":
        telco["No_PhoneServices"] = [1]
        telco["No_multiplelines"] = [0]
        telco["Multiplelines"] = [0]
    elif telco.MultipleLines.iloc[0]== "No":
        telco["No_PhoneServices"] = [0]
        telco["No_multiplelines"] = [1]
        telco["Multiplelines"] = [0]
    elif telco.MultipleLines.iloc[0] == "Yes":
        telco["No_PhoneServices"] = [0]
        telco["No_multiplelines"] = [0]
        telco["Multiplelines"] = [1]    
    telco.drop(columns = ["MultipleLines"],inplace = True)
    
    if telco.InternetService.iloc[0] == "DSL":
        telco["DSL"] = [1]
        telco["Fiber_optic"] = [0]
        telco["No_internet_services"] = [0]
    elif telco.InternetService.iloc[0] == "Fiber optic":
        telco["DSL"] = [0]
        telco["Fiber_optic"] = [1]
        telco["No_internet_services"] = [0]
    elif telco.InternetService.iloc[0] == "No":
        telco["DSL"] = [0]
        telco["Fiber_optic"] = [0]
        telco["No_internet_services"] = [1] 
    telco.drop(columns = ["InternetService"],inplace = True)
    
    
    if telco.OnlineSecurity.iloc[0] == "No":
        telco["No_OnlineSecurity"] = [1]
        telco["Yes_OnlineSecurity"] = [0]
    elif telco.OnlineSecurity.iloc[0] == "Yes":
        telco["No_OnlineSecurity"] = [0]
        telco["Yes_OnlineSecurity"] = [1]
    telco.drop(columns=["OnlineSecurity"],inplace=True)
    
    
    for i in ["OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
        str_yes = "Have_"+i
        str_no = "No_"+i
        if telco[i].iloc[0] == "No":
            telco[str_no] = [1]
            telco[str_yes] = [0]
        elif telco[i].iloc[0] == "Yes":
            telco[str_no] = [0]
            telco[str_yes] = [1]
        telco.drop(columns=[i],inplace=True)
    
    if telco.Contract.iloc[0] == "Month-to-month":
        telco["Month-to-month"] = [1]
        telco["One_year"] = [0]
        telco["Two_year"] = [0]
    elif telco.Contract.iloc[0] == "One year":
        telco["Month-to-month"] = [0]
        telco["One_year"] = [1]
        telco["Two_year"] = [0]
    elif telco.Contract.iloc[0] == "Two year":
        telco["Month-to-month"] = [0]
        telco["One_year"] = [0]
        telco["Two_year"] = [1] 
    telco.drop(columns = ["Contract"],inplace = True)
    
    if telco.PaymentMethod.iloc[0] == "Bank transfer (automatic)":
        telco["Bank transfer (automatic)"] = [1]
        telco["Credit card (automatic)"] = [0]
        telco["Electronic check"] = [0]
        telco["Mailed check"] = [0]
    elif telco.PaymentMethod.iloc[0] == "Credit card (automatic)":
        telco["Bank transfer (automatic)"] = [0]
        telco["Credit card (automatic)"] = [1]
        telco["Electronic check"] = [0]
        telco["Mailed check"] = [0]
    elif telco.PaymentMethod.iloc[0] == "Electronic check":
        telco["Bank transfer (automatic)"] = [0]
        telco["Credit card (automatic)"] = [0]
        telco["Electronic check"] = [1]
        telco["Mailed check"] = [0]
    elif telco.PaymentMethod.iloc[0] == "Mailed check":
        telco["Bank transfer (automatic)"] = [0]
        telco["Credit card (automatic)"] = [0]
        telco["Electronic check"] = [0]
        telco["Mailed check"] = [1] 
    telco.drop(columns = ["PaymentMethod"],inplace = True)
    
    
    return telco


def prediction(model,X):
    #test_dt_pca = prepro_pca(test_dt_pf)
    #try:
    
    st.write("Our data: ")
    st.write(read_data())

    st.write("Our predict is ------> :")
    if model.predict(X) == 0:
        st.write("------ > NO CHURN < ------")
    elif model.predict(X) == 1:
        st.write("------ > CHURN! < ------")
        
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