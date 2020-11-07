
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
    df = pd.read_csv("titanic/data/titanic.csv")
    return df

def read_lrmodel():
    log_reg = joblib.load("titanic/joblib/log_reg.joblib")
    return log_reg

def read_knnmodel():
    knn = joblib.load("titanic/joblib/knn.joblib")
    return knn

def read_decision_tree():
    dt = joblib.load("titanic/joblib/dt.joblib")
    return dt

def read_random_forest():
    rf = joblib.load("titanic/joblib/rf.joblib")
    return rf

def read_xgb_model():
    xgb = joblib.load("titanic/joblib/xgb.joblib")
    return xgb
def read_svcmodel():
    svc = joblib.load("titanic/joblib/svc.joblib")
    return svc
def read_pcamodel():
    pca = joblib.load("titanic/pca.joblib")
    return pca


def titanic_surv(df):
    st.title("Titanic Project")
    page_sub = st.sidebar.selectbox("Do something on the dataset", ["Prediction", "Exploration"])
    if page_sub == "Prediction":
        st.header("This is your prediction page.")
        st.write("Enter your values as you want.")
        page_model = st.sidebar.selectbox("Choose a Data Exploration", ["Logistic Regression","KNN","Decision Tree", "Random Forest","XGBoost","SVC"])
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
        elif page_model == "SVC":
            model = read_svcmodel()
            test_data = pd.DataFrame.from_dict(get_dummy())
            test_prepro = prepro(test_data)
            prediction(model,test_prepro)
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
        "Pclass" : [st.sidebar.selectbox("What is the passenger's ticket class? ",("First Class","Second Class","Third Class"),index=2)],
        "Name" : [st.sidebar.text_input("What is the passenger's name?",value = "Braund, Mr. Owen Harris")],
        "Sex" : [st.sidebar.selectbox('What is gender of passenger?',("Female","Male"),index = 1)],
        "Age" : [st.sidebar.number_input("What is passenger's age?",value = 22)],
        "SibSp" : [st.sidebar.number_input('Number of siblings / spouses aboard the Titanic',value = 1)],
        "Parch" : [st.sidebar.number_input("Number of parents / children aboard the Titanic",value = 0)],
        "ticket" : [st.sidebar.text_input("Ticket number?",value = "A/5 21171")],
        "Fare" : [st.sidebar.number_input("What is Passenger's fare?",value = 7.25)],
        "cabin" : [st.sidebar.text_input("What is passenger's cabin number?",value = " ")],
        "embarked" : [st.sidebar.selectbox("Port of Embarkation (Cherbourg, Queenstown, Southampton)",("Cherbourg", "Queenstown", "Southampton"),index = 1)],
        }
    return dummy

def prepro(titanic_df):
   
            
    if titanic_df.Pclass.iloc[0] == "First Class":
        titanic_df["Pclass"] = [0]
    elif titanic_df.Pclass.iloc[0]== "Second Class":
        titanic_df["Pclass"] = [1]
    elif titanic_df.Pclass.iloc[0] == "Third Class":
        titanic_df["Pclass"] = [2]
  
    titanic_df.drop(columns = ["Name"],inplace = True)
    
    if titanic_df.Sex.iloc[0] == "Male":
        titanic_df["is_female"] = [0]
        titanic_df["is_male"] = [1]
        
    elif titanic_df.Sex.iloc[0] == "Female":
        titanic_df["is_female"] = [1]
        titanic_df["is_male"] = [0]
    titanic_df.drop(columns=["Sex"],inplace=True)
    
    txt = titanic_df.ticket.iloc[0]
    lst=[]
    if txt == "LINE":
        txt = "0"
    df = txt.split(" ")
    if len(df) == 2:
        lst.append(df[1])
    elif len(df)>2:
        lst.append(df[2])
    else:
        lst.append(df[0])
        
    titanic_df["new_ticket"] = lst
    titanic_df["new_ticket"] = titanic_df.new_ticket.astype('float64')
    
    
    titanic_df.drop(columns="ticket",inplace=True)
    titanic_df.drop(columns="cabin",inplace=True)
    
    if titanic_df.embarked.iloc[0] == "Cherbourg":
        titanic_df["Cherbourg"] = [1]
        titanic_df["Queenstown"] = [0]
        titanic_df["Southampton"] = [0]
        
    elif titanic_df.embarked.iloc[0] == "Southampton":
        titanic_df["Cherbourg"] = [0]
        titanic_df["Queenstown"] = [0]
        titanic_df["Southampton"] = [1]
        
    elif titanic_df.embarked.iloc[0] == "Queenstown":
        titanic_df["Cherbourg"] = [0]
        titanic_df["Queenstown"] = [1] 
        titanic_df["Southampton"] = [0]
        
    titanic_df.drop(columns = ["embarked"],inplace = True)
    return titanic_df


def prediction(model,X):
    #test_dt_pca = prepro_pca(test_dt_pf)
    #try:
    st.write("Our data: ")
    st.write(read_data())
    st.write("Our predict is ------> :")
    if model.predict(X) == 0:
        st.write("------ > Not Survived < ------")
    elif model.predict(X) == 1:
        st.write("------ > Survived! < ------")
        
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