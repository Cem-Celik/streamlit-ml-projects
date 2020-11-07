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


import re
import string
from textblob import TextBlob
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer





def read_data():
    df = pd.read_csv("emotion_whatsapp/real.csv")
    return df

def read_lrmodel():
    log_reg = joblib.load("emotion_whatsapp/joblib/log_reg.joblib")
    return log_reg

def read_knnmodel():
    knn = joblib.load("emotion_whatsapp/joblib/knn.joblib")
    return knn

def read_decision_tree():
    dt = joblib.load("emotion_whatsapp/joblib/dec_tree_lem.joblib")
    return dt

def read_random_forest():
    rf = joblib.load("emotion_whatsapp/joblib/rf.joblib")
    return rf

def read_xgb_model():
    xgb = joblib.load("emotion_whatsapp/joblib/xgb.joblib")
    return xgb

def read_svc_model():
    svc = joblib.load("emotion_whatsapp/joblib/svc.joblib")
    return svc


def wh(df):
    st.title("Web Whatsapp Emotion Project")
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
            model = read_svc_model()
            test_data = pd.DataFrame.from_dict(get_dummy())
            test_prepro = prepro(test_data)
            prediction(model,test_prepro)
    elif page_sub == "Exploration":
        dataExploration(df)

def get_dummy():
    dummy = {
        "content" : [st.sidebar.text_input("What is your status? ",value="never hurt people love lot hurt back probably choice leave forever")],
        }
    return dummy

def get_cons():
    contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"thx"   : "thanks"
}
    return contractions

def clean(text):
    contractions = get_cons()
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<,*?>+"','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    text = re.sub("xa0'", '', text)
    text = re.sub(u"\U00002019", "'", text) # IMPORTANT: Their apostrophe character was not the usual one...
    words = text.split()
    for i in range(len(words)):
        if words[i].lower() in contractions.keys():
            words[i] = contractions[words[i].lower()]
    text = " ".join(words)
    #text = TextBlob(text).correct()
    return text

def clean_html(text):
    
    clean = re.compile('<.*?>')
    return re.sub(clean, '',text)
    
def convert_lower(text):
    return text.lower()

def remove_special(text):
        x=''
        for i in text:
            if i.isalnum():
                x=x+i
            else:
                x=x+' '
        return x


def remove_stopwords(text):
    x=[]
    for i in text.split():
        
        if i not in stopwords.words('english'):
            x.append(i)
    y=x[:]
    x.clear()
    return y



def stem_words(text):
    ps= PorterStemmer()
    y=[]
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z

def join_back(list_input):
    return " ".join(list_input)
    
def joinback2(list_input):
    return "".join(list_input)
    


def prepro(df):
    
    #if which == "normal":
    whatsapp = x_normal(df)
    #else:
        #whatsapp = x_lemm(df)
    return whatsapp


def x_normal(df):
    
    df['content'] = df['content'].apply(lambda x: clean(x))
    # First try
    df['content'].replace('', np.nan, inplace=True)
    df = df.dropna(subset = ['content'])
    df = df.reset_index(drop=True)
    df['content']=df['content'].apply(clean_html)
    df['content']=df['content'].apply(convert_lower)
    df['content']=df['content'].apply(remove_special)
    df['content']=df['content'].apply(remove_stopwords)
    df['content']=df['content'].apply(join_back)
    df['content']=df['content'].apply(stem_words)
    df['content']=df['content'].apply(joinback2)
    cv=CountVectorizer(max_features=1500)
    X_smt = cv.fit_transform(df.content).toarray()
    return X_smt

def prediction(model,X):
    #test_dt_pca = prepro_pca(test_dt_pf)
    #try:
    
    st.write("Our data: ")
    st.write(read_data())
    st.write(X)
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