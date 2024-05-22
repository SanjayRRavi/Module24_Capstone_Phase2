import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
import joblib

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline


## function for lemmatizing
def lemmatize(text):
    text = text[1:-1]  # delete "[" and "]"
    l1 = text.split(', ') # split string into a list
    lemma = WordNetLemmatizer()
    lemmatized_words = [lemma.lemmatize(w) for w in l1]
    lemma_str = ' '.join(lemmatized_words)
    
    return lemma_str


## Load classifier machine learning (ML) model using joblib
loaded_model = joblib.load('lgr_model.sr')


# Set the app title 
st.title('Prediction of Normal vs. Anomaly')

# Create a text input
user_input_dict = dict()
key1 = 'InputSequence'
user_input_dict[key1] = st.text_input(label=key1)

## Make predictions based on input sequence user will enter on Streamlit webpage
if st.button("Predict!"):
    #st.write(user_input_dict[key1])
    X1 = lemmatize(user_input_dict[key1])
    #st.write(X1)
    #y_pred = loaded_model.predict(pd.DataFrame(user_input_dict, index = [0]))
    y_pred = loaded_model.predict([X1])
    
    #st.write(y_pred)
    # prediction_dict = json.loads(response.text)
    if y_pred[0] == 1:
        st.write("Prediction will be anomaly")
    else:
        st.write("Prediction will be normal")

