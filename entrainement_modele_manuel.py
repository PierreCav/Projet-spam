# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:09:28 2024

@author: kaeli
"""
import time
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import re
import math
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score)
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn import metrics
import streamlit as st
import pickle

nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download("punkt")

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
sia = SentimentIntensityAnalyzer()


def mot(a):
    count=0
    for i in range(len(a)):
        if a[i] ==' ' and a[(i-1)]!=' ':
            count+=1
        if a[len(a)-1]!=' ' and i ==(len(a)-1):
            count+=1
    return count

def decoupage_mot(a,formatage=False,stemmer=None,tokenize=None):
    b=['&','"',"'",'-','_',')','=','~','#','{','[','|','`',"^","@","]","}","°","+",",",";",":","!","?",".","/","§","*","$","£","%","µ"]
    
    if isinstance(a, list):
        a=" ".join(a)
    if isinstance(a, str) and formatage is False :
        c=a.split()
    if isinstance(a, str) and formatage is True and tokenize ==None:
        for i in b:
            a=a.replace(i,' ')
        a=a.lower()
        c=a.split()
    if tokenize:
        c=word_tokenize(a)
    if stemmer:
        c = [stemmer.stem(word) for word in c]   
    return c

def caract_spec(a):
    b={'&','é','"',"'",'-','_',')','=','~','#','{','[','|','`',"^","@","]","}","°","+",",",";",":","!","?",".","/","§","*","$","£","%","µ"}
    count=0
    for i in a:
        if i in b:
            count+=1
    return count

def maj(a):
    b=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    count=0
    for i in a:
        if i in b:
            count+=1
    return count

def numeros_tel(a):
    count=0
    for i in a:
        if i.isnumeric():
            count+=1
            if count==2:
                return True
        else :
            count=0
    return False

def lien(a):
    b=['http://','https://','www']
    for i in b:
        if i in a:
            return True
    else:
        return False
    
def argent(a):
    b=['£','$','€']
    for i in b:
        if i in a:
            return True
    else:
        return False
    


def mot_to_colonne(df_liste_mot, colonne1, dataframe, colonne2):
    #start = time.time()
    if colonne1 == 'column':
        mots_a_rechercher = set(str(x) for x in df_liste_mot.columns)
    elif colonne1 != 'column':
        mots_a_rechercher = set(df_liste_mot[colonne1].apply(lambda x: ' '.join(x) if isinstance(x, tuple) else x))

    dataframe_copy = dataframe.copy()
    dataframe_copy[colonne2] = dataframe_copy[colonne2].apply(lambda x: ' '.join(decoupage_mot(x, formatage=True, stemmer=ps, tokenize=1)))

    temp_df = {mot: dataframe_copy[colonne2].str.contains(fr'\b{re.escape(str(mot))}\b', case=False, regex=True) for mot in mots_a_rechercher}
    
    dataframe_copy = pd.concat([dataframe_copy, pd.DataFrame(temp_df)], axis=1)
    #print(time.time() - start)
    return dataframe_copy


mots_filtrés = pd.read_csv("mots_filtrés.csv")
selection_spam_mot = pd.read_csv("selection_spam_mot.csv")
selection_ham_mot = pd.read_csv("selection_ham_mot.csv")
selection_spam_2_gramm = pd.read_csv("selection_spam_2_gramm.csv")
selection_ham_2_gramm = pd.read_csv("selection_ham_2_gramm.csv")
selection_spam_3_gramm = pd.read_csv("selection_spam_3_gramm.csv")
selection_ham_3_gramm = pd.read_csv("selection_ham_3_gramm.csv")
selection_spam_4_gramm = pd.read_csv("selection_spam_4_gramm.csv")
selection_ham_4_gramm = pd.read_csv("selection_ham_4_gramm.csv")
Spam1= pd.read_csv("Spam.csv")
#%%%




def creation_colonnes(df):
    df=mot_to_colonne(mots_filtrés,'column',df,'contenus')
    df=mot_to_colonne(selection_spam_mot,'Mot',df,'contenus')
    df=mot_to_colonne(selection_ham_mot,'Mot',df,'contenus')
    df=mot_to_colonne(selection_spam_2_gramm,'0',df,'contenus')
    df=mot_to_colonne(selection_ham_2_gramm,'0',df,'contenus')
    df=mot_to_colonne(selection_spam_3_gramm,'0',df,'contenus')
    df=mot_to_colonne(selection_ham_3_gramm,'0',df,'contenus')
    df=mot_to_colonne(selection_spam_4_gramm,'0',df,'contenus')
    df=mot_to_colonne(selection_ham_4_gramm,'0',df,'contenus')

    df['nb maj']=df['contenus'].apply(maj)
    df['num_tel']=df['contenus'].apply(numeros_tel)
    df['lien']=df['contenus'].apply(lien)
    df['argent']=df['contenus'].apply(argent)
    df['nb charact']=df['contenus'].apply(lambda x: len(str(x)))
    df['nb carac spec']=df['contenus'].apply(caract_spec)
    if 'nb mot' in df.columns:
        df=df.drop(labels='nb mot',axis=1)
    df['nb mot']=df['contenus'].apply(lambda x: mot(x))
    df['ratio charact/mot']=df['nb charact']/df['nb mot']
    df['nb maj']=df['nb maj'].apply(lambda x: True if x>25 else False)
    df['nb charact']=df['nb charact'].apply(lambda x: True if x>95 else False)
    df['nb mot']=df['nb mot'].apply(lambda x: True if x>20 else False)
    df['nb carac spec']=df['nb carac spec'].apply(lambda x: True if x>4 else False)
    df['sentiment']=df['contenus'].apply(lambda texte: sia.polarity_scores(texte)['compound'])
    df=df.drop(labels=['contenus'],axis=1)
    return df

from sklearn.decomposition import TruncatedSVD
svc=SVC(kernel='linear',C=1)
encod = Pipeline(steps=[('words', FunctionTransformer(creation_colonnes))])
preprocessor = ColumnTransformer(transformers=[('encod',encod, ['contenus'])])
model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', svc)])

y= Spam1.iloc[:,0]
X = Spam1.drop(labels=['sorte'],axis=1 , errors='ignore')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,stratify=Spam1['sorte'])

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
'''



model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_scores = model.decision_function(X_test)

print(confusion_matrix(y_test, y_pred))
accuray = accuracy_score(y_pred, y_test)
recall=recall_score(y_pred,y_test,pos_label='spam')
print("Accuracy:", accuray)
print("recall:",recall)

'''
with open('modele_projet2.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
