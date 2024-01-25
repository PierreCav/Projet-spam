# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:29:32 2024

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
from sklearn.naive_bayes import BernoulliNB
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

#%% Fonction de preprocessing global (train/final)
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
            if count==3:
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
    
def mot_to_colonne1(df_liste_mot, colonne1, dataframe, colonne2):
    dataframe_copy = dataframe.copy()

    mots_a_rechercher = set()
    if colonne1 == 'column':
        for i in df_liste_mot.columns:
            if isinstance(i, str):
                mots_a_rechercher.add(i)
            elif isinstance(i, tuple):
                mots_a_rechercher.add(' '.join(i))  
                
    elif colonne1 != 'column':
        for i in df_liste_mot[colonne1]:
            if isinstance(i, str):
                mots_a_rechercher.add(i)
            elif isinstance(i, tuple):
                mots_a_rechercher.add(' '.join(i))
       

    dataframe_copy[colonne2] = dataframe_copy[colonne2].apply(lambda x: ' '.join(decoupage_mot(x, formatage=True, stemmer=ps,tokenize=1)))
    temp_df = pd.DataFrame()
    

    for mot1 in mots_a_rechercher:
        mot_escaped = re.escape(mot1)
        temp_df[mot1] = dataframe_copy[colonne2].str.contains(fr'\b{mot_escaped}\b', case=False, regex=True)
    temp_df1=temp_df.copy()
    dataframe_copy_copy=dataframe_copy.copy()
    dataframe_copy_copy = pd.concat([dataframe_copy_copy, temp_df1], axis=1)
    return dataframe_copy_copy


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

#%% fonction de preprocessing train

def tableau_n_gramme(df,colonne,nb_n,nb_head=None,stemmer=None):
    liste1, liste2 =[],[]
    dico={}
    df_copy=df.copy()
    if stemmer:
       #df_copy[colonne] = df_copy[colonne].apply(filtre_stopwords)
        df_copy['Contenus_mot']=df_copy[colonne].apply(lambda x: decoupage_mot(x, formatage=True,stemmer=ps,tokenize=1))
    else:
       #df_copy[colonne] = df_copy[colonne].apply(filtre_stopwords)
        df_copy['Contenus_mot']=df_copy[colonne].apply(lambda x: decoupage_mot(x, formatage=True,tokenize=1))
        
    liste1.extend(df_copy['Contenus_mot'].apply(lambda x: relation(x,nb_n)).tolist())
    for i in liste1:
        for j in i:
            liste2.append(j)
            
    for element in liste2:
        key = tuple(element)
        if key in dico:
            dico[key]+=1
        else:
            dico[key]=1   
    df_final=pd.DataFrame(list(dico.items()))
    df_final=df_final.sort_values(by=1,ascending=False)  
    if nb_head:
        df_final=df_final.head(nb_head)
    return df_final

def relation(a,b):
    count=0
    c=[]
    for i in range(len(a)-(b-1)):
        c.append([])
    for i in range(len(a)):
        if i>=(b-1):
            for j in range((b-1),-1,-1):
                c[count].append(a[i-j])
            count+=1
    return c

def comptage_occurence(df,colonne,nb=None,stemmer=None):
    a=list(df[colonne].values.tolist())
    if stemmer:
        a=decoupage_mot(a,formatage=True,stemmer=ps,tokenize=1)
    else:
        a=decoupage_mot(a,formatage=True,tokenize=1)

    df_a=pd.DataFrame(data=a)
    dico_a = df_a[0].value_counts().to_dict()
    
    df_a=pd.DataFrame(list(dico_a.items()), columns=['Mot', 'Occurrences'])
    df_a=df_a.sort_values(by=['Occurrences'],ascending=False)
    if nb:
        df_a=df_a.head(nb)
    return df_a

def mot_to_colonne_count1(df_liste_mot, colonne1, dataframe, colonne2):
    dataframe_copy = dataframe.copy()

    mots_a_rechercher = set()
    for i in df_liste_mot[colonne1]:
        if isinstance(i, str):
            mots_a_rechercher.add(i)
        elif isinstance(i, tuple):
            mots_a_rechercher.add(' '.join(i))

    dataframe_copy[colonne2] = dataframe_copy[colonne2].apply(lambda x: ' '.join(decoupage_mot(x, formatage=True, stemmer=ps,tokenize=1)))

    for mot2 in mots_a_rechercher:
        mot_escaped = re.escape(mot2)
        dataframe_copy[mot2] = dataframe_copy[colonne2].str.count(fr'\b{mot_escaped}\b')

    print('fini')
    return dataframe_copy

def mot_to_colonne_count(df_liste_mot, colonne1, dataframe, colonne2):
    dataframe_copy = dataframe.copy()

    mots_a_rechercher = set()
    for i in df_liste_mot[colonne1]:
        if isinstance(i, str):
            mots_a_rechercher.add(i)
        elif isinstance(i, tuple):
            mots_a_rechercher.add(' '.join(i))

    dataframe_copy[colonne2] = dataframe_copy[colonne2].apply(lambda x: ' '.join(decoupage_mot(x, formatage=True, stemmer=ps, tokenize=1)))

    def count_occurrences(text, word):
        return text.count(word)

    for word in mots_a_rechercher:
        dataframe_copy[word] = dataframe_copy[colonne2].apply(lambda x: count_occurrences(x, word))

    return dataframe_copy

def filtre_stopwords(a):
    mot2 = decoupage_mot(a,stemmer=ps,tokenize=1)
    filtered_words = [mot for mot in mot2 if mot.lower() not in stop_words]
    return ' '.join(filtered_words)




def TF_IDF(df, group_column='sorte', weight_ratio=None):
    def computeTF(df, group_column):
        df['nb mot'] = df['contenus'].apply(mot)

        df=df.set_index('contenus')
        df_sorte=df['sorte']
        term_columns = df.columns[1:]
        df = df[term_columns].div(df.loc[:, 'nb mot'], axis=0)
        
        df = df.drop(labels='nb mot', axis=1)
        df=pd.concat([df,df_sorte],axis=1)
        df=df.reset_index()
        return df

    def computeIDF(df):
        if 'contenus' in df.columns:
            df = df.drop(labels=['contenus'], axis=1)
        if group_column in df.columns:
            df = df.drop(labels=[group_column], axis=1)
        N = len(df)
        df_gt_zero = df.applymap(lambda x: x > 0)
        df_df = df_gt_zero.sum()

        idfDict = df_df.apply(lambda df: math.log10(N / (df + 1)))
        return idfDict.to_dict()

    def computeTFIDF(df, idfs, weight_ratio, group_column='sorte'):
        count=0
        colonne_a_exclure='contenus'
        if weight_ratio:
            for i in df[group_column]:
                if i == 'ham':
                    count+=1
            ham_ratio = count / len(df)
            print(f"ham_ratio ={ham_ratio}")
            spam_ratio = 1 - ham_ratio
            print(f"spam_ratio ={spam_ratio}")
            df[group_column] = df[group_column].apply(lambda x: ham_ratio if x == 'ham' else spam_ratio)
            df[df.columns.difference([colonne_a_exclure])] = df[df.columns.difference([colonne_a_exclure])].apply(lambda x: x*df[group_column])
            
        for word, val in idfs.items():
            df[word] = val * df[word]
        return df
    df_copy = df.copy()
    df_copy1 = df.copy()
    df_computeTF = computeTF(df_copy,group_column)
    idfs = computeIDF(df_copy1)
    df_computeTF1=df_computeTF.copy()
    resultat_tf_IDF = computeTFIDF(df_computeTF1, idfs, weight_ratio, group_column)
    return resultat_tf_IDF

def selection_merge(df1,df2,df2_all,colonne,colonne2):
    df = pd.merge(df1, df2, how='left',on=[colonne], indicator=True).query("_merge == 'left_only'").drop('_merge', axis=1)
    df.drop(labels=f"{colonne2}_y",axis=1,inplace=True)
    df = pd.merge(df, df2_all, how='outer',on=[colonne])

    df.dropna(subset=f"{colonne2}_x",inplace=True)
    df.drop
    return df

#%% Debut train

Spam = pd.read_csv("https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection", delimiter='\t', header=None)
Spam_copy=Spam.copy()
Spam1=Spam_copy.drop_duplicates()
Spam1=Spam1.rename(columns={0:'sorte'})
Spam1=Spam1.rename(columns={1:'contenus'})
Spam1['contenus']=Spam1['contenus'].apply(lambda x: x.replace('&lt;', 'name1'))
Spam1['contenus']=Spam1['contenus'].apply(lambda x: x.replace('#&gt;', 'name2'))

only_spam=Spam1[Spam1.sorte.str.contains('spam')]
only_ham=Spam1[Spam1.sorte.str.contains('ham')]

#%% Creation bag of words
#1 liste de mot les plus vue dans les documents
print('start')
count_mot_spam_max=comptage_occurence(only_spam,'contenus',700,stemmer=1)
count_mot_spam_max1=comptage_occurence(only_spam,'contenus',700)
count_mot_spam=comptage_occurence(only_spam,'contenus',stemmer=1)

count_mot_ham_max=comptage_occurence(only_ham,'contenus',300,stemmer=1)
count_mot_ham_max1=comptage_occurence(only_ham,'contenus',300)
count_mot_ham=comptage_occurence(only_ham,'contenus',stemmer=1)

count_mot=comptage_occurence(Spam1,'contenus',stemmer=1)

selection_spam_mot=selection_merge(count_mot_spam_max,count_mot_ham_max,count_mot_ham,'Mot','Occurrences')    
selection_ham_mot=selection_merge(count_mot_ham_max,count_mot_spam_max,count_mot_spam,'Mot','Occurrences')  
print('end')
#2 Utilisation du TF-TDF
 #1 prepartion pour le TF-TDF
df_final_tot=mot_to_colonne(count_mot,'Mot',Spam1,'contenus')
total = set(count_mot.Mot.tolist())
spam_copy1=Spam.copy()
df_final_tot_count=mot_to_colonne_count(count_mot,'Mot',Spam1,'contenus')

 #2 Application TF-TDF
df_total_mot_TF_IDF=TF_IDF(df_final_tot_count)
if 'contenus' in df_total_mot_TF_IDF.columns:
    df_total_mot_TF_IDF=df_total_mot_TF_IDF.drop('contenus',axis=1)
if 'sorte' in df_total_mot_TF_IDF.columns:
    df_total_mot_TF_IDF=df_total_mot_TF_IDF.set_index('sorte')
    
 #3 Filtrage des mot grace au TF-TDF
seuil_tfidf_max = 1
masque_filtrage =  df_total_mot_TF_IDF > seuil_tfidf_max
mots_filtrés = df_total_mot_TF_IDF.loc[:, masque_filtrage.any(axis=0)]
df_total_mot_TF_IDF=df_total_mot_TF_IDF.reset_index()

for mot1 in selection_ham_mot.Mot:
    if mot1 in mots_filtrés.columns:
        mots_filtrés=mots_filtrés.drop(labels=[mot1],axis=1)

for mot2 in selection_spam_mot.Mot:
    if mot2 in mots_filtrés.columns:
        mots_filtrés=mots_filtrés.drop(labels=[mot2],axis=1)

#%%Creation des n-grammes
print('commence n gramme')
relation_mot_spam_max2=tableau_n_gramme(only_spam,'contenus',2,400,stemmer=1)
relation_mot_ham_max2=tableau_n_gramme(only_ham,'contenus',2,400,stemmer=1)

relation_mot_spam2=tableau_n_gramme(only_spam,'contenus',2,stemmer=1)
relation_mot_ham2=tableau_n_gramme(only_ham,'contenus',2,stemmer=1)

relation_mot_spam_max3=tableau_n_gramme(only_spam,'contenus',3,400,stemmer=1)
relation_mot_ham_max3=tableau_n_gramme(only_ham,'contenus',3,400,stemmer=1)

relation_mot_spam3=tableau_n_gramme(only_spam,'contenus',3,stemmer=1)
relation_mot_ham3=tableau_n_gramme(only_ham,'contenus',3,stemmer=1)

relation_mot_spam_max4=tableau_n_gramme(only_spam,'contenus',4,400,stemmer=1)
relation_mot_ham_max4=tableau_n_gramme(only_ham,'contenus',4,400,stemmer=1)

relation_mot_spam4=tableau_n_gramme(only_spam,'contenus',4,stemmer=1)
relation_mot_ham4=tableau_n_gramme(only_ham,'contenus',4,stemmer=1)

selection_spam_2_gramm=selection_merge(relation_mot_spam_max2,relation_mot_ham_max2,relation_mot_ham2,0,1)    
selection_ham_2_gramm=selection_merge(relation_mot_ham_max2,relation_mot_spam_max2,relation_mot_spam2,0,1) 

selection_spam_3_gramm=selection_merge(relation_mot_spam_max3,relation_mot_ham_max3,relation_mot_ham3,0,1)    
selection_ham_3_gramm=selection_merge(relation_mot_ham_max3,relation_mot_spam_max3,relation_mot_spam3,0,1) 

selection_spam_4_gramm=selection_merge(relation_mot_spam_max4,relation_mot_ham_max4,relation_mot_ham4,0,1)    
selection_ham_4_gramm=selection_merge(relation_mot_ham_max4,relation_mot_spam_max4,relation_mot_spam4,0,1) 


chemin_fichier_csv = 'mots_filtrés.csv'
chemin_fichier_csv1 = 'selection_spam_mot.csv'
chemin_fichier_csv2 = 'selection_ham_mot.csv'
chemin_fichier_csv3 = 'selection_spam_2_gramm.csv'
chemin_fichier_csv4 = 'selection_ham_2_gramm.csv'
chemin_fichier_csv5 = 'selection_spam_3_gramm.csv'
chemin_fichier_csv6 = 'selection_ham_3_gramm.csv'
chemin_fichier_csv7 = 'selection_spam_4_gramm.csv'
chemin_fichier_csv8 = 'selection_ham_4_gramm.csv'
chemin_fichier_csv9 = 'Spam.csv'

selection_spam_mot.to_csv(chemin_fichier_csv1, index=False)
mots_filtrés.to_csv(chemin_fichier_csv, index=False)
selection_ham_mot.to_csv(chemin_fichier_csv2, index=False)
selection_spam_2_gramm.to_csv(chemin_fichier_csv3, index=False)
selection_ham_2_gramm.to_csv(chemin_fichier_csv4, index=False)
selection_spam_3_gramm.to_csv(chemin_fichier_csv5, index=False)
selection_ham_3_gramm.to_csv(chemin_fichier_csv6, index=False)
selection_spam_4_gramm.to_csv(chemin_fichier_csv7, index=False)
selection_ham_4_gramm.to_csv(chemin_fichier_csv8, index=False)
Spam1.to_csv(chemin_fichier_csv9, index=False)

print('fini n-gramme')
#%% Fin de preparation des données pour l'entrainement

start = time.time()
def creation_colonnes(df):
    df=mot_to_colonne(mots_filtrés,'column',df,'contenus')
    df=mot_to_colonne(selection_spam_mot,'Mot',df,'contenus')
    df=mot_to_colonne(selection_ham_mot,'Mot',df,'contenus')
    df=mot_to_colonne(selection_spam_2_gramm,0,df,'contenus')
    df=mot_to_colonne(selection_ham_2_gramm,0,df,'contenus')
    df=mot_to_colonne(selection_spam_3_gramm,0,df,'contenus')
    df=mot_to_colonne(selection_ham_3_gramm,0,df,'contenus')
    df=mot_to_colonne(selection_spam_4_gramm,0,df,'contenus')
    df=mot_to_colonne(selection_ham_4_gramm,0,df,'contenus')

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
from sklearn.naive_bayes import MultinomialNB

svc=SVC(kernel='linear',C=0.2)
encod = Pipeline(steps=[('words', FunctionTransformer(creation_colonnes))])
preprocessor = ColumnTransformer(transformers=[('encod',encod, ['contenus'])])
model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', svc)])

y= Spam1.iloc[:,0]
X = Spam1.drop(labels=['sorte'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,stratify=Spam1['sorte'])


model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_scores = model.decision_function(X_test)

print(confusion_matrix(y_test, y_pred))
accuray = accuracy_score(y_pred, y_test)
recall=recall_score(y_pred,y_test,pos_label='spam')
print("Accuracy:", accuray)

print("recall:",recall)








