
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:49:05 2024

@author: pierrecavallo
"""
# Importations:
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn import datasets
from sklearn import svm
import pandas as pd
import numpy as np
import string 
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
    roc_curve,
    RocCurveDisplay
)
# Downloads:
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

stopwords = nltk.corpus.stopwords.words('english')
trans = WordNetLemmatizer()

# Lecture du CSV

df= pd.read_csv("https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection", sep ="\t",header=None,names=['Type','Contenu'])
df=df.drop_duplicates()

# Fonction 

def tokenisation():
    
    df['Contenu'] = df['Contenu'].replace(to_replace='[^\w\s]', value=' ', regex=True)
    for i in range (len(df)):
        txt = df['Contenu'].iloc[i]
        #print(i,txt)
        tokens=word_tokenize(txt)
        #print(tokens)
        txt=txt.lower()
        tokenizer = RegexpTokenizer(r"[a-zA-Z]\w+\'?\w*")
        tokens = tokenizer.tokenize(txt)
        #print(tokens)
        tokens2=[token for token in tokens if token not in stopwords]
        #print("token 2: ",tokens2)
        trans_text = [trans.lemmatize(i) for i in tokens2]
        #print("lem :",trans_text)
        str=" ".join(trans_text)
        #print("str:",str)
        corpus.append(str)
        #print("-----------")
        
 

y = df["Type"]



label_encod = LabelEncoder()
label_encod.fit(y)
y=label_encod.transform(y)

# Tokenisation
corpus=[]
tokenisation()
#print(corpus)

# Vectorisation : 
vectorizer=CountVectorizer(ngram_range=(0,4))
X=vectorizer.fit_transform(corpus)
#print(vectorizer.get_feature_names_out())
#print(X.toarray())
X=X.toarray()

scoring=['recall','f1','precision_macro','recall_macro']
cv = StratifiedKFold(n_splits=10, shuffle=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20,stratify=y
)

model = MultinomialNB()


parameters = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}

grid_search = GridSearchCV(model, parameters, cv=cv, scoring=scoring, refit='recall')
grid_search.fit(X, y)

# Print the best parameters found by Grid Search
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)



# Predictions
y_pred = grid_search.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")
recall = recall_score(y_test, y_pred,pos_label=1)

# Affichage résultat Predictions

print("Test Accuracy:", accuray)
print("Test F1 Score:", f1)
print("Test Matrice de confusion",confusion_matrix(y_test, y_pred))
print("Test Recall:",recall)
print(scoring)




fpr, tpr, thresholds = roc_curve(y_test, y_pred,pos_label=1)
print("False positive rate:", fpr)
print("True positive rate:", tpr)
print("Thresholds:", thresholds)


RocCurveDisplay.from_predictions(y_test, y_pred,pos_label=1)
print("AUC: ", metrics.auc(fpr, tpr))
