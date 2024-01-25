# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 19:46:01 2024

@author: kaeli
"""

import streamlit as st
import pandas as pd
import pickle
from entrainement_modele_manuel import *

st.title("Rentrer le message que vous voulez tester")

if 'model' not in st.session_state:
    with open('modele_projet2.pkl', 'rb') as model_file:
        st.session_state.model = pickle.load(model_file)
    
if 'df_message' not in st.session_state:
    st.session_state.df_message = pd.DataFrame(columns=['contenus'])

st.session_state.user_input = st.text_input("Entrez le message que vous voulez tester ici:")

if st.button("Enregistrer"):
    st.session_state.df_inter_message=pd.DataFrame(data=[st.session_state.user_input],columns=['contenus'])
    st.session_state.df_message = pd.concat([st.session_state.df_message, st.session_state.df_inter_message])
    st.session_state.X=st.session_state.df_message['contenus']
    if st.session_state.X.empty:
        st.warning("Aucune donnée à prédire. Veuillez entrer des données.")
    else:
        st.session_state.predictions = st.session_state.model.predict(st.session_state.X.to_frame())
        st.session_state.df_message['Prediction'] = st.session_state.predictions

st.session_state.user_input2 = st.file_uploader('rentrer un csv contenant une liste de message que vous voulez tester')

st.dataframe(st.session_state.df_message, width=800)

if st.session_state.user_input2 is not None:
    st.session_state.df_inter_message=pd.read_csv(st.session_state.user_input2, names=['contenus'])
    st.session_state.df_message = pd.concat([st.session_state.df_message, st.session_state.df_inter_message])
    st.session_state.X=st.session_state.df_message['contenus']
    if st.session_state.X.empty:
        st.warning("Aucune donnée à prédire. Veuillez entrer des données.")
    else:
        st.session_state.predictions = st.session_state.model.predict(st.session_state.X.to_frame())
        st.session_state.df_message['Prediction'] = st.session_state.predictions
        
if st.button("Teste tes messages !"):
    st.session_state.df_message.to_csv("output.csv", index=False)