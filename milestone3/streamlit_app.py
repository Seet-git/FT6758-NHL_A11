import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# from milestone3.ift6758.ift6758.client import *

# Ajouter le chemin du dossier `client` au `sys.path`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ift6758', 'ift6758', 'client')))

# Importer le fichier `serving_client`
import serving_client

"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

st.title("Hockey visualisation app")

WANDB_PROJECT_NAME = "IFT6758.2024-A11"
WANDB_TEAM_NAME = "youry-macius-universite-de-montreal"

actual_model = "LogisticRegression_Distance"
models_list = ["LogisticRegression_Distance", "LogisticRegression_Distance_Angle"]

with st.sidebar:
    # TODO: Add input for the sidebar
    workspace = WANDB_TEAM_NAME + "/" + WANDB_PROJECT_NAME

    # Créer une sidebar
    st.sidebar.title("Menu")

    # Ajouter une liste de sélection à la sidebar
    selected_workspace = st.sidebar.selectbox( "Workspace", [workspace])

    # Afficher la sélection
    st.write(f"Vous avez sélectionné : {selected_workspace}")

    # Créer une sidebar
    st.sidebar.title("Menu")

    # Ajouter une liste de sélection à la sidebar
    selected_model = st.sidebar.selectbox("Model", models_list)

    st.write(f"Vous avez sélectionné : {selected_model}")

    versions_list = ["latest"]

    # Ajouter une liste de sélection à la sidebar
    selected_version = st.sidebar.selectbox("Version", versions_list)

    st.write(f"Vous avez sélectionné : {selected_version}")

    if st.sidebar.button('Get model'):
        # have_it = animal.lower() in animal_shelter
        # 'We have that animal!' if have_it else 'We don\'t have that animal.'

        client = serving_client.ServingClient(ip="127.0.0.1")

        result = client.download_registry_model(workspace=selected_workspace, model=selected_model, version=selected_version)

        if result.status_code == 200: # success
            actual_model = selected_model
        else:
            st.error("Une érreur est survenue lors de la récupération du modèle.")

    st.write(f"Modèle actif : {actual_model}")

with st.container():
    # TODO: Add Game ID input
    pass

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass