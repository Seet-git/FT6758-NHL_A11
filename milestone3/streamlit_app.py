import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

from ift6758.ift6758.client import serving_client, game_client

# Ajouter le chemin du dossier `client` au `sys.path`
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ift6758', 'ift6758', 'client')))

# Importer le fichier `serving_client`
#import serving_client
#import game_client

"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

WANDB_PROJECT_NAME = "IFT6758.2024-A11"
WANDB_TEAM_NAME = "youry-macius-universite-de-montreal"

models_list = ["LogisticRegression_Distance", "LogisticRegression_Distance_Angle"]

BASE_URL = "https://api-web.nhle.com/v1/gamecenter"

PROVIDER_SERVICE_IP_ADDRESS = os.environ.get("PROVIDER_SERVICE_IP_ADDRESS", "127.0.0.1")

PROVIDER_SERVICE_URL = f"http://{PROVIDER_SERVICE_IP_ADDRESS}:5000"

# Initialisation de la session pour actual_model
if "actual_model" not in st.session_state:
    st.session_state.actual_model = "LogisticRegression_Distance"


def fetch_game_data(game_id):
    # Données simulées
    game_info = {
        'Game ID': game_id,
        'Date': '2024-12-21',
        'Teams': 'Team A vs Team B',
        'Location': 'Stadium XYZ'
    }

    client = game_client.GameClient(base_url=BASE_URL, model_service_url=PROVIDER_SERVICE_URL, model_name=st.session_state.actual_model)

    data = client.process_game(game_id)

    data = data.rename(columns={'goal_probs': 'Model output'})

    data.index = [f"event {i}" for i in range(len(data['Model output']))]

    return game_info, data

st.title("Hockey visualisation app")

with st.sidebar:
    # TODO: Add input for the sidebar
    workspace = WANDB_TEAM_NAME + "/" + WANDB_PROJECT_NAME

    # Créer une sidebar
    st.sidebar.title("Menu")

    # Ajouter une liste de sélection à la sidebar
    selected_workspace = st.sidebar.selectbox( "Workspace", [workspace])

    # Ajouter une liste de sélection à la sidebar
    selected_model = st.sidebar.selectbox("Model", models_list)

    # st.write(f"Vous avez sélectionné : {selected_model}")

    versions_list = ["latest"]

    # Ajouter une liste de sélection à la sidebar
    selected_version = st.sidebar.selectbox("Version", versions_list)

    if st.sidebar.button('Get model'):
        client = serving_client.ServingClient(ip=PROVIDER_SERVICE_IP_ADDRESS, model_name=st.session_state.actual_model)
        result = client.download_registry_model(workspace=selected_workspace, model=selected_model,
                                                version=selected_version)

        st.write(f"Result status code: {result.status_code}")
        if result.status_code == 200:  # success
            st.session_state.actual_model = selected_model
            st.success(f"Model '{selected_model}' successfully loaded!")
        else:
            st.error("An error occurred while fetching the model.")

    st.write(f"Current Model: {st.session_state.actual_model}")

with st.container():
    # Champ de saisie pour le Game ID
    game_id = st.text_input("Game ID", "2021020329")

    # Bouton pour ping le jeu
    if st.button("Ping game"):
        # Récupérer les données du jeu et les informations du jeu
        game_info, game_data = fetch_game_data(game_id)

        teams = game_data['team'].unique()
        game_info['Teams'] = f"{teams[0]} VS {teams[1]}"

        # Afficher les informations du jeu
        st.write("Game Information")
        for key, value in game_info.items():
            st.write(f"{key}: {value}")

        # Afficher les données du jeu
        st.header("Data used for predictions (and predictions)")
        game_data = game_data.drop("team", axis=1)
        st.dataframe(game_data)