import json

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

from ift6758.ift6758.client import serving_client, game_client

WANDB_PROJECT_NAME = "IFT6758.2024-A11"
WANDB_TEAM_NAME = "youry-macius-universite-de-montreal"


models_list = ["LogisticRegression_Distance", "LogisticRegression_Distance_Angle"]

BASE_URL = "https://api-web.nhle.com/v1/gamecenter"

PROVIDER_SERVICE_IP_ADDRESS = os.environ.get("PROVIDER_SERVICE_IP_ADDRESS", "127.0.0.1")

PROVIDER_SERVICE_URL = f"http://{PROVIDER_SERVICE_IP_ADDRESS}:5000"

if "actual_model" not in st.session_state:
    st.session_state.actual_model = "LogisticRegression_Distance"


def fetch_game_data(game_id):
    # Données simulées
    game_info = {
        'Game ID': game_id,
        'Teams': 'Team A vs Team B'
    }

    client = game_client.GameClient(base_url=BASE_URL, model_service_url=PROVIDER_SERVICE_URL,
                                    model_name=st.session_state.actual_model)

    data = client.process_game(game_id)

    return game_info, data


st.title("Hockey Visualization App")
game_id = st.text_input("Game ID", "2022030411")

with st.sidebar:

    workspace = WANDB_TEAM_NAME + "/" + WANDB_PROJECT_NAME

    st.sidebar.title("Menu")

    selected_workspace = st.sidebar.selectbox("Workspace", [workspace])

    selected_model = st.sidebar.selectbox("Model", models_list)

    versions_list = ["latest"]
    selected_version = st.sidebar.selectbox("Version", versions_list)

    if st.sidebar.button('Get model'):
        client = serving_client.ServingClient(ip=PROVIDER_SERVICE_IP_ADDRESS, model_name=st.session_state.actual_model)
        result = client.download_registry_model(workspace=selected_workspace, model=selected_model,
                                                version=selected_version)

        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

        st.write(f"Result status code: {result.status_code}")
        if result.status_code == 200:  # success
            st.session_state.actual_model = selected_model
            st.success(f"Model \n'{selected_model}' successfully loaded!")
        else:
            st.error("An error occurred while fetching the model.")

    st.write(f"Current Model: \n{st.session_state.actual_model}")

with st.container():
    # Bouton pour ping le jeu
    if st.button("Ping game"):
        # Récupérer les données du jeu et les informations du jeu
        game_info, game_data = fetch_game_data(game_id)

        teams = game_data['team']
        game_info['Teams'] = f"{teams[0]} VS {teams[1]}"
        game_info['Period'] = game_data["period"]
        game_info["Time left"] = game_data["time remaining"]

        # Display Metrics
        for team in teams:
            st.metric(
                label=f"{team} xG (Actual)",
                value=f"{game_data['xG'][team]:.2f} ({game_data['score'][team]})",
                delta=f"{game_data['xG diff'][team]:.2f}"
            )

        # Display Game Information
        st.subheader("Game Information")
        for key, value in game_info.items():
            st.write(f"{key}: {value}")

        # Display Predictions Data
        st.header("Data Used for Predictions (and Predictions)")
        predictions_df = pd.DataFrame(game_data["predictions"]).drop("team", axis=1)
        st.dataframe(predictions_df)
