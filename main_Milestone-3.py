import json
import os

import pandas as pd

from milestone3.ift6758.ift6758.client import *

WANDB_PROJECT_NAME = "IFT6758.2024-A11"
WANDB_TEAM_NAME = "youry-macius-universite-de-montreal"


def serving_test():
    print("\n=== Client fournisseur ===")
    model = "LogisticRegression_Distance"

    # Init
    client = ServingClient(ip="127.0.0.1", port=5000, model_name=model)

    # Download model
    print("\nTesting download model")
    workspace = WANDB_TEAM_NAME + "/" + WANDB_PROJECT_NAME

    version = "latest"

    result = client.download_registry_model(workspace=workspace, model=model, version=version)
    print("Model Download Result: ", result)

    # Predictions
    print("\nTesting Predict")
    df = pd.DataFrame({
        "distance": [8, 40],
        "angle": [20, 80],
    })
    predictions = client.predict(df)
    print("Predictions:", predictions)

    # Logs
    print("\nTesting Logs")
    logs = client.logs()
    print("Logs:", logs)


def game_client_test():
    print("\n=== Client de jeu ===")
    BASE_URL = "https://api-web.nhle.com/v1/gamecenter"
    MODEL_SERVICE_URL = "http://127.0.0.1:5000"
    MODEL_NAME = 'LogisticRegression_Distance_Angle'

    client = GameClient(BASE_URL, MODEL_SERVICE_URL, MODEL_NAME)
    game_id = "2022030411"

    # No predictions

    # Supprime le fichier
    path = f"./data/predictions/{MODEL_NAME}/{game_id}_predictions.json"
    if os.path.exists(path):
        os.remove(path)

    print("\nTesting no predictions")
    results = client.process_game(game_id)
    print(results)
    print("Process finished")

    # All predictions
    print("\nTesting all predictions already here")
    results = client.process_game(game_id)
    print(results)
    print("Process finished")

    # Update tracker

    # Supprimer les deux dernières prédictions
    with open(path, "r") as file:
        data = json.load(file)

    if len(data) >= 2:
        data = data[:-2]

    with open(path, "w") as file:
        json.dump(data, file, indent=1)

    print("\nTesting update tracker :")
    results = client.process_game(game_id)
    print(results)
    print("Process finished")


if __name__ == "__main__":
    serving_test()
    game_client_test()