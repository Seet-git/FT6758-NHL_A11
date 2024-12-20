import numpy as np
import requests
import json
import os
from src import convert_game_to_dataframe
import pandas as pd
import pprint


class GameClient:
    def __init__(self, base_url, model_service_url):
        self.old_predictions = None
        self.base_url = base_url
        self.model_service_url = model_service_url

        if not os.path.exists("./data/predictions"):
            os.makedirs("./data/predictions")

    def _load_predictions(self, game_id):
        prediction_file = os.path.join("data/predictions", f"{game_id}_predictions.json")
        if os.path.exists(prediction_file):
            with open(prediction_file, 'r') as file:
                return pd.read_json(file, orient="records")
        return None

    def _save_predictions(self, game_id, predictions):
        if self.old_predictions is not None:
            predictions = pd.concat([self.old_predictions, predictions], axis=0, ignore_index=True)
        prediction_file = os.path.join("data/predictions", f"{game_id}_predictions.json")
        predictions.to_json(prediction_file, orient="records", indent=1)

    def fetch_game_data(self, game_id):
        try:
            response = requests.get(self.base_url + '/' + game_id + '/play-by-play')
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError("Fetch data error : ", e)

    def predict(self, features):
        try:
            features.columns = ['distance', 'angle']
            response = requests.post(f"{self.model_service_url}/predict", json=features.to_dict(orient="list"))
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError("Predict data error : ", e)

    def process_game(self, game_id):
        saved_predictions = self._load_predictions(game_id)

        data = self.fetch_game_data(game_id)
        game = convert_game_to_dataframe(data)
        game = game[["shotDistance", "shotAngle"]]

        if saved_predictions is not None:
            if len(game) > len(saved_predictions):
                print(f"Update predictions for game {game_id}")
                self.old_predictions = saved_predictions
                game = game.drop(index=np.arange(0, len(saved_predictions)))
                
            else:
                print(f"Updates not found for game {game_id}")
                return ""

        predictions = self.predict(game)
        predictions_df = pd.DataFrame(predictions)

        self._save_predictions(game_id, predictions_df)
        return predictions_df