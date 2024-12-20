import requests
import json
import pandas as pd

from src import convert_game_to_dataframe
import pprint


class GameClient:
    def __init__(self, base_url, model_service_url):
        self.base_url = base_url
        self.model_service_url = model_service_url
        self.processed_events = set()

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
            raise ValueError("Fetch data error : ", e)

    def process_game(self, game_id):
        data = self.fetch_game_data(game_id)
        game = convert_game_to_dataframe(data)
        game = game[["shotDistance", "shotAngle"]]
        predictions = self.predict(game)
        return predictions


if __name__ == "__main__":
    BASE_URL = "https://api-web.nhle.com/v1/gamecenter"
    MODEL_SERVICE_URL = "http://127.0.0.1:5000"

    client = GameClient(BASE_URL, MODEL_SERVICE_URL)
    game_id = "2022030411"

    results = client.process_game(game_id)
    print(pprint.pprint(results))
    print("Process finished")
