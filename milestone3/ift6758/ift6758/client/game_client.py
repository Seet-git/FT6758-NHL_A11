import json
import requests
import os
# from src import convert_game_to_dataframe
from .game_client_utils import *

struct_model = {
    "LogisticRegression_Distance": "distance",
    "LogisticRegression_Distance_Angle": "angle"
}


class GameClient:
    def __init__(self, base_url, model_service_url, model_name):
        self.old_predictions = None
        self.base_url = base_url
        self.model_service_url = model_service_url
        self.model_name = struct_model[model_name]
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(PROJECT_ROOT, '../../../../', 'data', 'predictions', model_name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def make_serializable(self, obj):
        if isinstance(obj, (np.int64, np.float64)):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(v) for v in obj]
        return obj

    def _load_predictions(self, game_id):
        prediction_file = os.path.join(self.model_path, f"{game_id}_predictions.json")
        if os.path.exists(prediction_file):
            with open(prediction_file, 'r') as file:
                return json.load(file)
        return None

    def _save_predictions(self, game_id, result):
        prediction_file = os.path.join(self.model_path, f"{game_id}_predictions.json")
        result['predictions'] = result['predictions'].to_dict(orient='records')
        result = {k: self.make_serializable(v) for k, v in result.items()}
        with open(prediction_file, 'w') as f:
            json.dump(result, f, indent=4, default=str)

    def fetch_game_data(self, game_id):
        try:
            response = requests.get(self.base_url + '/' + game_id + '/play-by-play')
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError("Fetch data error : ", e)

    def predict(self, features):
        try:
            if self.model_name == "distance":
                features = pd.DataFrame(features["distance"])
            else:
                features = features[["distance", "angle"]]
            response = requests.post(f"{self.model_service_url}/predict", json=features.to_dict(orient="list"))
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError("Predict data error : ", e)

    def process_results(self, predictions_df):
        if self.old_predictions is not None:
            predictions_df = pd.concat([self.old_predictions, predictions_df], axis=0, ignore_index=True)

        teams = predictions_df['team'].unique()

        xg_totals = predictions_df.groupby('team')['Model output'].sum().to_dict()

        period_seconds = 20 * 60
        current_period = predictions_df['numberPeriod'].iloc[-1]
        elapsed = predictions_df['gameSeconds'].iloc[-1] - (
                current_period - 1) * period_seconds

        elapsed = max(0, min(period_seconds, elapsed))

        time_remaining_seconds = period_seconds - elapsed
        time_remaining = f"{time_remaining_seconds // 60:.0f}m {time_remaining_seconds % 60:.0f}s"

        scores = predictions_df[predictions_df['isGoal'] == 1]['team'].value_counts().to_dict()

        score_xg_diff = {
            team: round(xg_totals.get(team, 0) - scores.get(team, 0), 2)
            for team in teams
        }

        results = {
            "team": teams,
            "period": predictions_df['numberPeriod'].max(),
            "time remaining": time_remaining,
            "score": scores,
            "xG": xg_totals,
            "xG diff": score_xg_diff,
            "predictions": predictions_df[["team", "distance", "Model output"]]
        }

        if self.model_name != "distance":
            results["predictions"] = predictions_df[["team", "distance", "angle", "Model output"]]

        return results

    def process_game(self, game_id):
        saved_predictions = self._load_predictions(game_id)

        data = self.fetch_game_data(game_id)
        game = convert_game_to_dataframe(data)
        game = game.rename(columns={"eventOwnerTeam": "team", "shotDistance": "distance", "shotAngle": "angle"})
        if self.model_name == "distance":
            game = game.drop('angle', axis=1)

        if saved_predictions is not None:
            if "predictions" in saved_predictions and isinstance(saved_predictions["predictions"], list):
                predictions = saved_predictions["predictions"].copy()
                predictions_length = len(predictions)

                if len(game) > predictions_length:
                    print(f"Tracker updated for game {game_id}")
                    self.old_predictions = saved_predictions
                    game = game.drop(index=np.arange(0, predictions_length))
                    game = game.reset_index(drop=True)
                else:
                    print(f"Updates not found for game {game_id}")
                    return saved_predictions
            else:
                print("Invalid format in saved_predictions.")

        predictions = self.predict(game)
        predictions_df = pd.DataFrame(predictions)
        predictions_df = pd.concat([game, predictions_df], axis=1)
        predictions_df = predictions_df.drop('is_goal', axis=1)
        predictions_df = predictions_df.rename(columns={'goal_probs': 'Model output'})
        predictions_df.index = [f"event {i}" for i in range(len(predictions_df['Model output']))]

        results = self.process_results(predictions_df)
        self._save_predictions(game_id, results)

        return results
