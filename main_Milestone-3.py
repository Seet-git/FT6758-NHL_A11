import pandas as pd

from milestone3.ift6758.ift6758.client import *

WANDB_PROJECT_NAME = "IFT6758.2024-A11"
WANDB_TEAM_NAME = "youry-macius-universite-de-montreal"


def serving_test():
    # Init
    client = ServingClient(ip="127.0.0.1", port=5000)

    # Download model
    print("\nTesting download model")
    workspace = WANDB_TEAM_NAME + "/" + WANDB_PROJECT_NAME
    model = "LogisticRegression_Distance_Angle"
    version = "latest"

    try:
        result = client.download_registry_model(workspace=workspace, model=model, version=version)
        print("Model Download Result:")
        print(result)
    except Exception as e:
        print(f"Error downloading model: {e}")

    # Predictions
    print("\nTesting Predict")
    df = pd.DataFrame({
        "distance": [8, 40],
        "angle": [20, 80],
    })

    try:
        predictions = client.predict(df)
        print("Predictions:")
        print(predictions)
    except Exception as e:
        print(f"Error prediction: {e}")

    # Logs
    print("\nTesting Logs")
    try:
        logs = client.logs()
        print("Logs:", logs)
    except Exception as e:
        print(f"Error logs: {e}")

def game_client_test():
    BASE_URL = "https://api-web.nhle.com/v1/gamecenter"
    MODEL_SERVICE_URL = "http://127.0.0.1:5000"

    client = GameClient(BASE_URL, MODEL_SERVICE_URL)
    game_id = "2022030411"

    results = client.process_game(game_id)
    print(results)
    print("Process finished")

if __name__ == "__main__":
    # serving_test()
    game_client_test()
