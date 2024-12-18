"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import wandb
from dotenv import load_dotenv
import pickle
import numpy as np

# import ift6758

load_dotenv()

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

appHasRunBefore:bool = False

model = None

@app.before_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    global appHasRunBefore, model

    if not appHasRunBefore:
        # TODO: setup basic logging configuration
        # en configurant le loggging ici, ca ne marchait. on le faire avant le declaration de l'instance app
        # TODO: any other initialization before the first request (e.g. load default model)

        model_dir = "./models/LogisticRegression_Distance.pkl"
        with open(model_dir, "rb") as file:
            model = pickle.load(file)

        appHasRunBefore = True

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # TODO: read the log file specified and return the data
    try:
        # Lire le fichier log
        with open("flask.log", "r") as log_file:
            logs_content = log_file.readlines()

        # Préparer la réponse
        response = {"logs": logs_content}

        return jsonify(response)

    except Exception as e:
        # Gérer les erreurs de lecture du fichier
        return jsonify({"error": str(e)}), 500

@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    global model
    """
    inputs:
    - project_name: should be IFT6758.2024-A11
    - entity_name: should be youry-macius-universite-de-montreal
    - model_name: LogisticRegression_Distance ou LogisticRegression_Distance_Angle
    """

    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    project_name = json["project_name"]
    entity_name = json["entity_name"]
    model_name = json["model_name"]
    output_path = "./models"

    # TODO: check to see if the model you are querying for is already downloaded

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model

    os.makedirs(output_path, exist_ok=True)

    # Define the local model path
    model_dir = os.path.join(output_path, f"{model_name}.pkl")

    # Check if the model is already downloaded
    if os.path.exists(model_dir):
        app.logger.info(f"Model already exists at {model_dir}. Loading the model...")

        with open(model_dir, "rb") as file:
            model = pickle.load(file)

        response = {
            "status": "success",
            "message": f"Model {model_name} loaded from {model_dir}",
            "model_path": model_dir
        }
        return jsonify(response), 200

    try:
        app.logger.info(f"Model not found locally. Attempting to download {model_name} from WandB...")

        wandb.login(key=WANDB_API_KEY)
        # run = wandb.init(project=project_name, entity=entity_name)

        api = wandb.Api()
        artifact = api.artifact(f"{entity_name}/{project_name}/{model_name}:latest")

        artifact_dir = artifact.download(root=output_path)

        with open(f"{artifact_dir}/{model_name}.pkl", "rb") as file:
            model = pickle.load(file)

        app.logger.info(f"Model {model_name} successfully downloaded to {artifact_dir}")
        response = {
            "status": "success",
            "message": f"Model {model_name} downloaded successfully",
            "model_path": artifact_dir
        }
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Failed to download model {model_name} from WandB: {str(e)}")
        response = {
            "status": "error",
            "message": f"Failed to download model {model_name} : {str(e)}"
        }
        return jsonify(response), 500


@app.route("/predict", methods=["POST"])
def predict():
    global model

    """
    inputs: 
        - distance
        - angle (si on utlise le modele distance et angle)
    
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    try:
        distances = json["distance"]
        x_input = np.array(distances).reshape(-1, 1)  # Si angle n'est pas fourni, on utilise uniquement distance

        if "angle" in json:
            angles = json["angle"]
            x_input = np.column_stack((distances, angles))

        goal_probs = model.predict_proba(x_input)[:, 1]
        is_goal = model.predict(x_input)

        response = {
            "goal_probs": goal_probs.tolist(),
            "is_goal": is_goal.tolist()
        }

        app.logger.info(response)
        return jsonify(response), 200 # response must be json serializable!

    except Exception as e:
        app.logger.error(f"Failed to do the prediction with the received input.")

        response = {
            "status": "error",
            "message": f"Failed to do the prediction with the received input."
        }
        return jsonify(response), 500


