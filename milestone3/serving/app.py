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

    print("logs route")

    # TODO: read the log file specified and return the data
    raise NotImplementedError("TODO: implement this endpoint")

    response = None
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    global model
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required), # LogisticRegression_Distance, LogisticRegression_Distance_Angle
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

    print("predict route")

    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO:
    raise NotImplementedError("TODO: implement this enpdoint")
    
    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!
