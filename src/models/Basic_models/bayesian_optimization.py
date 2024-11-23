import os
import pickle
import optuna
import wandb
import numpy as np
from scipy.special import expit
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import config
from src.models.Neural_network.bayesian_optimization import delete_files_with_prefix
from src.models.Neural_network.utils import plot_all_visualizations
from src.models.Basic_models.models import get_mlp_1_hidden, get_mlp_2_hidden, get_perceptron, get_random_forest, get_knn
import pytz
from datetime import datetime
import urllib.parse

# Configuration pour l'heure actuelle
montreal_timezone = pytz.timezone('America/Montreal')
current_time = datetime.now(montreal_timezone).strftime("%m/%d-%H:%M:%S")
global_best_score = -float('inf')


def get_hyperparameters(trial):
    """"""
    hyperparameters_dict = {
        "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 20, 50),
    }

    if config.ALGORITHM == "MLP_H1":
        hyperparameters_dict["hidden_layer"] = trial.suggest_int("hidden_layer", 50, 500, step=50)
    elif config.ALGORITHM == "MLP_H2":
        hyperparameters_dict["hidden_layer1"] = trial.suggest_int("hidden_layer1", 50, 500, step=50)
        hyperparameters_dict["hidden_layer2"] = trial.suggest_int("hidden_layer2", 10, 200, step=10)
    elif config.ALGORITHM == "KNN":
        hyperparameters_dict["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 5)
        hyperparameters_dict["algorithm"] = trial.suggest_categorical("algorithm",
                                                                      ["auto", "ball_tree", "kd_tree", "brute"])
    elif config.ALGORITHM == "RandomForest":
        hyperparameters_dict["n_estimators"] = trial.suggest_int("n_estimators", 1, 5)
        hyperparameters_dict["max_depth"] = trial.suggest_int("max_depth", 1, 5)
    else:
        raise ValueError("Bad ALGORITHM Value")
    return hyperparameters_dict


def initialize_model(hyperparameters_dict):
    """Initialise le modèle en fonction de l'algorithme configuré."""
    if config.ALGORITHM == "MLP_H1":
        model = get_mlp_1_hidden()
        model.set_params(hidden_layer_sizes=(hyperparameters_dict["hidden_layer"],))
    elif config.ALGORITHM == "MLP_H2":
        model = get_mlp_2_hidden()
        model.set_params(hidden_layer_sizes=(
            hyperparameters_dict["hidden_layer1"],
            hyperparameters_dict["hidden_layer2"],
        ))
    elif config.ALGORITHM == "Perceptron":
        model = get_perceptron()
        model.set_params(eta0=hyperparameters_dict["learning_rate"])
    elif config.ALGORITHM == "RandomForest":
        model = get_random_forest()
        model.set_params(
            n_estimators=hyperparameters_dict["n_estimators"],
            max_depth=hyperparameters_dict["max_depth"]
        )
    elif config.ALGORITHM == "KNN":
        model = get_knn()
        model.set_params(
            n_neighbors=hyperparameters_dict["n_neighbors"],
            algorithm=hyperparameters_dict["algorithm"]
        )
    else:
        raise ValueError("Bad ALGORITHM Value")
    return model


def train_and_evaluate_model(model, X_train, X_val, y_train, y_val, fold):
    """Entraîne et évalue le modèle sur un fold."""
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_val)
        y_score = expit(y_score)  # Appliquer la transformation sigmoïde
    else:
        raise ValueError("Le modèle ne supporte ni predict_proba ni decision_function")

    y_pred = (y_score > 0.5).astype(int)
    f1 = f1_score(y_val, y_pred, average="weighted")

    if config.WANDB_ACTIVATE:
        wandb.log({
            "Fold": fold,
            "F1 Score": f1
        })

    return y_score, y_pred, f1


def save_best_model(model, all_y_true, all_y_scores, all_y_pred, mean_f1, trial_number):
    """Sauvegarde le meilleur modèle et génère des visualisations."""
    global global_best_score
    global_best_score = mean_f1
    delete_files_with_prefix()

    # Génération des visualisations
    plot_all_visualizations(all_y_true, all_y_scores, all_y_pred, f"{config.ALGORITHM}_trial_{trial_number}")

    # Sauvegarde du modèle
    model_dir = f"./artifacts/{config.ALGORITHM}/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Enregistrement des artefacts WandB
    if config.WANDB_ACTIVATE:
        artifact = wandb.Artifact(
            name=f"{config.ALGORITHM}_best_model",
            type="model",
            description=f"Best model for {config.ALGORITHM} with F1: {mean_f1:.4f}",
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)


def objective(trial):
    global global_best_score

    X = config.INPUTS_DATA
    y = config.LABELS_DATA

    # HP
    hyperparameters_dict = get_hyperparameters(trial)

    # Initialisation de WandB
    if config.WANDB_ACTIVATE:
        wandb.init(
            project=config.WANDB_PROJECT_NAME,
            name=f"{config.ALGORITHM}_Trial_{trial.number}",
            group=f"{config.ALGORITHM}",
            config=hyperparameters_dict,
            entity=config.WANDB_TEAM_NAME,
        )

    model = None

    # Validation croisée K-Fold
    all_y_true, all_y_scores, all_y_pred, f1_scores = [], [], [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = initialize_model(hyperparameters_dict)
        y_score, y_pred, f1 = train_and_evaluate_model(model, X_train, X_val, y_train, y_val, fold=fold)

        all_y_true.append(y_val)
        all_y_scores.append(y_score)
        all_y_pred.append(y_pred)
        f1_scores.append(f1)

    # Moyenne des scores F1
    mean_f1 = np.mean(f1_scores)

    # Mise à jour du meilleur modèle
    if mean_f1 > global_best_score:
        save_best_model(model, all_y_true, all_y_scores, all_y_pred, mean_f1, trial.number)

    return mean_f1


def bayesian_optimization(model_type, n_trials):
    config.ALGORITHM = model_type
    storage_url = f"mysql+pymysql://{config.USER}:{urllib.parse.quote(config.PASSWORD)}@{config.ENDPOINT}/{config.DATABASE_NAME}"
    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=f"{config.ALGORITHM} Optimizer - {current_time}",
        sampler=optuna.samplers.TPESampler(seed=1)
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"Meilleurs hyper-paramètres pour {model_type}:")
    print(study.best_params)
    print(f"Meilleur score F1: {study.best_value}")
    return study.best_params
