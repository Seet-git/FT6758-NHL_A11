import config
from src.models.Neural_network.bayesian_optimization import bayesian_optimization
from src.models.Neural_network.predict import predict


def optimize(model_name: str):
    if model_name not in ["Perceptron", "MLP_H1", "MLP_H2"]:
        raise ValueError("Bad ALGORITHM value")
    config.PREDICTION_FILENAME = f"{model_name}_pred"
    config.ALGORITHM = model_name
    print(f"device: {config.DEVICE}")

    bayesian_optimization(n_trials=config.N_TRIALS)


def predict_data(model_name: str):
    if model_name not in ["Perceptron", "MLP_H1", "MLP_H2"]:
        raise ValueError("Bad ALGORITHM value")
    config.OUTPUT_HP_FILENAME = f"hp_{model_name}"
    config.INPUT_HP_FILENAME = f"hp_{model_name}"
    config.ALGORITHM = model_name
    predict()
