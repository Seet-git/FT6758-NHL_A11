import os

import numpy as np
import matplotlib.pyplot as plt
import config
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score, precision_score, \
    recall_score, accuracy_score
import seaborn as sns

from torch.utils.data import WeightedRandomSampler

from src.models.Neural_network.models import *

from sklearn.calibration import calibration_curve


def set_seed(seed: int):
    """
    Ensures that the experiment is reproducible
    :param seed: seed number
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_balanced_sampler(y_train):
    # Count class
    class_counts = [(y_train == 0).sum().item(), y_train.sum().item()]

    weights = 1 / torch.tensor(class_counts, dtype=torch.float, device=y_train.device)

    y_train = y_train.to(weights.device)

    sample_weights = weights[y_train.long()]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def get_model(input_size, hp):
    if input_size <= 0:
        raise ValueError(f"Invalid input_size: {input_size}. It must be positive.")
    if config.ALGORITHM == "MLP_H2":
        return MLP_H2(input_size, hp.hidden_layer1, hp.hidden_layer2, hp.dropout_rate)
    elif config.ALGORITHM == "MLP_H1":
        return MLP_H1(input_size, hp.hidden_layer, hp.dropout_rate)
    elif config.ALGORITHM == "Perceptron":
        return Perceptron(input_size)
    else:
        raise ValueError("Bad ALGORITHM value")


def plot_roc_curve(y_true_list, y_scores_list, name=""):
    plt.figure(figsize=(10, 6))
    for fold, (y_true, y_scores) in enumerate(zip(y_true_list, y_scores_list)):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Fold {fold + 1} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux faux positifs')
    plt.ylabel('Taux vrai positif')
    plt.title('ROC Curve - Tous les folds')
    plt.legend(loc="lower right")
    plt.savefig(f"./images/{config.ALGORITHM}/{name}_roc_curve.svg", format="svg")
    plt.show()


def plot_precision_recall_curve(y_true_list, y_scores_list, name=""):
    plt.figure(figsize=(10, 6))
    for fold, (y_true, y_scores) in enumerate(zip(y_true_list, y_scores_list)):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, lw=2, label=f'Fold {fold + 1}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Tous les folds')
    plt.legend(loc="lower left")
    plt.savefig(f"./images/{config.ALGORITHM}/{name}_precision_recall_curve.svg",
                format="svg")
    plt.show()


def plot_confusion_matrix(y_true_list, y_pred_list, name=""):
    cm = confusion_matrix(y_true_list[-1], y_pred_list[-1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Dernier Fold)')
    plt.savefig(f"./images/{config.ALGORITHM}/{name}_confusion_matrix.svg", format="svg")
    plt.show()


def performance_metrics_bar(y_true_list, y_pred_list, name=""):
    f1 = f1_score(y_true_list[-1], y_pred_list[-1], average="macro")
    precision = precision_score(y_true_list[-1], y_pred_list[-1], average="macro", zero_division=0)
    recall = recall_score(y_true_list[-1], y_pred_list[-1], average="macro", zero_division=0)
    accuracy = accuracy_score(y_true_list[-1], y_pred_list[-1])

    metrics = ['Precision', 'Recall', 'F1-score', 'Accuracy']
    values = [precision, recall, f1, accuracy]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Metriques')
    plt.ylabel('Scores')
    plt.title('Métriques de performances (Dernier Fold)')
    plt.ylim(0, 1)
    plt.savefig(f"./images/{config.ALGORITHM}/{name}_performance_metrics.svg",
                format="svg")
    plt.show()


def goal_rate_vs_probability_percentile(y_true_list, y_scores_list, name=""):
    plt.figure(figsize=(10, 6))
    markers = ['o', 'x', 's', 'D', '^']
    for fold, (y_true, y_scores) in enumerate(zip(y_true_list, y_scores_list)):
        percentiles = np.percentile(y_scores, np.linspace(0, 100, 11))  # Divise en 10 centiles
        goal_rates = []
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        for i in range(len(percentiles) - 1):
            mask = (y_scores >= percentiles[i]) & (
                    y_scores < percentiles[i + 1])  # Sélection des valeurs dans chaque intervalle
            if mask.any():
                goal_rate = np.mean(y_true[mask])  # Moyenne des valeurs de y_true dans cet intervalle
                goal_rates.append(goal_rate)
            else:
                goal_rates.append(0)
        marker = markers[fold % len(markers)]
        plt.plot(range(10), goal_rates, marker=marker, label=f'Fold {fold + 1}', lw=2)

    plt.legend(loc="lower right")
    plt.xlabel('Centiles')
    plt.ylabel('Taux de buts')
    plt.title('Taux de buts par centile de probabilité')
    plt.show()


def cumulative_goal_rate(y_true_list, y_scores_list, name=""):
    plt.figure(figsize=(10, 6))
    for fold, (y_true, y_scores) in enumerate(zip(y_true_list, y_scores_list)):
        sorted_indices = np.argsort(y_scores)[::-1]  # Trie les scores de probabilité par ordre décroissant
        y_sorted = np.array(y_true)[sorted_indices]  # Trie y_true selon les scores triés
        cumulative_goals = np.cumsum(y_sorted) / np.sum(y_sorted)  # Calcule la proportion cumulée
        plt.plot(np.linspace(0, 1, len(cumulative_goals)), cumulative_goals, label=f'Fold {fold + 1}', lw=2)

    plt.legend(loc="lower right")
    plt.xlabel('Proportion des prédictions')
    plt.ylabel('Proportion cumulée de buts')
    plt.title('Proportion cumulée de buts par centile')
    plt.show()


def reliability_curve(y_true_list, y_scores_list, name=""):
    plt.figure(figsize=(10, 6))
    markers = ['o', 'x', 's', 'D', '^']  # Marqueurs différents
    for fold, (y_true, y_scores) in enumerate(zip(y_true_list, y_scores_list)):
        prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy='uniform')
        marker = markers[fold % len(markers)]
        plt.plot(prob_pred, prob_true, marker=marker, label=f'Fold {fold + 1}', lw=2)

    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Probabilité observée')
    plt.title('Courbe de fiabilité')
    plt.show()


def plot_all_visualizations(y_true_list, y_scores_list, y_pred_list, name=""):
    if not os.path.exists(f"./images/{config.ALGORITHM}"):
        os.makedirs(f"./images/{config.ALGORITHM}")

    plot_roc_curve(y_true_list, y_scores_list, name=name)
    plot_precision_recall_curve(y_true_list, y_scores_list, name=name)
    plot_confusion_matrix(y_true_list, y_pred_list, name=name)
    performance_metrics_bar(y_true_list, y_pred_list, name=name)
    goal_rate_vs_probability_percentile(y_true_list, y_scores_list, name=name)
    cumulative_goal_rate(y_true_list, y_scores_list, name=name)
    reliability_curve(y_true_list, y_scores_list, name=name)
