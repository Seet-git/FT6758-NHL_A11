from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier

def get_knn():
    return KNeighborsClassifier(n_neighbors=5, algorithm='auto')

def get_random_forest():
    return RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1)

def get_mlp_1_hidden():
    return MLPClassifier(random_state=1, max_iter=500)

def get_mlp_2_hidden():
    return MLPClassifier(random_state=1, max_iter=500)

def get_perceptron():
    return Perceptron(random_state=1, max_iter=500)
