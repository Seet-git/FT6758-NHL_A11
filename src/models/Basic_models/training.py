
from sklearn.model_selection import train_test_split

def train_model(model, X, y, test_size=0.3, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_test, y_test
