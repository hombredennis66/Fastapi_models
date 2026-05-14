from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Save model
    joblib.dump(model, 'model.joblib')
    print("Model saved to model.joblib")

if __name__ == "__main__":
    train_model()
