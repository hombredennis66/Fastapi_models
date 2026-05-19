import logging
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Set up logging for better tracking (safer than plain print statements)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model() -> None:
    """Trains a Logistic Regression model on the Iris dataset and saves it."""
    
    try:
        # 1. Load dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # 2. Split dataset into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Train model
        # max_iter=200 ensures the model converges properly
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        
        # Optional: Check the accuracy on test data to ensure good performance
        accuracy = model.score(X_test, y_test)
        logger.info(f"Model trained successfully with an accuracy of {accuracy * 100:.2f}%")

        # 4. Save model securely using pathlib
        # This ensures the file is saved relative to where this script is located
        output_dir = Path(__file__).parent
        model_path = output_dir / 'model.joblib'
        
        joblib.dump(model, model_path)
        logger.info(f"Model successfully saved to {model_path}")

    except Exception as e:
        # Catch any unexpected errors (like write permission issues)
        logger.error(f"An error occurred during model training/saving: {e}")

if __name__ == "__main__":
    train_model()
