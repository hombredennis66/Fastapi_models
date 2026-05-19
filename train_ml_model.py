"""ML model training script with validation and security checks."""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = Path(__file__).parent / 'model.joblib'
MIN_PERFORMANCE_THRESHOLD = 0.80
RANDOM_STATE = 42

def train_model():
    """Train and validate iris classification model.
    
    Returns:
        bool: True if training successful and model saved, False otherwise
    """
    try:
        # Load dataset
        logger.info("Loading iris dataset...")
        iris = load_iris()
        X, y = iris.data, iris.target
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        # Initialize and train model
        logger.info("Training Logistic Regression model...")
        model = LogisticRegression(
            max_iter=200,
            random_state=RANDOM_STATE,
            solver='lbfgs',
            multi_class='multinomial'
        )
        model.fit(X_train, y_train)
        
        # Cross-validation to prevent overfitting
        logger.info("Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        std_cv_score = cv_scores.std()
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
        
        # Test set evaluation
        test_score = model.score(X_test, y_test)
        logger.info(f"Test set accuracy: {test_score:.4f}")
        
        # Generate classification report
        y_pred = model.predict(X_test)
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=iris.target_names))
        
        # Check performance threshold
        if mean_cv_score < MIN_PERFORMANCE_THRESHOLD:
            logger.warning(
                f"Model performance ({mean_cv_score:.4f}) below threshold "
                f"({MIN_PERFORMANCE_THRESHOLD}). Not saving model."
            )
            return False
        
        # Save model with protocol 2 for compatibility
        logger.info(f"Saving model to {MODEL_PATH}...")
        joblib.dump(model, MODEL_PATH, protocol=2, compress=3)
        
        # Verify model was saved
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size
            logger.info(f"Model saved successfully ({file_size} bytes)")
            logger.info("Training completed successfully!")
            return True
        else:
            logger.error("Model file was not created")
            return False
            
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = train_model()
    exit(0 if success else 1)
