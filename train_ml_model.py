"""ML model training script with validation and security checks."""

import logging
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Configure logging for better tracking (safer than plain print statements)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration constants
MODEL_PATH = Path(__file__).parent / 'model.joblib'
MIN_PERFORMANCE_THRESHOLD = 0.80
RANDOM_STATE = 42


def train_model() -> bool:
    """Train and validate Logistic Regression model on Iris dataset.
    
    Implements security and performance best practices:
    - Stratified train/test split for balanced validation
    - 5-fold cross-validation to prevent overfitting
    - Performance threshold checks before saving
    - Secure model serialization with joblib protocol 2
    - Comprehensive logging and error handling
    
    Returns:
        bool: True if training successful and model saved, False otherwise
    """
    
    try:
        # 1. Load dataset
        logger.info("Loading iris dataset...")
        iris = load_iris()
        X, y = iris.data, iris.target
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

        # 2. Split dataset into training and testing sets (80% train, 20% test)
        # Using stratified split to ensure balanced class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=RANDOM_STATE, 
            stratify=y
        )
        logger.info(f"Train/test split: {len(X_train)} train, {len(X_test)} test samples")

        # 3. Train model
        # max_iter=200 ensures the model converges properly
        logger.info("Training Logistic Regression model...")
        model = LogisticRegression(
            max_iter=200,
            random_state=RANDOM_STATE,
            solver='lbfgs',
            multi_class='multinomial'
        )
        model.fit(X_train, y_train)
        
        # 4. Perform 5-fold cross-validation to prevent overfitting
        logger.info("Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        std_cv_score = cv_scores.std()
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
        
        # 5. Check accuracy on test data to ensure good performance
        test_score = model.score(X_test, y_test)
        logger.info(f"Test set accuracy: {test_score:.4f}")
        
        # 6. Generate detailed classification report
        y_pred = model.predict(X_test)
        logger.info("\\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=iris.target_names))
        
        # 7. Check performance threshold (security: prevent deploying poor models)
        if mean_cv_score < MIN_PERFORMANCE_THRESHOLD:
            logger.warning(
                f"Model performance ({mean_cv_score:.4f}) below threshold "
                f"({MIN_PERFORMANCE_THRESHOLD}). Model not saved."
            )
            return False
        
        # 8. Save model securely using pathlib and joblib protocol 2
        # Protocol 2 ensures compatibility with older Python versions
        # Compression level 3 balances file size and speed
        logger.info(f"Saving model to {MODEL_PATH}...")
        joblib.dump(model, MODEL_PATH, protocol=2, compress=3)
        
        # 9. Verify model was saved correctly
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size
            logger.info(f"Model saved successfully ({file_size} bytes)")
            logger.info("Training completed successfully!")
            return True
        else:
            logger.error("Model file was not created")
            return False
            
    except Exception as e:
        # Catch any unexpected errors (like write permission issues)
        logger.error(f"Error during model training/saving: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = train_model()
    exit(0 if success else 1)
