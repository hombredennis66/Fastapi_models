from functools import cached_property, lru_cache
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def _get_cached_prediction(model, features_tuple):
    """
    Internal cached prediction function.
    Taking model as an argument and being top-level avoids the lru_cache
    memory leak associated with instance methods.
    """
    # Scikit-learn can accept a list containing a tuple.
    # This avoids converting the tuple back to a list in the hot path.
    prediction = model.predict([features_tuple])
    return int(prediction[0])

class MLService:
    @cached_property
    def model(self):
        """Lazy load the Scikit-learn model."""
        import joblib
        try:
            model = joblib.load('model.joblib')
            return model
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            return None

    def predict(self, features_list):
        """Perform prediction with caching and low overhead."""
        if self.model is None:
            raise RuntimeError("ML model could not be loaded")

        # Convert list to tuple for hashability in lru_cache
        features_tuple = tuple(features_list)
        return _get_cached_prediction(self.model, features_tuple)
