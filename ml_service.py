from functools import cached_property, lru_cache
import logging

logger = logging.getLogger(__name__)

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

    @cached_property
    def _cached_predict(self):
        """Per-instance cache for predictions to avoid memory leaks and handle 'self'."""
        @lru_cache(maxsize=128)
        def _predict(features_tuple):
            # Scikit-learn models can accept a list of sequences (like a list of tuples).
            # This avoids numpy array creation and reshaping overhead in the hot path.
            prediction = self.model.predict([features_tuple])
            return int(prediction[0])
        return _predict

    def predict(self, features_list):
        """Perform prediction using the lazy-loaded model and an LRU cache."""
        if self.model is None:
            raise RuntimeError("ML model could not be loaded")

        # Convert list to tuple to ensure it is hashable for the lru_cache.
        features_tuple = tuple(features_list)
        return self._cached_predict(features_tuple)
