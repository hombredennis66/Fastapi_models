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
        """Internal cached prediction function to avoid unhashable 'self'."""
        @lru_cache(maxsize=128)
        def _predict(features_tuple):
            # Pass a list containing the features tuple directly to the model.
            # This avoids the overhead of numpy array creation and reshaping.
            prediction = self.model.predict([features_tuple])
            return int(prediction[0])
        return _predict

    def predict(self, features_list):
        """Perform prediction using the lazy-loaded model with caching."""
        if self.model is None:
            raise RuntimeError("ML model could not be loaded")

        # Convert list to tuple to make it hashable for LRU cache
        features_tuple = tuple(features_list)
        return self._cached_predict(features_tuple)
