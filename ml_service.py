from functools import cached_property
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
    def _predict_internal(self):
        """Internal method to provide instance-level caching for predictions."""
        from functools import lru_cache

        @lru_cache(maxsize=128)
        def _cached_predict(features_tuple):
            # Scikit-learn models can often take a list of lists directly,
            # avoiding the overhead of numpy array creation for a single sample.
            prediction = self.model.predict([features_tuple])
            return int(prediction[0])

        return _cached_predict

    def predict(self, features_list):
        """Perform prediction using the lazy-loaded model with caching."""
        if self.model is None:
            raise RuntimeError("ML model could not be loaded")

        # Use a tuple to make features hashable for lru_cache
        features_tuple = tuple(features_list)
        return self._predict_internal(features_tuple)
