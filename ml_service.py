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

    def predict(self, features_list):
        """Perform prediction using the lazy-loaded model with caching."""
        if self.model is None:
            raise RuntimeError("ML model could not be loaded")

        # Convert list to tuple for caching (tuples are hashable)
        features_tuple = tuple(features_list)
        prediction = self._predict_with_cache(features_tuple)
        return int(prediction)

    @cached_property
    def _predict_with_cache(self):
        """Lazy-loaded LRU cache for predictions."""
        from functools import lru_cache

        @lru_cache(maxsize=128)
        def _get_prediction(features_tuple):
            # The model is accessed from the outer scope (the MLService instance)
            # This avoids including the unhashable model object in the cache key.
            return self.model.predict([features_tuple])[0]

        return _get_prediction
