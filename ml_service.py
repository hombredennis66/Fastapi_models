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
        """Internal cached prediction function to avoid self-reference in lru_cache."""
        @lru_cache(maxsize=128)
        def _predict(features_tuple):
            if self.model is None:
                raise RuntimeError("ML model could not be loaded")
            # Optimization: pass [features_tuple] directly to avoid numpy overhead
            prediction = self.model.predict([features_tuple])
            return int(prediction[0])
        return _predict

    def predict(self, features_list):
        """Perform prediction using the lazy-loaded model and cache results."""
        return self._cached_predict(tuple(features_list))
