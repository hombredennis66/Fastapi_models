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
        """Internal cached prediction function to avoid instance-level lru_cache issues."""
        @lru_cache(maxsize=128)
        def predict_func(features_tuple):
            # scikit-learn models can accept a list of tuples/lists directly,
            # bypassing the need for numpy array creation for single samples.
            prediction = self.model.predict([features_tuple])
            return int(prediction[0])
        return predict_func

    def predict(self, features_list):
        """Perform prediction using the lazy-loaded model with caching."""
        if self.model is None:
            raise RuntimeError("ML model could not be loaded")

        # Convert to tuple for hashability in lru_cache
        features_tuple = tuple(features_list)
        return self._cached_predict(features_tuple)
