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
        """Perform prediction using the lazy-loaded model."""
        if self.model is None:
            raise RuntimeError("ML model could not be loaded")

        import numpy as np
        features = np.array(features_list).reshape(1, -1)
        prediction = self.model.predict(features)
        return int(prediction[0])
