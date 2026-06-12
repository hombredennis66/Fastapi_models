from functools import cached_property
from typing import List, Optional

class MLService:
    @cached_property
    def model(self):
        """Lazy load the ML model using joblib."""
        import joblib
        try:
            return joblib.load('model.joblib')
        except Exception as e:
            print(f"Error loading ML model: {e}")
            return None

    def predict(self, features: List[float]) -> Optional[int]:
        """Perform prediction using the lazy-loaded model."""
        if self.model is None:
            return None

        import numpy as np
        try:
            features_array = np.array(features).reshape(1, -1)
            prediction = self.model.predict(features_array)
            return int(prediction[0])
        except Exception as e:
            print(f"Prediction error: {e}")
            raise e
