from functools import cached_property

class MLService:
    @cached_property
    def model(self):
        # Lazy load joblib and the model to improve startup time
        import joblib
        try:
            return joblib.load('model.joblib')
        except Exception as e:
            print(f"Error loading ML model: {e}")
            return None

    def predict(self, features):
        # Lazy load numpy for prediction
        import numpy as np

        model = self.model
        if model is None:
            raise RuntimeError("ML model not loaded")

        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        return int(prediction[0])
