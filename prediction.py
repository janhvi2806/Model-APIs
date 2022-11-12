import pandas as pd
import numpy as np
from tensorflow import keras


class Prediction:

    def __init__(self):
        model_path = "DiaCare.h5"
        self.model: keras.engine.sequential.Sequential = keras.models.load_model(model_path)

    def predict(self, user_data: dict):
        input_df: pd.DataFrame = pd.DataFrame([user_data])
        result: np.ndarray = self.model.predict(input_df)
        return result[0][0]
