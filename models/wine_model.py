import joblib
import pandas as pd


class WineModel:
    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)
        # Retornar el resultado en forma legible
        return f'El vino es de la clase "{prediction[0]}" en una escala de 3 al 8'