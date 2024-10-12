import joblib
import pandas as pd


class ChurnModel:
    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        print(input_df.shape)
        prediction = self.model.predict(input_df)
        # Retornar el resultado en forma legible
        return 'Churn' if prediction[0] == 1 else 'No Churn'