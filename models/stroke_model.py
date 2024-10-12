import joblib
import pandas as pd


class StrokeModel:
    types = {
        "gender": {
            "Male": 1,
            "Female": 0,
            "Other": 2
        },
        "ever_married": {
            "Yes": 1,
            "No": 0
        },
        "work_type": {
            "Private": 0,
            "Self-employed": 1,
            "Govt_job": 2,
            "Children": 3,
            "Never_worked": 4
        },
        "Residence_type": {
            "Urban": 1,
            "Rural": 0
        },
        "smoking_status": {
            "formerly smoked": 1,
            "never smoked": 0,
            "smokes": 2,
            "Unknown": 3
        }
    }

    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        # Mapear las variables categóricas
        for column, mapping in self.types.items():
            input_df[column] = input_df[column].map(mapping)
        # Realizar la predicción
        prediction = self.model.predict(input_df)
        # Retornar el resultado en forma legible
        return 'Stroke' if prediction[0] == 1 else 'No Stroke'