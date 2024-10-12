import joblib
import pandas as pd

class HepatitisModel:
    category_map = {
        0: "0=Blood Donor",
        1: "1=Hepatitis",
        2: "2=Fibrosis",
        3: "3=Cirrhosis",
        4: "0s=suspect Blood Donor"
    }
    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        input_df['Sex'] = input_df['Sex'].apply(lambda x: 0 if x == 'm' else 1)
        prediction = self.model.predict(input_df)

        # Retornar el resultado en forma legible
        return f'El paciente se categoriza como {self.category_map[prediction[0]]}'