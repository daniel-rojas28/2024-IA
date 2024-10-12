import joblib
import pandas as pd


class CirrhosisModel:
    types = {
        'Drug': {0: 'D-penicillamine', 1: 'Placebo', 2: 'nan'},
        'Sex': {0: 'F', 1: 'M'},
        'Ascites': {0: 'N', 1: 'Y', 2: 'nan'},
        'Hepatomegaly': {0: 'N', 1: 'Y', 2: 'nan'},
        'Spiders': {0: 'N', 1: 'Y', 2: 'nan'},
        'Edema': {0: 'N', 1: 'S', 2: 'Y'}
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
        return f'El paciente esta en la etapa {int(prediction[0])}'
