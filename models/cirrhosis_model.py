import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


class CirrhosisModel:
    
    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None
        self.types = None

    def train(self):
        df = pd.read_csv(self.dataset_path)
        df = df.drop(columns=['ID', 'Status', 'N_Days'])
        df['Age'] = df['Age'] / 365

        label_encoders = {}
        categorical = df.select_dtypes(include=['object']).columns
        for col in categorical:
            le = LabelEncoder()
            label = le.fit_transform(df[col].astype(str))
            df[col] = label
            label_encoders[col] = le

        mappings = {}
        for col, le in label_encoders.items():
            mappings[col] = {index: label for index, label in enumerate(le.classes_)}
        self.types = mappings
        imputer = SimpleImputer(strategy='median')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        X = df.drop(columns=['Stage'])
        y = df['Stage']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)
        self.model = rf_model
        joblib.dump(rf_model, self.model_path)


    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        for column, mapping in self.types.items():
            input_df[column] = input_df[column].map(mapping)
        prediction = self.model.predict(input_df)
        return f'El paciente esta en la etapa {int(prediction[0])}'
