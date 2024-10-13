import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample


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
        self.model_path = model_path
        self.model = None

    def train(self):
        data = pd.read_csv(self.dataset_path)

        data['bmi'] = data['bmi'].fillna(data['bmi'].mean())
        data.drop(columns='id', inplace=True)

        categorical_cols = data.select_dtypes(include=['object']).columns

        for column, mapping in self.types.items():
            data[column] = data[column].map(mapping)

        X = data.drop(columns='stroke')
        y = data['stroke']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


        train_data = pd.concat([X_train, y_train], axis=1)

        majority = train_data[train_data.stroke == 0]
        minority = train_data[train_data.stroke == 1]

        minority_upsampled = resample(minority,
                                    replace=True, 
                                    n_samples=len(majority), 
                                    random_state=42)

        upsampled_data = pd.concat([majority, minority_upsampled])

        X_data = train_data.drop(columns='stroke')
        y_data = train_data['stroke']

        model = RandomForestClassifier(random_state=42)
        model.fit(X_data, y_data)

        self.model = model
        joblib.dump(model, self.model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        for column, mapping in self.types.items():
            input_df[column] = input_df[column].map(mapping)
        prediction = self.model.predict(input_df)
        return 'Stroke' if prediction[0] == 1 else 'No Stroke'