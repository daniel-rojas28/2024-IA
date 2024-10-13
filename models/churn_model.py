import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class ChurnModel:
    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None
    
    def train(self):
        data = pd.read_csv(self.dataset_path)

        data = data.drop('customerID', axis=1)
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data.fillna(data.median(numeric_only=True), inplace=True)

        label_encoders = {}
        for column in data.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])
        X = data.drop('Churn', axis=1)
        y = data['Churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
        log_reg.fit(X_train, y_train)

        self.model = log_reg
        joblib.dump(self.model, self.model_path)


    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        print(input_df.shape)
        prediction = self.model.predict(input_df)
        # Retornar el resultado en forma legible
        return 'Churn' if prediction[0] == 1 else 'No Churn'