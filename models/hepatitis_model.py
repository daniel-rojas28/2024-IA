import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

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
        self.model_path = model_path
        self.model = None

    def train(self):
        df = pd.read_csv(self.dataset_path)
        df = df.drop(columns=['Unnamed: 0'])
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns  
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'm' else 0)

        mappings ={
            "0=Blood Donor": 0,
            "1=Hepatitis": 1,
            "2=Fibrosis": 2,
            "3=Cirrhosis": 3,
            "0s=suspect Blood Donor": 4,
        }
        df['Category'] = df['Category'].map(mappings)

        X = df.drop(columns=['Category'])
        y = df['Category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)

        self.model = knn
        joblib.dump(knn, self.model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        input_df['Sex'] = input_df['Sex'].apply(lambda x: 0 if x == 'm' else 1)
        prediction = self.model.predict(input_df)

        # Retornar el resultado en forma legible
        return f'El paciente se categoriza como {self.category_map[prediction[0]]}'