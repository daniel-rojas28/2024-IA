import joblib
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BodyFatModel:
    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None

    def train(self):
        data = pd.read_csv(self.dataset_path)

        X = data.drop(columns=['BodyFat'])
        y = data['BodyFat']
        X = sm.add_constant(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = sm.OLS(y_train, X_train).fit()
        joblib.dump(self.model, self.model_path)


    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        input_df = sm.add_constant(input_df, has_constant='add')
        prediction = self.model.predict(input_df)
        return prediction[0]
