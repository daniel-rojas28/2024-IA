import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from models.model import Model

class WineModel(Model):
 
    def train(self):
        wine_data = pd.read_csv(self.dataset_path)

        X = wine_data.drop(columns=['quality'])
        y = wine_data['quality']

        X = pd.get_dummies(X, columns=['type'], drop_first=True)

        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42, class_weight='balanced' )
        model.fit(X_train, y_train)

        self.model = model
        self.save_model()


    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)
        # Retornar el resultado en forma legible
        return f'El vino es de la clase "{prediction[0]}" en una escala de 3 al 8'
