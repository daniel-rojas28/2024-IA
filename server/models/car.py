import pandas as pd
from sklearn.linear_model import LinearRegression
from models.model import Model

class CarModel(Model):

    def train(self):
        data = pd.read_csv(self.dataset_path)

        # Transform the categorical variables to numerical
        label_encodings = {
            'Fuel_Type': {'CNG': 0, 'Diesel': 1, 'Petrol': 2},
            'Seller_Type': {'Dealer': 0, 'Individual': 1},
            'Transmission': {'Automatic': 0, 'Manual': 1}
        }

        for column, mapping in label_encodings.items():
            data[column] = data[column].map(mapping)

        X = data.drop(columns=['Car_Name', 'Selling_Price'])
        y = data['Selling_Price']

        model = LinearRegression()
        model.fit(X, y)
        self.model = model
        self.save_model()

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)
        return f'La prediccion del precio de venta del carro es: {prediction[0]}'
