import joblib
import itertools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import statsmodels.api as sm

class SPStockModel:
    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None

    def train(self):
        df = pd.read_csv(self.dataset_path)
        df = df.drop(columns=['Name'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.groupby('date').mean().reset_index()

        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].fillna(df[col].median())


        df = df.sort_values(by='date')
        df = df.set_index('date')

        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

        df = df.reindex(full_range)

        df = df.interpolate(method='linear')
        y = df['close']

        mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(0, 0, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        results = mod.fit(disp=False)

        self.model = results
        joblib.dump(self.model, self.model_path)


    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, forecasted_date):
        forecast = self.model.get_forecast(steps=100)
        forecast.conf_int()
        forecasted_sales = forecast.predicted_mean[forecasted_date]
        return f'La prediccion del precio de SP Stock para la fecha {forecasted_date} es: {forecasted_sales}'