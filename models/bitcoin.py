import joblib
import itertools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import statsmodels.api as sm
from models.model import Model

warnings.filterwarnings("ignore")

class BitcoinModel(Model):
    def train(self):
        data = pd.read_csv(self.dataset_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data['Volume'] = data['Volume'].replace('-', np.nan).str.replace(',', '').astype(float)
        data['Market Cap'] = data['Market Cap'].replace('-', np.nan).str.replace(',', '').astype(float)
        data = data.sort_values(by='Date')
        data = data.set_index('Date')
        data['Volume'] = data['Volume'].fillna(data['Volume'].median())
        data['Market Cap'] = data['Market Cap'].fillna(data['Market Cap'].median())
        y = data['Close']
        mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        model = mod.fit(disp=False)
        self.model = model
        self.save_model()

    def predict(self, forecasted_date):
        forecast = self.model.get_forecast(steps=100)
        forecast.conf_int()
        forecasted_sales = forecast.predicted_mean[forecasted_date]
        return f'La prediccion del precio de Bitcoin para la fecha {forecasted_date} es: {forecasted_sales}'