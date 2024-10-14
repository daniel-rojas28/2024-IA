import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from models.model import Model

class RossmanModel(Model):

    def train(self):
        dtype_dict = {
            'StateHoliday': str,
        }
        train_data = pd.read_csv(self.dataset_path, dtype=dtype_dict)

        train_data = train_data[train_data['Store'] == 2]

        train_data["Date"]=pd.to_datetime(train_data["Date"])
        train_data["Sales"]=pd.to_numeric(train_data["Sales"], downcast='float')

        train_data = train_data.sort_values(by='Date')
        train_data = train_data.groupby('Date')['Sales'].sum().reset_index()
        train_data = train_data.set_index('Date')
        y = train_data['Sales'].resample('MS').mean()
        
        mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        results = mod.fit(disp=False)
        self.model = results
        self.save_model()

    def predict(self, forecasted_date):
        forecast = self.model.get_forecast(steps=100)
        forecast.conf_int()
        forecasted_sales = forecast.predicted_mean[forecasted_date]
        return f'La predicción de ventas para la fecha {forecasted_date} es: {forecasted_sales}'