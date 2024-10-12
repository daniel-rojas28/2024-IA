import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Cargar datos de entrenamiento
dtype_dict = {
    'StateHoliday': str,
}
train_file_path = '../datasets/rossman.csv'  # Cambia esto a la ruta de tu archivo de entrenamiento
train_data = pd.read_csv(train_file_path, dtype=dtype_dict)

# Manejar solo los datos de la tienda 1
train_data = train_data[train_data['Store'] == 2]

train_data["Date"]=pd.to_datetime(train_data["Date"])
train_data["Sales"]=pd.to_numeric(train_data["Sales"], downcast='float')

print(train_data.shape)
print(train_data.head(5))

train_data = train_data.sort_values(by='Date')
train_data = train_data.groupby('Date')['Sales'].sum().reset_index()
train_data = train_data.set_index('Date')
print(train_data.head(5))

y = train_data['Sales'].resample('MS').mean()
y.plot(figsize=(15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit(disp=False)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit(disp=False)
print(results.summary().tables[1])


# Crear un grafico con la predicción
pred = results.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2013':].plot(label='Observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

# Dame un ejemplo de como puedo obtener la predicción para una fecha en específico
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='Observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

# Si doy una fecha en específico, ¿cómo puedo obtener la predicción?
forecasted_date = '2016-01-01'
forecast = results.get_forecast(steps=100)
forecast_ci = forecast.conf_int()
forecasted_sales = forecast.predicted_mean[forecasted_date]
print(f'La predicción de ventas para la fecha {forecasted_date} es: {forecasted_sales}')

# Guardar el modelo
import joblib

joblib.dump(results, '../models/rossman_model.pkl')
print("Modelo guardado correctamente en ../models/rossman_model.pkl")
