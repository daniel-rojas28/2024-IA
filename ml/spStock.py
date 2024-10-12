import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Cargar el dataset
df = pd.read_csv('../datasets/spStocks.csv')
df = df.drop(columns=['Name'])

# Usar los primeros 10000 registros para el análisis
# df = df.iloc[:10000]
# Preprocesamiento
df['date'] = pd.to_datetime(df['date'])

print(df.info())
# Agrupar filas con la misma fecha
df = df.groupby('date').mean().reset_index()

# Cambiar los valores nulos con la medianas en todas las columnas
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].fillna(df[col].median())


# Ordenar el dataset por fecha
df = df.sort_values(by='date')
df = df.set_index('date')

# Crear un rango de fechas completo
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

# Reindexar el dataframe
df = df.reindex(full_range)

# Rellenar o interpolar los valores faltantes
df = df.interpolate(method='linear')

# Revisar si hay valores nulos
print(df.isnull().sum())
print(df.head(5))
print(df.tail(5))

# Definir la serie temporal objetivo (Close)
y = df['close']
print(y)
import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

import statsmodels.api as sm

# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(y,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#             results = mod.fit(disp=False)
#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue
#
# Seleccionar el mejor modelo SARIMAX
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit(disp=False)
print(results.summary().tables[1])


# Predicción en una fecha específica
fecha_especifica = pd.to_datetime('2020-08-01')  # Cambia a la fecha que desees

# Verificar si la fecha está dentro del rango de predicción
if fecha_especifica > y.index[-1]:
    # Si la fecha es futura, generamos una predicción
    steps_a_predecir = (fecha_especifica - y.index[-1]).days
    forecast = results.get_forecast(steps=steps_a_predecir)
    print(forecast.predicted_mean.index)

    prediccion = forecast.predicted_mean.iloc[-1]
    print(f'Predicción para {fecha_especifica}: {prediccion}')
else:
    print(f'La fecha {fecha_especifica} está dentro de los datos históricos.')

# Crear un gráfico con la predicción
pred = results.get_prediction(start=pd.to_datetime('2017-06-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2013':].plot(label='Observado')
pred.predicted_mean.plot(ax=ax, label='Predicción', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Fecha')
ax.set_ylabel('Precio de S&P 500')
plt.legend()
plt.show()

# Guardar el modelo
import joblib
joblib.dump(results, '../models/spStock.pkl')

