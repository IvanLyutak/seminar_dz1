import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('data.csv', parse_dates=True)
data['выход'] = data['выход'].str.replace(',', '.', regex=False).astype(float)

data_old = data

data = data.drop(columns=['направление'])

data.columns = ['ds', 'y']
print(data)

# Инициализация и обучение модели Prophet
model = Prophet()
model.fit(data)

specific_dates = [
    "03.01.2022", "04.01.2022", "05.01.2022", "06.01.2022", "07.01.2022", 
    "10.01.2022", "11.01.2022", "12.01.2022", "13.01.2022", "14.01.2022", 
    "17.01.2022", "18.01.2022", "19.01.2022", "20.01.2022", "21.01.2022", 
    "24.01.2022", "25.01.2022", "26.01.2022", "27.01.2022", "28.01.2022", 
    "31.01.2022", "01.02.2022", "02.02.2022", "03.02.2022", "04.02.2022", 
    "07.02.2022", "08.02.2022", "09.02.2022", "10.02.2022", "11.02.2022", 
    "14.02.2022", "15.02.2022", "16.02.2022", "17.02.2022", "18.02.2022", 
    "21.02.2022", "22.02.2022", "23.02.2022", "24.02.2022", "25.02.2022"
]

# Преобразуем список дат в формат datetime
last_dates = model.make_future_dataframe(periods=0)
future_dates = pd.DataFrame({'ds': pd.to_datetime(specific_dates, format='%d.%m.%Y')})
all_dates = pd.concat([last_dates, future_dates], ignore_index=True)

forecast = model.predict(all_dates)

print('forecast', forecast.columns)
print(forecast['yhat'][-40:])

predicted_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'дата', 'yhat': 'выход'})
predicted_data['направление'] = data_old['направление']

# predicted_data.to_csv('predicted_data.csv', index=False)    

import json
data1 = forecast['yhat'].tolist()[-40:]

with open('forecast_value.json', 'w') as file:
    json.dump(data1, file)
    
# # Визуализация прогноза
# fig = model.plot(forecast)
# plt.title('Прогноз с использованием Prophet')
# plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Загружаем данные
data = predicted_data

# Закодируем целевую переменную
data['направление'] = data['направление'].replace({'ш': 0, 'л': 1})

# Преобразование даты
data['дата'] = pd.to_datetime(data['дата'])  # Преобразование в datetime
data['год'] = data['дата'].dt.year
data['месяц'] = data['дата'].dt.month
data['день'] = data['дата'].dt.day
data['день_недели'] = data['дата'].dt.dayofweek  # Пн=0, Вск=6

# Разделим данные на известные и неизвестные направления
known_data = data[data['направление'].notna()]
unknown_data = data[data['направление'].isna()]

# Подготовка известных данных
X_known = known_data[['выход', 'год', 'месяц', 'день', 'день_недели']]
y_known = known_data['направление']

# Масштабирование данных
scaler = StandardScaler()
X_known_scaled = scaler.fit_transform(X_known)

# Обработка дисбаланса классов с помощью SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_known_scaled, y_known)

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.05, random_state=42)

# Определение модели
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Расчет MAE
mae = mean_absolute_error(y_test, y_pred)

print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")

# Предсказание направления для неизвестных данных
if not unknown_data.empty:
    unknown_data['направление'] = model.predict(unknown_data[['выход', 'год', 'месяц', 'день', 'день_недели']])

unknown_data['направление'] = unknown_data['направление'].astype(int)

print(unknown_data)

import json
with open('forecast_class.json', 'w') as file:
    json.dump(unknown_data['направление'].tolist(), file)

