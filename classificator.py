import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Загружаем данные
data = pd.read_csv('predicted_data.csv')

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

