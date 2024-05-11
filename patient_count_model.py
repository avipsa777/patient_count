import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv('New_generated_data.csv')

# Assuming 'Date' is a string, convert it to a datetime object
df['Arrival Time'] = pd.to_datetime(df['Arrival Time'])
df['Date'] = pd.to_datetime(df['Date'])
df['Hour'] = df['Arrival Time'].dt.hour

# Group by hour and day to get patient counts
patient_count = df.groupby(['Hour', df['Date'].dt.dayofweek]).size()
patient_count = patient_count.astype(np.int64).reset_index(name='PatientCount')

X = patient_count[['Hour']].values.reshape(-1, 1)
y = patient_count['PatientCount'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression() # create
model.fit(X_train, y_train) # fit

# Predictions on test data
y_pred = model.predict(X_test)

# Evaluating the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# print(f'R-Squared Error: {r2}')

# Visualizing the predictions
plt.scatter(X_test, y_test, color='b', label='Actual')
plt.plot(X_test, y_pred, color='r', label='Predicted')
plt.xlabel("Hour of the Day")
plt.ylabel("Patient count")
plt.legend()
plt.show() 

# Using the trained model for Demand Forecasting
future_hours = [9, 10, 11, 12, 13, 14, 15, 16, 19, 20]
forecasted_patient_count = model.predict(np.array(future_hours).reshape(-1, 1))
for hour, count in zip(future_hours, forecasted_patient_count):
    print(f'Forecasted Patient Count at {hour} hours: {count}')


# # Serialize the trained model to a file
#joblib.dump(model, 'patient_count_model.pkl')