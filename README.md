# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```c

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and prepare data
iot_data = pd.read_csv('IOT-temp.csv')
iot_data['noted_date'] = pd.to_datetime(iot_data['noted_date'], format='%d-%m-%Y %H:%M')
iot_data.set_index('noted_date', inplace=True)
monthly_temp = iot_data['temp'].resample('MS').mean()

# Scale the data
scaler = MinMaxScaler()
scaled_temp = pd.Series(scaler.fit_transform(monthly_temp.values.reshape(-1, 1)).flatten(), index=monthly_temp.index)
scaled_temp += 1  # for multiplicative seasonality

# Split into training and testing sets
train_size = int(len(scaled_temp) * 0.8)
train_data = scaled_temp[:train_size]
test_data = scaled_temp[train_size:]

# Fit Holt-Winters model
model = ExponentialSmoothing(train_data, trend='add').fit()
forecast = model.forecast(len(test_data))


# Plot
plt.figure(figsize=(10, 6))
train_data.plot(label='Train')
test_data.plot(label='Test')
forecast.plot(label='Forecast')
plt.legend()
plt.title('Holt-Winters Forecast vs Actual')
plt.grid(True)
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(test_data, forecast))
print("RMSE:", rmse)




```
### OUTPUT:


TEST_PREDICTION
![image](https://github.com/user-attachments/assets/41ed737d-2f04-4a5c-9f5f-a14de535c172)



FINAL_PREDICTION

![image](https://github.com/user-attachments/assets/039b2922-8ab7-4f2e-a320-2f18e415e6de)



### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
