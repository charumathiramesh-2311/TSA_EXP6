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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# Load the IoT temperature dataset
data = pd.read_excel('/path/to/IOT-temp.xls') 

print(data.head()) 

# Assuming the second column is temperature readings, and first column is datetime
data.columns = ['Date', 'Temperature'] 

data['Date'] = pd.to_datetime(data['Date'])  
data.set_index('Date', inplace=True)

# Resample to monthly data
data_monthly = data.resample('MS').mean() 
print(data_monthly.head())

# Plot original monthly resampled data
data_monthly.plot(title='Monthly Resampled IoT Temperature Data')
plt.show()

# Scale the data
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(), 
                        index=data_monthly.index)

# Plot scaled data
scaled_data.plot(title='Scaled Monthly IoT Temperature Data')
plt.show()

# Decomposition to check trend and seasonality
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()

# Handle non-positive values for multiplicative seasonality
scaled_data = scaled_data + 1

# Split into training and testing
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Build Holt-Winters model (additive trend, multiplicative seasonality)
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

# Forecast on test data
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot the results
ax = train_data.plot(label='Train Data')
test_data.plot(ax=ax, label='Test Data')
test_predictions_add.plot(ax=ax, label='Predictions')
ax.legend()
ax.set_title('Holt-Winters Model Prediction on Test Data')
plt.show()

# Model performance
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f'RMSE on test data: {rmse:.4f}')

print(f"Standard Deviation: {np.sqrt(scaled_data.var()):.4f}")
print(f"Mean: {scaled_data.mean():.4f}")

# Train final model on all data and predict future
final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4)) 

# Plot final prediction
ax = scaled_data.plot(label='Scaled Data')
final_predictions.plot(ax=ax, label='Future Predictions')
ax.legend()
ax.set_title('Final Prediction using Holt-Winters')
plt.show()



```
### OUTPUT:


TEST_PREDICTION

![image](https://github.com/user-attachments/assets/160c753d-eba0-42f2-95c3-1df6a6a7cfbd)


FINAL_PREDICTION

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
