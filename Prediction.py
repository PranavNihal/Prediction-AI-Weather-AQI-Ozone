import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

# Generate random data
np.random.seed(42)  # For reproducibility
num_samples = 1000

# Randomly generated values for features
pressure = np.random.uniform(980, 1030, num_samples)  # Pressure between 980 hPa and 1030 hPa
humidity = np.random.uniform(20, 100, num_samples)  # Humidity between 20% and 100%
gas_vocs = np.random.uniform(0, 1000, num_samples)  # Random VOCs concentration
altitude = np.random.uniform(0, 3000, num_samples)  # Altitude between 0m and 3000m
co2 = np.random.uniform(300, 5000, num_samples)  # CO2 levels
alcohol = np.random.uniform(0, 100, num_samples)  # Alcohol levels
toluene = np.random.uniform(0, 100, num_samples)  # Toluene levels
nh4 = np.random.uniform(0, 100, num_samples)  # NH4 levels
acetone = np.random.uniform(0, 100, num_samples)  # Acetone levels

# Randomly generated values for ozone, AQI, PM2.5
ozone = np.random.uniform(0, 300, num_samples)  # Ozone levels in ppb
aqi = np.random.uniform(0, 500, num_samples)  # AQI (0-500 scale)
pm25 = np.random.uniform(0, 250, num_samples)  # PM2.5 levels in µg/m³

# Randomly generated labels for weather and air quality
weather_conditions = np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Snowy'], num_samples)
air_quality_labels = np.random.choice(['Good', 'Moderate', 'Poor'], num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Pressure': pressure,
    'Humidity': humidity,
    'Gas_VOCs': gas_vocs,
    'Altitude': altitude,
    'CO2': co2,
    'Alcohol': alcohol,
    'Toluene': toluene,
    'NH4': nh4,
    'Acetone': acetone,
    'Ozone': ozone,
    'AQI': aqi,
    'PM2.5': pm25,
    'Weather': weather_conditions,
    'Air_Quality': air_quality_labels
})

# Preprocessing
X = data.drop(['Weather', 'Air_Quality', 'Ozone', 'AQI', 'PM2.5'], axis=1)  # Features
y_class = data[['Weather', 'Air_Quality']]  # Categorical target variables
y_reg = data[['Ozone', 'AQI', 'PM2.5']]  # Continuous target variables

# Splitting the dataset into training and testing sets for classification and regression tasks
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42)

# Initialize and train the model for categorical variables
clf_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf_model.fit(X_train, y_class_train)

# Initialize and train the model for continuous variables (Ozone, AQI, PM2.5)
reg_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
reg_model.fit(X_train, y_reg_train)

# Make predictions on the test set
y_class_pred = clf_model.predict(X_test)
y_reg_pred = reg_model.predict(X_test)

# Split the predictions for classification
weather_pred, air_quality_pred = y_class_pred[:, 0], y_class_pred[:, 1]

# Evaluate classification model
weather_accuracy = accuracy_score(y_class_test['Weather'], weather_pred)
air_quality_accuracy = accuracy_score(y_class_test['Air_Quality'], air_quality_pred)

print(f'Weather Classification Accuracy: {weather_accuracy * 100:.2f}%')
print(f'Air Quality Classification Accuracy: {air_quality_accuracy * 100:.2f}%')

# Classification reports for both outputs
print("\nWeather Classification Report:")
print(classification_report(y_class_test['Weather'], weather_pred))

print("\nAir Quality Classification Report:")
print(classification_report(y_class_test['Air_Quality'], air_quality_pred))

# Evaluate regression model
mse_ozone = mean_squared_error(y_reg_test['Ozone'], y_reg_pred[:, 0])
mse_aqi = mean_squared_error(y_reg_test['AQI'], y_reg_pred[:, 1])
mse_pm25 = mean_squared_error(y_reg_test['PM2.5'], y_reg_pred[:, 2])

print(f"\nMean Squared Error for Ozone: {mse_ozone:.2f}")
print(f"Mean Squared Error for AQI: {mse_aqi:.2f}")
print(f"Mean Squared Error for PM2.5: {mse_pm25:.2f}")

# Predicting with new random readings
new_readings = pd.DataFrame([[1010, 50, 300, 500, 400, 5, 3, 1, 2]],
                              columns=['Pressure', 'Humidity', 'Gas_VOCs', 'Altitude',
                                       'CO2', 'Alcohol', 'Toluene', 'NH4', 'Acetone'])

# Predictions for categorical (weather and air quality)
predicted_conditions = clf_model.predict(new_readings)
# Predictions for continuous (Ozone, AQI, PM2.5)
predicted_ozone_aqi_pm25 = reg_model.predict(new_readings)

print(f'\nPredicted Weather: {predicted_conditions[0][0]}')
print(f'Predicted Air Quality: {predicted_conditions[0][1]}')
print(f'Predicted Ozone: {predicted_ozone_aqi_pm25[0][0]:.2f} ppb')
print(f'Predicted AQI: {predicted_ozone_aqi_pm25[0][1]:.2f}')
print(f'Predicted PM2.5: {predicted_ozone_aqi_pm25[0][2]:.2f} µg/m³')

# Visualization of feature importance for weather prediction
importances = clf_model.estimators_[0].feature_importances_
feature_names = X.columns

# Sort the feature importance
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances for Weather Prediction")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()  # Adjust layout for better spacing
plt.show()