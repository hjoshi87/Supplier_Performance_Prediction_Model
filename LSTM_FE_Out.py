import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# Load the data
file_path = 'supplier_performance1.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Set the 'Suppliers' column as the index
data.set_index('Suppliers', inplace=True)

# Transpose the dataframe to have dates as rows and suppliers as columns
data = data.transpose()

# 1. Percentage of Missing Data
missing_percentage = data.isnull().mean() * 100
print("Percentage of missing data for each supplier:")
print(missing_percentage)

# 2. Data Cleaning

# Record percentage of missing data before imputation for the top 5 suppliers
data_before_imputation = data.copy()

# Remove suppliers with 50% or more missing data
data = data.loc[:, missing_percentage < 50]

# Apply KNN imputation where 20%-49% of data is missing (considering months as neighbors)
knn_imputer = KNNImputer(n_neighbors=5)
columns_knn = missing_percentage[(missing_percentage >= 20) & (missing_percentage < 50)].index
data_knn = data[columns_knn].copy()  # Copy to avoid changes in the original dataframe
data_knn = pd.DataFrame(knn_imputer.fit_transform(data_knn), index=data.index, columns=columns_knn)
data[columns_knn] = data_knn

# Apply linear interpolation where 1%-19% of data is missing
columns_interpolate = missing_percentage[(missing_percentage >= 1) & (missing_percentage < 20)].index
data[columns_interpolate] = data[columns_interpolate].interpolate(method='linear', axis=0)

# Forward fill and backward fill to handle any remaining missing values
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)
print(data)
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Function to create sequences for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    return np.array(sequences), np.array(targets)

# Define sequence length
sequence_length = 3  # Use the past 3 months to predict the next month

# Prepare the data for LSTM
X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Create and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(X_train.shape[2]))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=1, validation_data=(X_test, y_test))

# Make predictions for the test set
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Calculate deviations and forecast errors
deviations = predictions - y_test
errors = np.mean(np.abs(deviations), axis=0)  # Mean Absolute Error for each supplier

# Select 5 suppliers with the highest forecast errors
top_5_suppliers = data.columns[np.argsort(errors)[-5:]]

# Prepare data for Excel output
forecast_errors_df = pd.DataFrame({
    'Supplier': top_5_suppliers,
    'Forecast Error': errors[np.argsort(errors)[-5:]]
})

# Include predicted values, actual values, and percentage of missing data before imputation
predicted_values_df = pd.DataFrame(predictions, index=data.index[sequence_length+train_size:], columns=data.columns)
actual_values_df = pd.DataFrame(y_test, index=data.index[sequence_length+train_size:], columns=data.columns)

for supplier in top_5_suppliers:
    supplier_index = data.columns.get_loc(supplier)
    forecast_errors_df.loc[forecast_errors_df['Supplier'] == supplier, 'Predicted Values'] = \
        predicted_values_df[supplier].iloc[-1]
    forecast_errors_df.loc[forecast_errors_df['Supplier'] == supplier, 'Actual Values'] = \
        actual_values_df[supplier].iloc[-1]
    forecast_errors_df.loc[forecast_errors_df['Supplier'] == supplier, 'Percentage Missing Data'] = \
        missing_percentage[supplier]

# Save the forecast errors with additional information to Excel
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
    forecast_errors_df.to_excel(writer, sheet_name='LSTM_Forecast_Errors_')

print("Detailed Forecast Errors for Top 5 Suppliers:")
print(forecast_errors_df)

# 3. Visualization and Saving Plots

# Create a directory to save the plots
output_dir = 'LSTM_Visualizations2'
os.makedirs(output_dir, exist_ok=True)

# Plot and save scatter plots for the top 5 suppliers
for supplier in top_5_suppliers:
    supplier_index = data.columns.get_loc(supplier)
    
    actual_values = y_test[:, supplier_index]
    predicted_values = predictions[:, supplier_index]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(actual_values)), actual_values, color='blue', label='Actual Values')
    plt.scatter(range(len(predicted_values)), predicted_values, color='red', label='Predicted Values')
    plt.title(f'Performance Forecast for Supplier {supplier}')
    plt.xlabel('Months')
    plt.ylabel('Performance')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'Supplier_{supplier}_Forecast.png')
    plt.savefig(plot_path)
    plt.close()

print(f"Visualizations saved in '{output_dir}' directory.")
