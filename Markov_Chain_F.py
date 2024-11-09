import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# Step 1: Read the Excel file
file_path = 'supplier_performance.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Step 2: Transform data to have suppliers as rows and dates as columns
data = data.set_index('Suppliers')

# Convert all data to numeric, coerce errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Step 3: Identify the top 5 suppliers based on average performance
average_performance = data.mean(axis=1)
top_5_suppliers = average_performance.nlargest(5).index.tolist()

# Step 4: Define bins and convert performance data to states
bins = [0, 40,60 , 75, 85, 97]
data_states = data.apply(lambda x: pd.cut(x, bins=bins, labels=bins[1:], include_lowest=True))

# Step 5: Train Markov Chain Model and Predict
transition_matrices = {}
predictions = []
actual_values = []
predicted_values = []
forecast_errors = []

for supplier in top_5_suppliers:
    supplier_data = data_states.loc[supplier].astype(str)
    supplier_data = supplier_data.dropna()  # Drop NaN values
    
    if len(supplier_data) == 0:
        # If no valid data for the supplier, skip to the next one
        predictions.append((supplier, 'Insufficient data', 'Insufficient data'))
        continue
    
    transitions = pd.crosstab(supplier_data.shift(), supplier_data, normalize='index')
    transition_matrices[supplier] = transitions

    # Predict the next state for the supplier
    current_state = supplier_data.iloc[-1]
    if pd.isna(current_state) or current_state not in transitions.index:
        next_state = "Insufficient data"
        predicted_value = np.nan
    else:
        next_state_prob = transition_matrices[supplier].loc[current_state]
        next_state = next_state_prob.idxmax()
        predicted_value = int(next_state)
    
    # Get the actual value (before converting to states)
    actual_value = data.loc[supplier].iloc[-1]
    
    # Calculate forecast error
    if next_state == "Insufficient data":
        forecast_error = np.nan
    else:
        forecast_error = abs(int(actual_value) - int(predicted_value))
    
    # Store results
    predictions.append((supplier, current_state, next_state))
    actual_values.append(actual_value)
    predicted_values.append(predicted_value)
    forecast_errors.append(forecast_error)

# Step 6: Save predictions and errors to a new tab in the same Excel file
output_df = pd.DataFrame({
    'Supplier': top_5_suppliers,
    'Actual Value': actual_values,
    'Predicted Value': predicted_values,
    'Forecast Error': forecast_errors
})

with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
    output_df.to_excel(writer, sheet_name='ForecastErrorsMarkov', index=False)

# Step 7: Scatter Plot Visualization for the top 5 suppliers
output_folder = 'Markov_Chain_Visualizations'
os.makedirs(output_folder, exist_ok=True)

for i, supplier in enumerate(top_5_suppliers):
    plt.figure(figsize=(12, 6))
    plt.scatter(data.columns, data.loc[supplier], marker='o', color='b', label='Actual Value')
    plt.axhline(y=predicted_values[i], color='r', linestyle='--', label='Predicted Value')
    plt.title(f'Supplier {supplier} - Actual vs Predicted Performance')
    plt.xlabel('Month')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_folder, f'{supplier}_performance.png'))
    plt.close()

print("Analysis completed. Results saved to Excel and visualizations generated.")
