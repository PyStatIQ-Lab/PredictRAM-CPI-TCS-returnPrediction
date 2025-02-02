import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Fetch TCS.NS Stock Data
tcs = yf.download('TCS.NS', start="2022-01-01", end="2024-10-01")

# Compute Daily Returns using 'Close' price
tcs['Daily Return'] = tcs['Close'].pct_change()

# Calculate Rolling Annualized Volatility (252 trading days in a year)
tcs['Volatility'] = tcs['Daily Return'].rolling(window=30).std() * np.sqrt(252)

# Prepare CPI Inflation Data (Month-Year format -> CPI values)
cpi_data = {
    'Date': [
        '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01',
        '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01',
        '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01',
        '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01',
        '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01',
        '2024-07-01', '2024-08-01', '2024-09-01'
    ],
    'CPI Inflation': [
        5.837563452, 5.042016807, 5.351170569, 6.32805995, 6.965174129, 6.162695152, 5.781758958,
        5.853658537, 6.488240065, 6.084867894, 5.409705648, 5.502392344, 6.155075939, 6.16,
        5.793650794, 5.090054816, 4.418604651, 5.572755418, 7.544264819, 6.912442396, 5.02, 4.87,
        5.55, 5.69, 5.1, 5.09, 4.85, 4.83, 4.75, 5.08, 3.54, 3.65, 5.49
    ]
}

# Convert the CPI data into DataFrame
cpi_df = pd.DataFrame(cpi_data)

# Convert 'Date' column in CPI data to datetime format
cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])

# Ensure 'Date' column in TCS data is also in datetime format
tcs['Date'] = pd.to_datetime(tcs.index)

# Resample TCS data to monthly frequency and reset index
tcs_resampled = tcs.resample('M').last()  # Resample to monthly frequency
tcs_resampled = tcs_resampled.reset_index(drop=True)

# Check for matching date ranges between TCS and CPI data
tcs_dates = tcs_resampled['Date'].dt.date
cpi_dates = cpi_df['Date'].dt.date

# Filter out CPI data to match only the dates available in TCS stock data
cpi_df_filtered = cpi_df[cpi_df['Date'].dt.date.isin(tcs_dates)]

# Ensure both 'Date' columns are in the same format
tcs_resampled['Date'] = tcs_resampled['Date'].dt.date
cpi_df_filtered['Date'] = cpi_df_filtered['Date'].dt.date

# Check for common dates between TCS and CPI data
common_dates = set(tcs_resampled['Date']).intersection(set(cpi_df_filtered['Date']))
if not common_dates:
    print("No common dates found between TCS and CPI data.")
else:
    print(f"Common dates found: {len(common_dates)}")

# Check if either DataFrame is empty after filtering
if cpi_df_filtered.empty:
    print("Filtered CPI data is empty.")
if tcs_resampled.empty:
    print("TCS resampled data is empty.")

# Merge the two datasets on 'Date'
tcs = pd.merge(tcs_resampled, cpi_df_filtered, on='Date', how='inner')

# Drop any rows with missing data after merging
tcs.dropna(inplace=True)

# Define Features (CPI Inflation) and Target Variables (Return & Volatility)
X = tcs[['CPI Inflation']]
y_return = tcs['Daily Return']
y_volatility = tcs['Volatility']

# Split data for training & testing
X_train, X_test, y_return_train, y_return_test = train_test_split(X, y_return, test_size=0.2, random_state=42)
X_train, X_test, y_vol_train, y_vol_test = train_test_split(X, y_volatility, test_size=0.2, random_state=42)

# Train Random Forest Model for Return and Volatility
rf_return = RandomForestRegressor(n_estimators=100, random_state=42)
rf_vol = RandomForestRegressor(n_estimators=100, random_state=42)

rf_return.fit(X_train, y_return_train)
rf_vol.fit(X_train, y_vol_train)

# Predict on test data
y_return_pred = rf_return.predict(X_test)
y_vol_pred = rf_vol.predict(X_test)

# Model Accuracy
return_mae = mean_absolute_error(y_return_test, y_return_pred)
volatility_mae = mean_absolute_error(y_vol_test, y_vol_pred)

# Streamlit UI
st.title("TCS.NS Stock Prediction Based on CPI Inflation")
st.write("This app predicts TCS.NS stock return and risk based on expected inflation changes.")

# User Input for Expected CPI Inflation
cpi_input = st.number_input("Enter Expected CPI Inflation:", min_value=0.0, max_value=10.0, value=5.0, step=0.01)

# Predict Future Risk & Return
future_cpi = np.array([cpi_input]).reshape(-1, 1)
predicted_return = rf_return.predict(future_cpi)[0]
predicted_volatility = rf_vol.predict(future_cpi)[0]

# Display Results
st.subheader("Prediction Results:")
st.write(f"📈 **Predicted Stock Return:** {predicted_return:.4f}")
st.write(f"📉 **Predicted Stock Volatility:** {predicted_volatility:.4f}")

# Show Model Accuracy
st.subheader("Model Accuracy:")
st.write(f"✅ **Mean Absolute Error (Return Prediction):** {return_mae:.6f}")
st.write(f"✅ **Mean Absolute Error (Volatility Prediction):** {volatility_mae:.6f}")

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Inflation vs Stock Return
ax[0].scatter(tcs['CPI Inflation'], tcs['Daily Return'], label='Actual Returns', color='blue')
ax[0].plot(tcs['CPI Inflation'], rf_return.predict(tcs[['CPI Inflation']]), color='red', label='Predicted Returns')  # Use the full data for prediction
ax[0].set_xlabel('CPI Inflation')
ax[0].set_ylabel('Stock Return')
ax[0].set_title('Inflation vs Stock Return')
ax[0].legend()

# Inflation vs Stock Volatility
ax[1].scatter(tcs['CPI Inflation'], tcs['Volatility'], label='Actual Volatility', color='blue')
ax[1].plot(tcs['CPI Inflation'], rf_vol.predict(tcs[['CPI Inflation']]), color='red', label='Predicted Volatility')  # Use the full data for prediction
ax[1].set_xlabel('CPI Inflation')
ax[1].set_ylabel('Stock Volatility')
ax[1].set_title('Inflation vs Stock Volatility')
ax[1].legend()

# Show plot in Streamlit
st.pyplot(fig)
