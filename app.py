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

# Compute Daily Returns
tcs['Daily Return'] = tcs['Adj Close'].pct_change()

# Calculate Rolling Annualized Volatility (252 trading days in a year)
tcs['Volatility'] = tcs['Daily Return'].rolling(window=30).std() * np.sqrt(252)

# Prepare CPI Inflation Data
cpi_data = {
    'Date': pd.date_range(start='2022-01-01', periods=len([
        5.837563452, 5.042016807, 5.351170569, 6.32805995, 6.965174129, 6.162695152, 5.781758958,
        5.853658537, 6.488240065, 6.084867894, 5.409705648, 5.502392344, 6.155075939, 6.16,
        5.793650794, 5.090054816, 4.418604651, 5.572755418, 7.544264819, 6.912442396, 5.02, 4.87,
        5.55, 5.69, 5.1, 5.09, 4.85, 4.83, 4.75, 5.08, 3.54, 3.65, 5.49
    ]), freq='M'),
    'CPI Inflation': [
        5.837563452, 5.042016807, 5.351170569, 6.32805995, 6.965174129, 6.162695152, 5.781758958,
        5.853658537, 6.488240065, 6.084867894, 5.409705648, 5.502392344, 6.155075939, 6.16,
        5.793650794, 5.090054816, 4.418604651, 5.572755418, 7.544264819, 6.912442396, 5.02, 4.87,
        5.55, 5.69, 5.1, 5.09, 4.85, 4.83, 4.75, 5.08, 3.54, 3.65, 5.49
    ]
}
cpi_df = pd.DataFrame(cpi_data)

# Merge with stock data
tcs = tcs.resample('M').last().reset_index()
tcs = pd.merge(tcs, cpi_df, on='Date', how='inner')
tcs.dropna(inplace=True)

# Define Features (CPI Inflation) and Target Variables (Return & Volatility)
X = tcs[['CPI Inflation']]
y_return = tcs['Daily Return']
y_volatility = tcs['Volatility']

# Split data for training & testing
X_train, X_test, y_return_train, y_return_test = train_test_split(X, y_return, test_size=0.2, random_state=42)
X_train, X_test, y_vol_train, y_vol_test = train_test_split(X, y_volatility, test_size=0.2, random_state=42)

# Train Random Forest Model
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
ax[0].plot(X_test, y_return_pred, color='red', label='Predicted Returns')
ax[0].set_xlabel('CPI Inflation')
ax[0].set_ylabel('Stock Return')
ax[0].set_title('Inflation vs Stock Return')
ax[0].legend()

# Inflation vs Stock Volatility
ax[1].scatter(tcs['CPI Inflation'], tcs['Volatility'], label='Actual Volatility', color='blue')
ax[1].plot(X_test, y_vol_pred, color='red', label='Predicted Volatility')
ax[1].set_xlabel('CPI Inflation')
ax[1].set_ylabel('Stock Volatility')
ax[1].set_title('Inflation vs Stock Volatility')
ax[1].legend()

# Show plot in Streamlit
st.pyplot(fig)
