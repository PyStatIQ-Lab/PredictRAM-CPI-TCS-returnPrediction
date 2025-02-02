import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📌 Step 1: Fetch TCS.NS Stock Data from Yahoo Finance
tcs = yf.download('TCS.NS', start='2022-01-01', end='2024-09-30')

# 📌 Step 2: Prepare Stock Data
tcs = tcs[['Close']]  # Use 'Close' price instead of 'Adj Close'
tcs = tcs.reset_index()  # Reset index to get 'Date' column
tcs['Date'] = pd.to_datetime(tcs['Date'])
tcs['Daily Return'] = tcs['Close'].pct_change()  # Calculate Daily Return
tcs.dropna(inplace=True)  # Remove NaN values

# 📌 Step 3: CPI Inflation Data (Manually Provided)
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
        5.83, 5.04, 5.35, 6.32, 6.96, 6.16, 5.78, 5.85, 6.48, 6.08, 5.40, 5.50,
        6.15, 6.16, 5.79, 5.09, 4.41, 5.57, 7.54, 6.91, 5.02, 4.87, 5.55, 5.69,
        5.10, 5.09, 4.85, 4.83, 4.75, 5.08, 3.54, 3.65, 5.49
    ]
}

cpi_df = pd.DataFrame(cpi_data)
cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])  # Convert Date column to datetime

# 📌 Step 4: Merge TCS Data with CPI Inflation Data
tcs = pd.merge(tcs, cpi_df, on='Date', how='inner')

# 📌 Step 5: Prepare Data for Machine Learning
X = tcs[['CPI Inflation']]
y = tcs['Daily Return']

# Split into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 6: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Step 7: Evaluate Model Performance
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Accuracy Metrics:")
print(f"   ✅ Mean Absolute Error (MAE): {mae:.6f}")
print(f"   ✅ Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"   ✅ R-squared (R²): {r2:.6f}")

# 📌 Step 8: Predict Stock Return for User-Input CPI Inflation
expected_cpi = float(input("\n📌 Enter Expected CPI Inflation (%): "))
predicted_return = model.predict([[expected_cpi]])[0]

print(f"\n📈 Predicted Stock Return for CPI {expected_cpi}%: {predicted_return:.6f}")

# 📌 Step 9: Visualizing Results
plt.figure(figsize=(10, 5))
sns.regplot(x=tcs['CPI Inflation'], y=tcs['Daily Return'], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel("CPI Inflation (%)")
plt.ylabel("TCS Daily Return")
plt.title("TCS Stock Returns vs CPI Inflation")
plt.grid(True)
plt.show()
