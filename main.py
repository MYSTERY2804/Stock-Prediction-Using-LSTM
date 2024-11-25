import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Step 1: Download historical data
symbol = 'RELIANCE.NS'  # RELIANCE.NS is the ticker for Reliance Industries on NSE
data = yf.download(symbol, start="1980-01-01", end="2024-10-05")

# Step 2: Prepare data for Prophet
df = data.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']

# Step 3: Train the Prophet model
model = Prophet(daily_seasonality=True)
model.fit(df)

# Step 4: Create future dates and make predictions
future_dates = model.make_future_dataframe(periods=3)
forecast = model.predict(future_dates)

# Step 5: Plot the forecast
model.plot(forecast)
plt.title("Reliance Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Close Price")
plt.show()

# Step 6: Extract prediction for October 7, 2024
predicted_price = forecast[forecast['ds'] == '2024-10-07']['yhat'].values[0]
print(f"Predicted price for Reliance on 7th October 2024: {predicted_price}")
