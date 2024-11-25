import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from textblob import TextBlob
import requests
import datetime

# Constants
API_KEY = 'XYZ enter api key here for news'  # Replace with your News API key
TICKER = 'RELIANCE.NS'
START = '2015-01-01'
END = datetime.date.today()

# Step 1: Download stock data
df = yf.download(TICKER, start=START, end=END)
df.reset_index(inplace=True)

# Check if data is downloaded
if df.empty:
    raise ValueError("Stock data is empty. Please check the ticker symbol or the date range.")

# Step 2: Fetch news headlines and perform sentiment analysis
def get_news_sentiment(stock_ticker):
    url = f"https://newsapi.org/v2/everything?q={stock_ticker}&from={START}&to={END}&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    if not articles:
        raise ValueError("No articles found for the specified stock ticker.")

    sentiment_scores = []
    
    for article in articles:
        headline = article['title']
        analysis = TextBlob(headline)
        sentiment_scores.append(analysis.sentiment.polarity)
        
    return np.mean(sentiment_scores)

# Step 3: Add sentiment data to stock data
try:
    sentiment_score = get_news_sentiment(TICKER)
except ValueError as e:
    print(e)
    sentiment_score = 0  # Default to 0 if no news is available

df['Sentiment'] = sentiment_score

# Step 4: Prepare data for LSTM
data = df[['Date', 'Close', 'Sentiment']]
data['Close'] = data['Close'].shift(-1)  # Predicting the next day's price
data.dropna(inplace=True)

# Check if the data is valid after processing
if data.empty:
    raise ValueError("Data is empty after processing. Check if there are sufficient rows after shifting.")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'Sentiment']])

# Create training and test sets
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]

# Create the dataset for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0])  # Predicting the closing price
    return np.array(X), np.array(y)

# Set time step
time_step = 10
X_train, y_train = create_dataset(train, time_step)
X_test, y_test = create_dataset(test, time_step)

# Reshape input for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)

# Step 5: Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 2)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train the model
model.fit(X_train, y_train, batch_size=1, epochs=3)

# Step 7: Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros_like(predictions)), axis=1))[:, 0]

# Step 8: Plot results
plt.figure(figsize=(14, 5))
# Adjust the x-values to match the length of predictions
plt.plot(df['Date'][train_size + time_step + 1: train_size + time_step + 1 + len(predictions)], predictions, color='red', label='Predicted Price')
plt.plot(df['Date'], df['Close'], color='blue', label='Actual Price')
plt.title('Stock Price Prediction with News Sentiment')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Step 9: Predicting the next week's stock price
if len(scaled_data) >= time_step:
    last_data = scaled_data[-time_step:].reshape(1, time_step, 2)
    next_week_prediction = model.predict(last_data)
    next_week_prediction = scaler.inverse_transform(np.concatenate((next_week_prediction, np.zeros_like(next_week_prediction)), axis=1))[:, 0]
    print("Predicted stock price for next week:", next_week_prediction)
else:
    print("Not enough data to make predictions for next week.")

# Step 10: Feature Importance using Permutation Importance
# First, we need to flatten the test set for permutation importance
X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Reshape to (samples, features)
y_test_flat = y_test.reshape(-1, 1)

# Get permutation importance
perm_importance = permutation_importance(model, X_test_flat, y_test_flat, n_repeats=10, random_state=42)

# Display feature importance
feature_importance = perm_importance.importances_mean
features = ['Close', 'Sentiment']  # Your feature names
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importance:\n", importance_df)
