import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load the dataset
df = pd.read_csv(r'PhoneModel1.csv')

# Check the unique values
X = df[['Ram', 'ROM']].values
y_online = df['Online-Price'].values
y_retail = df['Retail-Price'].values

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train_online, y_test_online, y_train_retail, y_test_retail = train_test_split(
    X, y_online, y_retail, test_size=0.3, random_state=42)

# Standardize the feature data (RNN benefits from normalized input)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for RNN input (samples, timesteps, features), assuming timestep=1 since no explicit time-series data is provided
X_train_rnn = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_rnn = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build RNN model (LSTM)
rnn_model_online = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
    tf.keras.layers.Dense(1)  # Single output for price prediction
])

rnn_model_online.compile(optimizer='adam', loss='mse')

# Train the RNN model for Online-Price
rnn_model_online.fit(X_train_rnn, y_train_online, epochs=50, batch_size=8, verbose=1)

# Predict Online-Price with RNN
y_pred_online_rnn = rnn_model_online.predict(X_test_rnn)

# Visualize the predictions for Online Price
plt.figure(figsize=(10, 6))
plt.plot(y_test_online, label='True Online Price', marker='o')
plt.plot(y_pred_online_rnn, label='Predicted Online Price (RNN)', marker='x')
plt.legend()
plt.title('True vs Predicted Online Prices (RNN)')
plt.xlabel('Test Samples')
plt.ylabel('Price (in currency)')

plt.ylim(10, 100)
plt.grid(True)
plt.show()

