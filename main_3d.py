import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sea
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
rnn_model_online = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
    Dense(1)  # Single output for price prediction
])

rnn_model_online.compile(optimizer='adam', loss='mse')

# Train the RNN model for Online-Price
rnn_model_online.fit(X_train_rnn, y_train_online, epochs=50, batch_size=8, verbose=1)

# Predict Online-Price with RNN
y_pred_online_rnn = rnn_model_online.predict(X_test_rnn)

# Create animation for True vs Predicted Online Prices
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('True vs Predicted Online Prices (RNN)', fontsize=16)
ax.set_xlabel('Test Samples', fontsize=14)
ax.set_ylabel('Price (in currency)', fontsize=14)
ax.set_xlim(0, len(y_test_online))
ax.set_ylim(min(y_test_online.min(), y_pred_online_rnn.min()) - 10, max(y_test_online.max(), y_pred_online_rnn.max()) + 10)

true_line, = ax.plot([], [], label='True Online Price', marker='o', color='blue')
pred_line, = ax.plot([], [], label='Predicted Online Price (RNN)', marker='x', color='red')
ax.legend()
ax.grid(True)

# Initialize the animation
def init():
    true_line.set_data([], [])
    pred_line.set_data([], [])
    return true_line, pred_line

# Update the animation
def update(frame):
    true_line.set_data(range(frame), y_test_online[:frame])
    pred_line.set_data(range(frame), y_pred_online_rnn[:frame].flatten())
    return true_line, pred_line

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(y_test_online), init_func=init, blit=True, interval=100)

# Show the animation
plt.show()

