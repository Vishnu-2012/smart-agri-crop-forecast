import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(data, target, batch_size=64, epochs=10):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    
    model = create_lstm_model((X_train.shape[1], 1))
    
    # Reshaping data to fit LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    
    # Save the model
    model.save("lstm_model.h5")
    
    return model

def load_model(model_path):
    # Load a saved LSTM model
    return tf.keras.models.load_model(model_path)
