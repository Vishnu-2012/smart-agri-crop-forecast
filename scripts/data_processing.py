import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    # Load the raw data
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Handle missing values, incorrect data, etc.
    data = data.dropna()
    return data

def filter_data(data, columns):
    # Select only necessary columns for modeling
    filtered_data = data[columns]
    return filtered_data

def scale_data(data):
    # Scale data using MinMaxScaler for LSTM models
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def preprocess_data(file_path, columns):
    # Full pipeline of data loading, cleaning, filtering, and scaling
    data = load_data(file_path)
    data = clean_data(data)
    filtered_data = filter_data(data, columns)
    scaled_data, scaler = scale_data(filtered_data)
    return scaled_data, scaler
