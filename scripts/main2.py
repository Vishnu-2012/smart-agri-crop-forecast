from data_processing import preprocess_data
from model_training import train_model, load_model
from prediction import predict_for_years
from visualization import plot_predictions

# File path to your CSV data
file_path = "crop_data.csv"

# Specify the columns you are interested in (including 'Year' or others if needed)
columns = ['Year', 'Tomato', 'Onion', 'Okra', 'Potato', 'Cauliflower', 'Peas']

# Preprocess the data
scaled_data, scaler = preprocess_data(file_path, columns)

# Split the data into input features and target labels
# Example: Predict the Onion production (so 'Onion' would be your target)
input_data = scaled_data[:, :-1]  # Use all columns except the last one for input
target_data = scaled_data[:, -1]  # The last column (e.g., 'Onion') is the target

# Train the model on this crop's data
lstm_model = train_model(input_data, target_data, batch_size=64, epochs=20)

# Future prediction
years = list(range(2024, 2031))  # Example future years
predictions = predict_for_years("lstm_model.h5", input_data, years)

# Visualize actual vs predicted data
actual_data = target_data  # Real production data for Onion (or whichever crop you're predicting)
plot_predictions(actual_data, predictions, years, crop_name='Onion')
