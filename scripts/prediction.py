import numpy as np
from tensorflow.keras.models import load_model

def predict_future(model, data, future_steps):
    predictions = []
    
    # Take the last batch of data as input
    current_input = data[-1]
    
    for _ in range(future_steps):
        current_input = np.reshape(current_input, (1, current_input.shape[0], 1))
        predicted_value = model.predict(current_input)
        predictions.append(predicted_value[0][0])
        
        # Update current input with new prediction
        current_input = np.append(current_input[0][1:], predicted_value[0][0])
    
    return predictions

def predict_for_years(model_path, data, years):
    # Load model
    model = load_model(model_path)
    
    # Predict for a number of years
    predictions = predict_future(model, data, len(years))
    return predictions
