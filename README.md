

# Smart Agriculture Crop Forecasting

## Project Overview
This project is part of a government-sanctioned initiative to develop a **smart agriculture website**. The primary goal is to forecast crop production trends using machine learning models, enabling better agricultural planning and decision-making.

The project leverages **Long Short-Term Memory (LSTM)** networks to predict future crop production based on historical data. The results provide valuable insights into crop yields, helping farmers, policymakers, and other stakeholders make informed decisions.

## Objectives
- **Predict future crop production**: Using historical data on crop production, we aim to build models that forecast crop yields for future years.
- **Analyze production trends**: Visualize past and predicted crop production trends, allowing stakeholders to monitor crop performance over time.
- **Provide insights for decision-making**: Enable informed agricultural planning by offering reliable predictions of crop yields.

## Data
The project uses historical data on crop production from 1988 to 2023, covering several key crops:
- **Tomato**
- **Onion**
- **Okra**
- **Potato**
- **Cauliflower**
- **Peas**

### Data Files
- Raw data is stored in the `data/raw/` directory.

## Analysis
The project performs the following analyses:
1. **Data Preprocessing**: 
   - Cleaning and organizing the raw data for model input.
   - Handling missing values and outliers, if any.
2. **Time-Series Forecasting**:
   - Using **LSTM neural networks** for predicting crop production trends.
   - Training separate models for each crop to maximize accuracy.
3. **Visualization**:
   - Generate plots to visualize historical and predicted crop production trends.
   - The graphs are stored in the `results/figures/` directory.

## Model
The **LSTM model** is used due to its ability to capture patterns in sequential data. The model is trained separately for each crop, allowing for more accurate predictions. The models are saved in the `models/` directory in `.keras` format for future use.

### Key Features:
- **Multi-year forecasting**: Predict crop production for a specified range of years.
- **Separate models for each crop**: Individual models tailored to the unique production patterns of each crop.
- **Visualization of trends**: Clear, color-coded graphs depicting future crop yields.



## Results

The predictions and trends are visualized and saved in the `results/figures/` directory. These insights can help guide agricultural planning and resource allocation for the coming years.

### Requirements:
1. Python 3.x
2. Jupyter Notebook
3. Libraries (specified in `requirements.txt`):
   - `tensorflow`
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`

## Future Work
- Integrate the forecast models with the smart agriculture website for real-time predictions.
- Expand the model to include other factors like weather patterns and soil conditions.
- Explore advanced machine learning techniques for improving prediction accuracy.

## License
This project is licensed under the **Apache 2.0 License**. 

