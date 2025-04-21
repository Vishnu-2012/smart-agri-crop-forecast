# 🌾 Smart Agriculture: Crop Production Forecasting using LSTM
## Overview
-This project is part of a government-sanctioned initiative to develop a **smart agriculture website**. The primary goal is to forecast crop production trends using machine learning models, enabling better agricultural planning and decision-making.
-The project leverages **Long Short-Term Memory (LSTM)** networks to predict future crop production based on historical data. The results provide valuable insights into crop yields, helping farmers, policymakers, and other stakeholders make informed decisions.

## 🗂️ Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Data](#data)
- [Methodology](#methodology)
- [Model](#model)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)

## Objectives
- 🧠 Predict future crop production using historical yield data
- 📈 Visualize production trends for key crops across multiple years
- 📊 Deliver insights for better agricultural resource planning

## Data
 **Time span**: 1988–2023  
- **Crops included**: Tomato, Onion, Okra, Potato, Cauliflower, Peas  
- **Source**: Government-authorized agriculture production datasets  
- **Storage**: Located in the `data/raw/` directory


## Methodology

### 1. 📥 Data Preparation
- Collected annual crop production data from **1988 to 2023** for multiple crops including tomato, okra, and potato.
- Normalized and interpolated data to generate **day-wise production figures**, ensuring smooth input for time-series modeling.

### 2. 🧠 Feature Engineering
- Structured data into input-output sequences using a **sliding window approach** to create training sets for the LSTM model.
- Designed separate datasets for each crop to capture unique yield patterns.

### 3. 🏗️ LSTM Model Construction
- **Input Layer**: Accepts sequential time-series crop data.
- **LSTM Layers**: Captures long-term dependencies and temporal patterns.
- **Dense Layers**: Outputs the forecasted production value for each time step.

### 4. 🧪 Model Training
- Trained crop-specific LSTM models to forecast production from **2024 to 2030**.
- Used **MSE** as the loss function and **Adam optimizer** for efficient convergence.

### 5. 📊 Evaluation & Visualization
- Compared predicted values against historical trends.
- Visualized results using **Matplotlib**, highlighting both historical and future crop production trends.
- Saved plots in the `results` folder.

> 📌 The use of LSTM was critical for learning long-term dependencies — something traditional models fail at — making it ideal for forecasting crop yields based on decades of historical data.

# Model

This project uses **LSTM neural networks**, specifically designed to learn from sequential data such as time-series crop production.

### 🔑 Key Features:
- Crop-specific model training
- Multi-year forecasting window
- `.keras` model format for deployment-ready architecture

# 📈 Results

Visual trend plots and predictions are stored in:
results directory
Example Output (Optional):
![Production Forecast](results/figures/tomato_forecast.png)


> These predictions are valuable for understanding long-term yield patterns and enhancing agricultural planning.

---

## Technologies Used

- **Language**: Python 3.x  
- **Libraries**:
  - `tensorflow`, `keras`
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
- **IDE**: Jupyter Notebook  
- 📄 All dependencies listed in `requirements.txt`

## Future Work

- 🌐 Deploy the model on a Smart Agriculture platform/dashboard
- 🌦️ Include weather, irrigation, and soil features for richer forecasts
- 🤖 Explore hybrid modeling approaches (e.g., LSTM + GRU or XGBoost)


### Requirements:
1. Python 3.x
2. Jupyter Notebook
3. Libraries (specified in `requirements.txt`):
   - `tensorflow`
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `keras`

##  License

This project is licensed under the **Apache 2.0 License**
. 

