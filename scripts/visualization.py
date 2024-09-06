import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(actual_data, predicted_data, years, crop_name):
    plt.figure(figsize=(10,6))
    
    # Plot actual data
    plt.plot(years[:len(actual_data)], actual_data, label=f"Actual {crop_name}", marker='o', color='blue')
    
    # Plot predicted data
    predicted_years = years[len(actual_data):]
    plt.plot(predicted_years, predicted_data, label=f"Predicted {crop_name}", marker='x', color='red')
    
    # Labels and title
    plt.title(f"{crop_name} Production Area Prediction")
    plt.xlabel("Year")
    plt.ylabel("Production Area")
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

def visualize_all_crops(crop_data, crop_predictions, years):
    plt.figure(figsize=(12,8))
    
    # Plot for each crop
    for crop_name, (actual_data, predicted_data) in crop_data.items():
        all_years = years + [year for year in range(len(actual_data), len(actual_data) + len(predicted_data))]
        plt.plot(all_years, actual_data + predicted_data, label=crop_name)
    
    plt.title("Crop Production Area Prediction for Multiple Crops")
    plt.xlabel("Year")
    plt.ylabel("Production Area")
    plt.legend()
    plt.grid(True)
    plt.show()
