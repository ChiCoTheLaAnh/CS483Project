import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'master_dataset.csv')
TEST_SIZE_RATIO = 0.2

def evaluate_forecast(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

def main():
    print("--- Running Baseline Random Walk Model ---", flush=True)
    
    # 1. Load the fixed Master Dataset
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Could not find {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Setup Data
    prices = df['Gold_Price']
    split_index = int(len(prices) * (1 - TEST_SIZE_RATIO))
    train_data = prices[:split_index]
    test_data = prices[split_index:]
    
    print(f"Training Samples: {len(train_data)}")
    print(f"Testing Samples: {len(test_data)}")

    # 3. Random Walk Prediction (Predict t using t-1)
    predictions = test_data.shift(1)
    predictions.iloc[0] = train_data.iloc[-1]

    # 4. Calculate RMSE
    rmse, mae = evaluate_forecast(test_data, predictions)
    
    print("-" * 30)
    print("BASELINE MODEL RESULTS")
    print("-" * 30)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("-" * 30)

    # 5. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data.values, label='Actual Price', color='blue')
    plt.plot(test_data.index, predictions.values, label='Random Walk', color='red', linestyle='--')
    plt.title('Baseline Model: Random Walk vs Actual')
    plt.legend()
    plt.grid(True)
    plt.savefig('baseline_results.png')
    print("Plot saved as baseline_results.png")

if __name__ == "__main__":
    main()