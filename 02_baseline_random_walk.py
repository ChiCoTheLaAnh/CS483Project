import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import load_config


def evaluate_forecast(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def main(cfg=None):
    cfg = cfg or load_config()
    print("--- Running Baseline Random Walk Model ---", flush=True)

    data_path = cfg.master_path
    if not os.path.exists(data_path):
        print(f"ERROR: Could not find {data_path}")
        return

    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    prices = df['Gold_Price']
    split_index = int(len(prices) * (1 - cfg.test_size_ratio))
    train_data = prices[:split_index]
    test_data = prices[split_index:]

    print(f"Training Samples: {len(train_data)}")
    print(f"Testing Samples: {len(test_data)}")

    predictions = test_data.shift(1)
    predictions.iloc[0] = train_data.iloc[-1]

    rmse, mae = evaluate_forecast(test_data, predictions)

    print("-" * 30)
    print("BASELINE MODEL RESULTS")
    print("-" * 30)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("-" * 30)

    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data.values, label='Actual Price', color='blue')
    plt.plot(test_data.index, predictions.values, label='Random Walk', color='red', linestyle='--')
    plt.title('Baseline Model: Random Walk vs Actual')
    plt.legend()
    plt.grid(True)
    baseline_plot = os.path.join(cfg.run_output_dir, 'baseline_results.png')
    plt.savefig(baseline_plot)
    print(f"Plot saved as {baseline_plot}")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "test_size_ratio": cfg.test_size_ratio,
        "data_path": data_path,
    }
    metrics_path = os.path.join(cfg.run_output_dir, 'baseline_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
