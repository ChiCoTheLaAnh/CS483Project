import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss

from config import load_config

warnings.filterwarnings("ignore")


def load_data(cfg):
    if not os.path.exists(cfg.master_path):
        raise FileNotFoundError(f"Data not found at {cfg.master_path}")

    df = pd.read_csv(cfg.master_path, index_col=0, parse_dates=True)
    if 'Gold_Price' not in df.columns:
        raise KeyError("Gold_Price column missing")
    return df


def check_stationarity(series, name):
    print(f"\n--- Analyzing: {name} ---")

    adf_result = adfuller(series.dropna())
    adf_p = adf_result[1]
    print(f"ADF p-value:  {adf_p:.4f}", end=" ")
    if adf_p < 0.05:
        print("(Stationary - REJECT Null)")
    else:
        print("(Non-Stationary - FAIL to Reject)")

    kpss_result = kpss(series.dropna(), regression='c', nlags="auto")
    kpss_p = kpss_result[1]
    print(f"KPSS p-value: {kpss_p:.4f}", end=" ")
    if kpss_p > 0.05:
        print("(Stationary - FAIL to Reject)")
    else:
        print("(Non-Stationary - REJECT Null)")


def main(cfg=None):
    cfg = cfg or load_config()
    print("--- 02_2: Statistical Diagnosis (Colleague's Findings) ---")

    df = load_data(cfg)
    price = df['Gold_Price']

    log_returns = np.log(price / price.shift(1)).dropna()

    print("\n[Finding 1]: Raw Prices follow a Random Walk")
    check_stationarity(price, "Raw Gold Price")

    print("\n[Finding 2]: Log Returns are Stationary (Predictable)")
    check_stationarity(log_returns, "Log Returns")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(price, color='blue')
    axes[0, 0].set_title('Raw Gold Prices (Non-Stationary)')
    axes[0, 0].grid(True)

    axes[0, 1].plot(log_returns, color='green')
    axes[0, 1].set_title('Log Returns (Stationary)')
    axes[0, 1].grid(True)

    plot_acf(price, ax=axes[1, 0], lags=40, title='ACF: Raw Price (High Correlation)')
    plot_acf(log_returns, ax=axes[1, 1], lags=40, title='ACF: Returns (Low Correlation)')

    plt.tight_layout()
    output_img = os.path.join(cfg.run_output_dir, 'statistical_diagnosis.png')
    plt.savefig(output_img)
    print(f"\nSaved diagnosis plot to {output_img}")
    print("-" * 40)
    print("CONCLUSION: Raw prices cannot be predicted via regression.")
    print("CONCLUSION: We must model Returns to beat the baseline.")
    print("-" * 40)


if __name__ == "__main__":
    main()
