import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import load_config

PREVIOUS_BEST_RMSE = 8.5126


def load_and_engineer_data(cfg):
    if not os.path.exists(cfg.model_ready_path):
        raise FileNotFoundError(f"Data not found at {cfg.model_ready_path}")

    df = pd.read_csv(cfg.model_ready_path, index_col=0, parse_dates=True)

    df['Log_Ret'] = np.log(df['Gold_Price'] / df['Gold_Price'].shift(1)) * 10

    for i in range(1, 4):
        df[f'Log_Ret_Lag{i}'] = df['Log_Ret'].shift(i)

    df['Recession_Regime'] = (df['SAHMREALTIME'] > 0.3).astype(int)
    df['Dollar_Regime_Interaction'] = df['DEXUSEU'].pct_change() * df['Recession_Regime']
    df['VIX_Regime_Interaction'] = df['VIXCLS'].diff() * df['SAHMREALTIME']

    df['Real_Rate_Change'] = (df['FEDFUNDS'] - (df['CPIAUCSL'].pct_change() * 1200)).diff()
    df['Yield_Spread'] = df['T10Y2Y'].diff()
    df['Dollar_Change'] = df['DEXUSEU'].pct_change() * 10

    delta = df['Gold_Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

    features = [col for col in df.columns if 'Lag' in col] + [
        'Real_Rate_Change', 'Yield_Spread', 'Dollar_Change', 'RSI',
        'SAHMREALTIME', 'Dollar_Regime_Interaction', 'VIX_Regime_Interaction'
    ]

    return df, features


def save_metrics(cfg, metrics):
    metrics_path = os.path.join(cfg.run_output_dir, 'regime_boost_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main(cfg=None):
    cfg = cfg or load_config()
    np.random.seed(cfg.random_seed)

    print("=" * 70)
    print("REGIME-AWARE GOLD PRICE PREDICTION")
    print("=" * 70)
    print(f"Run ID: {cfg.run_id}")
    print(f"Outputs will be written to: {cfg.run_output_dir}\n")

    df, features = load_and_engineer_data(cfg)
    print(f"\n✓ Loaded {len(df)} days of data")
    print(f"✓ Engineered {len(features)} features")

    X = df[features].values
    y = df['Log_Ret'].values

    split = int(len(X) * (1 - cfg.test_split))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    split_date = df.index[split]
    train_dates = df.index[:split]
    test_dates = df.index[split:]

    print(f"\n{'=' * 70}")
    print("TRAIN/TEST SPLIT")
    print(f"{'=' * 70}")
    print(f"Training:   {train_dates[0].date()} to {train_dates[-1].date()} ({len(X_train)} days, {(1 - cfg.test_split) * 100:.0f}%)")
    print(f"Testing:    {test_dates[0].date()} to {test_dates[-1].date()} ({len(X_test)} days, {cfg.test_split * 100:.0f}%)")
    print(f"Split Date: {split_date.date()}")
    print(f"{'=' * 70}\n")

    print("Training Gradient Boosting model...")
    model = GradientBoostingRegressor(
        **cfg.gb_params,
        random_state=cfg.random_seed
    )

    model.fit(X_train, y_train)
    print("✓ Training complete\n")

    pred_returns_train = model.predict(X_train)
    pred_returns_test = model.predict(X_test)

    train_prev_prices = df['Gold_Price'].iloc[0:split - 1].values
    train_actual_prices = df['Gold_Price'].iloc[1:split].values
    train_predicted_prices = train_prev_prices * np.exp(pred_returns_train[1:] / 10)

    test_prev_prices = df['Gold_Price'].iloc[split - 1:-1].values
    test_actual_prices = df['Gold_Price'].iloc[split:].values
    test_predicted_prices = test_prev_prices * np.exp(pred_returns_test / 10)

    train_rmse = np.sqrt(mean_squared_error(train_actual_prices, train_predicted_prices))
    test_rmse = np.sqrt(mean_squared_error(test_actual_prices, test_predicted_prices))
    train_mae = mean_absolute_error(train_actual_prices, train_predicted_prices)
    test_mae = mean_absolute_error(test_actual_prices, test_predicted_prices)

    print(f"{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Metric':<20} {'Training':<15} {'Testing':<15}")
    print(f"{'-' * 70}")
    print(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")
    print(f"{'MAE':<20} {train_mae:<15.4f} {test_mae:<15.4f}")
    print(f"{'=' * 70}\n")

    print(f"🎯 OUT-OF-SAMPLE TEST RMSE: {test_rmse:.4f}")
    print("📊 Random Walk Baseline:    9.34")
    print(f"💪 Improvement:             {9.34 - test_rmse:.4f} ({((9.34 - test_rmse) / 9.34) * 100:.1f}%)\n")

    if train_rmse < test_rmse * 0.7:
        print("⚠️  Warning: Possible overfitting (train << test)")
    else:
        print(f"✅ Good generalization (Train/Test ratio: {test_rmse / train_rmse:.2f})")

    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print(f"\n{'=' * 70}")
    print("TOP 10 FEATURE IMPORTANCE")
    print(f"{'=' * 70}")
    for idx, row in importance.head(10).iterrows():
        print(f"{row['Feature']:<30} {row['Importance']:.4f}")
    print(f"{'=' * 70}\n")

    print("Generating plots...\n")

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    ax1 = axes[0]
    ax1.plot(df.index, df['Gold_Price'], label='Actual Price',
             color='black', linewidth=1.5, alpha=0.7)
    ax1.plot(train_dates[1:], train_predicted_prices,
             label='Training Predictions', color='blue', linewidth=1.2, alpha=0.6)
    ax1.plot(test_dates, test_predicted_prices,
             label='Test Predictions (Out-of-Sample)', color='gold', linewidth=2)
    ax1.axvline(x=split_date, color='red', linestyle='--', linewidth=2.5,
                label=f'Train/Test Split ({split_date.date()})')
    ax1.axvspan(train_dates[0], train_dates[-1], alpha=0.1, color='blue')
    ax1.axvspan(test_dates[0], test_dates[-1], alpha=0.1, color='orange')

    ax1.set_title(f'Regime-Aware Model: Full Timeline | Test RMSE: {test_rmse:.4f} | Train RMSE: {train_rmse:.4f}',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Gold Price (USD)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(test_dates, test_actual_prices, label='Actual',
             color='black', linewidth=2, marker='o', markersize=2)
    ax2.plot(test_dates, test_predicted_prices, label='Predicted',
             color='gold', linewidth=2, marker='s', markersize=2)
    ax2.fill_between(test_dates, test_actual_prices, test_predicted_prices,
                      alpha=0.3, color='red')

    ax2.set_title(f'Test Set Performance | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Gold Price (USD)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    timeline_path = os.path.join(cfg.run_output_dir, 'final_model_with_split.png')
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {timeline_path}")

    fig2, ax = plt.subplots(1, 1, figsize=(14, 6))
    errors = test_actual_prices - test_predicted_prices
    ax.scatter(test_dates, errors, alpha=0.5, color='red', s=50)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=test_rmse, color='orange', linestyle='--', linewidth=1,
               label=f'+RMSE (+{test_rmse:.2f})')
    ax.axhline(y=-test_rmse, color='orange', linestyle='--', linewidth=1,
               label=f'-RMSE (-{test_rmse:.2f})')
    ax.set_title('Prediction Errors on Test Set', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Error (USD)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    residuals_path = os.path.join(cfg.run_output_dir, 'test_residuals.png')
    plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {residuals_path}")

    model_path = os.path.join(cfg.run_output_dir, 'regime_boost_model.joblib')
    dump(model, model_path)
    print(f"Model saved to {model_path}")

    metrics = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "test_split": cfg.test_split,
        "random_seed": cfg.random_seed,
        "gb_params": cfg.gb_params,
        "data_path": cfg.model_ready_path,
        "timestamp": datetime.now().isoformat(),
        "previous_best_rmse": PREVIOUS_BEST_RMSE,
    }
    save_metrics(cfg, metrics)

    importance_path = os.path.join(cfg.run_output_dir, 'feature_importance.csv')
    importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")

    print(f"\n{'=' * 70}")
    print("COMPLETE - RESULTS READY FOR SUBMISSION")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
