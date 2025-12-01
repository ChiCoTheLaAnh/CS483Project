# SAHM vs Gold Price Analysis Project

This project analyzes the relationship between the SAHM (Smoothed Aggregate Hourly Minus) recession indicator and gold prices to identify periods of strong correlation.

## Overview

The SAHM recession indicator is an economic metric used to predict recessions. Gold is often considered a "safe haven" asset that tends to rise during economic uncertainty. This project investigates whether there are periods when these two variables move together, and identifies the strength and nature of their relationship.

## Files

### Data Files
- **`SAHMREALTIME.csv`** - SAHM recession indicator data with observation dates
- **`XAU_USD Historical Data.csv`** - Historical gold price data (USD per ounce)
- **`SAHM_vs_Gold_Monthly.csv`** - Generated output: merged and cleaned data with both SAHM and monthly average gold prices

### Python Scripts

#### `data_preparation.py`
The main data processing script that:
- Loads raw SAHM and gold price data
- Converts dates to datetime format and handles various CSV column naming conventions
- Cleans price data (removes commas, converts to numeric)
- Resamples daily gold prices to monthly averages
- Merges the two datasets on matching dates
- Calculates the Pearson correlation coefficient between SAHM and gold prices
- Saves the merged dataset to `SAHM_vs_Gold_Monthly.csv`

**Run with:** `python data_preparation.py`

#### `analyze_relationships.py`
Performs detailed correlation analysis:
- Finds "strong correlation intervals" using rolling window correlations (2-12 month windows)
- Identifies consecutive periods where correlation magnitude exceeds 0.5 threshold
- Calculates average, maximum, and minimum correlations for each interval
- Identifies the top 15 individual time windows with strongest absolute correlations
- Generates two output CSV files with results

**Output files:**
- `strong_intervals_summary.csv` - Summary of all strong correlation intervals
- `top_windows_summary.csv` - Top 15 windows by absolute correlation

**Run with:** `python analyze_relationships.py`

#### `plot_relationships.py`
Creates visualizations of the SAHM-Gold relationship:
- Plots four specific high-correlation periods with dual y-axes (one for SAHM, one for gold price)
- Generates a full timeline plot showing both variables across the entire dataset
- Uses color-coded lines and combined legends for clarity

**Output files:**
- `sahm_gold_relationships.png` - 4-panel figure showing key periods
- `sahm_gold_timeline.png` - Full timeline comparison

**Run with:** `python plot_relationships.py`

### Configuration
- **`requirements.txt`** - Python package dependencies (pandas >= 1.5, numpy)

## Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

1. **Start with `data_preparation.py`** - This must be run first to create the merged dataset
2. **Run `analyze_relationships.py`** - Analyze correlation patterns and generate summary statistics
3. **Run `plot_relationships.py`** - Visualize the relationships and key periods

## Key Parameters (in `analyze_relationships.py`)

- `THRESHOLD = 0.5` - Correlation threshold for "strong" correlation
- `MIN_WINDOW = 2` - Minimum rolling window size (months)
- `MAX_WINDOW = 12` - Maximum rolling window size (months)
- `TOP_N = 15` - Number of top correlation windows to report

Adjust these values to explore different correlation patterns

## Results Interpretation

- **Positive correlation**: SAHM and gold prices move together (both increase or decrease)
- **Negative correlation**: SAHM and gold prices move inversely
- **Correlation strength**: Values closer to 1 or -1 indicate stronger relationships
- **Intervals**: Periods where correlation magnitude stays above the threshold, suggesting sustained relationships

## Notes

- All prices are in USD
- Gold prices are aggregated to monthly averages for alignment with SAHM data frequency
- The analysis uses Pearson correlation coefficient, which measures linear relationships
- Missing data is removed during the merge process
