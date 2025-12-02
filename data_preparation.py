# Description: Main Data Processing Script
# Loads SAHM and Gold data, processes it, merges it, computes correlation, and saves the final dataset.
# Inputs: SAHMREALTIME.csv, XAU_USD Historical Data.csv
# Outputs: SAHM_vs_Gold_Monthly.csv

import pandas as pd
from pathlib import Path


def process_and_analyze_data():
    try:
        script_dir = Path(__file__).resolve().parent # load SAHM REALTIME data
        data_dir = script_dir / 'data'
        if not data_dir.exists():
            data_dir = script_dir

        sahm_df = pd.read_csv(data_dir / "SAHMREALTIME.csv")
        sahm_df['observation_date'] = pd.to_datetime(sahm_df['observation_date'], format='%Y-%m-%d') # convert to datetime
        
        sahm_df = sahm_df.set_index('observation_date') # set data as the index
        print(sahm_df.head())
        print("-" * 30)

        gold_candidates = [data_dir / "XAU_USD Historical Data.csv"]
        gold_path = None
        for p in gold_candidates:
            if p.exists():
                gold_path = p
                break

        gold_df = pd.read_csv(gold_path, quotechar='"', infer_datetime_format=True)

        # Normalize: either ('Date','Price') or ('Date','Value')
        if 'Price' in gold_df.columns and 'Date' in gold_df.columns:
            gold_df = gold_df[['Date', 'Price']]
            date_col = 'Date'
            price_col = 'Price'
        elif 'Value' in gold_df.columns and 'Date' in gold_df.columns:
            gold_df = gold_df[['Date', 'Value']]
            date_col = 'Date'
            price_col = 'Value'
            gold_df = gold_df.rename(columns={price_col: 'Price'})
            price_col = 'Price'
        else:
            date_col = None
            for c in gold_df.columns:
                if 'date' in c.lower():
                    date_col = c
                    break
            price_col = [c for c in gold_df.columns if c != date_col][0]
            gold_df = gold_df[[date_col, price_col]]
            gold_df = gold_df.rename(columns={date_col: 'Date', price_col: 'Price'})

        gold_df['Date'] = pd.to_datetime(gold_df['Date'], infer_datetime_format=True, errors='coerce')
        gold_df = gold_df.set_index('Date')
        
        gold_df = gold_df.sort_index(ascending=True)
        
        print("Raw gold data loaded and sorted.")
        print(gold_df.head())
        print("-" * 30)
        
        # The 'Price' column may be a string with commas (e.g., "1,301.38") or already numeric.
        if gold_df['Price'].dtype == object:
            gold_df['Price'] = gold_df['Price'].str.replace(',', '', regex=False)
        gold_df['Price'] = pd.to_numeric(gold_df['Price'], errors='coerce')
        gold_df = gold_df.dropna(subset=['Price'])
        
        # Resample the daily data into monthly averages
        gold_monthly_avg = gold_df['Price'].resample('MS').mean().to_frame()
        gold_monthly_avg.columns = ['Gold_Price_Monthly_Avg']
        
        print("Gold data aggregated to monthly averages:")
        print(gold_monthly_avg.head())
        print("-" * 30)

        # --- 4. Merge DataFrames ---
        print("Merging SAHM and Monthly Gold data...")
        
        # Join the two dataframes on their index (the date)
        # 'how='inner'' ensures we only keep dates where *both* datasets have data
        merged_df = sahm_df.join(gold_monthly_avg, how='inner')
        
        # Drop any rows that might have missing values after the merge
        merged_df = merged_df.dropna()
        
        print("Final Merged Data:")
        print(merged_df.head())
        print("-" * 30)

        # --- 5. Calculate Correlation ---
        print("Calculating correlation...")
        
        # Calculate the Pearson correlation coefficient
        correlation = merged_df['SAHMREALTIME'].corr(merged_df['Gold_Price_Monthly_Avg'])
        
        print("\n" + "=" * 30)
        print(f"Correlation between SAHMREALTIME and Gold_Price_Monthly_Avg: {correlation:.4f}")
        print("=" * 30 + "\n")

        # --- 6. Save Final Data ---
        output_filename = "SAHM_vs_Gold_Monthly.csv"
        merged_df.to_csv(output_filename)
        
        print(f"Successfully saved the final merged data to: {output_filename}")

    except FileNotFoundError as e:
        print(f"ERROR: File not found.")
        print(f"Please make sure '{e.filename}' is in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the file formats and contents.")

if __name__ == "__main__":
    process_and_analyze_data()