# Description: Main Data Processing Script
# Loads SAHM and Gold data, and integrates global event datasets 
# (GDELT, epidemics/pandemics, and natural disasters). 
# Processes it, merges it, computes correlation, and saves the final dataset.
# Inputs: SAHMREALTIME.csv, XAU_USD Historical Data.csv, gdelt_daily_world_2013_present.csv
# epidemic_and_pandemics.csv, natural_disasters.csv
# Outputs: SAHM_vs_Gold_with_Events_Monthly.csv

import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path

def daily_to_monthly_features(csv_path, keep_cols=None):
    df = pd.read_csv(csv_path)

    # detect date column
    date_col = None
    for c in df.columns:
        if 'date' in c.lower():
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column found in {csv_path}")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # filter to only the needed columns
    if keep_cols is not None:
        df = df[keep_cols]

    # aggregate
    monthly = pd.DataFrame()
    for col in df.columns:
        series = df[col].dropna()

        # 0/1 flags → max
        unique_vals = set(series.unique())
        if unique_vals.issubset({0, 1}):
            monthly[col] = df[col].resample("MS").max()
        else:
            # if it's integer → sum, else mean
            if pd.api.types.is_integer_dtype(series):
                monthly[col] = df[col].resample("MS").sum()
            else:
                monthly[col] = df[col].resample("MS").mean()

    return monthly

def build_pandemic_monthly(epi_path, monthly_index):
    epi = pd.read_csv(epi_path)

    # parse Date column
    epi["Date"] = pd.to_datetime(epi["Date"], errors="coerce")
    epi = epi.dropna(subset=["Date"])

    # convert to month start
    epi["month"] = epi["Date"].dt.to_period("M").dt.to_timestamp()

    # start with all zeros aligned to your monthly index
    epi_monthly = pd.DataFrame(index=monthly_index.copy())
    epi_monthly["pandemic_event"] = 0
    epi_monthly["covid_event"] = 0

    for _, row in epi.iterrows():
        m = row["month"]
        if m in epi_monthly.index:
            epi_monthly.loc[m, "pandemic_event"] = 1
            disease = str(row.get("Disease", "")).lower()
            if "covid" in disease:
                epi_monthly.loc[m, "covid_event"] = 1

    return epi_monthly

import math

def _build_date_from_ymd(row, prefix):
    y = row.get(f"{prefix} Year")
    m = row.get(f"{prefix} Month")
    d = row.get(f"{prefix} Day")

    try:
        y = int(y)
    except Exception:
        return pd.NaT

    try:
        m = int(m)
        if not (1 <= m <= 12):
            m = 1
    except Exception:
        m = 1

    try:
        d = int(d)
        if not (1 <= d <= 31):
            d = 1
    except Exception:
        d = 1

    try:
        return pd.Timestamp(y, m, d)
    except Exception:
        return pd.NaT


def build_disaster_monthly(nd_path, monthly_index):
    """
    Build monthly disaster flags from EM-DAT-style file with:
    Start Year/Month/Day, End Year/Month/Day, Disaster Type, Country/ISO, etc.
    Output columns:
      - disaster_any
      - disaster_flood
      - disaster_storm
      - disaster_wildfire
      - disaster_eq
      - disaster_drought
    """
    nd = pd.read_csv(nd_path)

    nd["start"] = nd.apply(lambda r: _build_date_from_ymd(r, "Start"), axis=1)
    nd["end"]   = nd.apply(lambda r: _build_date_from_ymd(r, "End"), axis=1)
    nd = nd.dropna(subset=["start"])

    # prepare empty monthly frame
    dm = pd.DataFrame(index=monthly_index.copy())
    dm["disaster_any"] = 0
    dm["disaster_flood"] = 0
    dm["disaster_storm"] = 0
    dm["disaster_wildfire"] = 0
    dm["disaster_eq"] = 0
    dm["disaster_drought"] = 0

    for _, row in nd.iterrows():
        s = row["start"]
        e = row["end"] if not pd.isna(row["end"]) else s

        # convert start/end to month starts
        s_m = s.to_period("M").to_timestamp()
        e_m = e.to_period("M").to_timestamp()

        # mask for months within [s_m, e_m]
        mask = (dm.index >= s_m) & (dm.index <= e_m)
        if not mask.any():
            continue

        dm.loc[mask, "disaster_any"] = 1

        dtype = str(row.get("Disaster Type", "")).lower()
        if "flood" in dtype:
            dm.loc[mask, "disaster_flood"] = 1
        if "storm" in dtype or "hurricane" in dtype or "cyclone" in dtype or "typhoon" in dtype:
            dm.loc[mask, "disaster_storm"] = 1
        if "wildfire" in dtype or "fire" in dtype:
            dm.loc[mask, "disaster_wildfire"] = 1
        if "earthquake" in dtype:
            dm.loc[mask, "disaster_eq"] = 1
        if "drought" in dtype:
            dm.loc[mask, "disaster_drought"] = 1

    return dm


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

        monthly_index = gold_monthly_avg.index

        gdelt_path = data_dir / "gdelt_daily_world_2013_present.csv"
        epi_path   = data_dir / "epidemic_and_pandemics.csv"
        nd_path    = data_dir / "natural_disasters.csv"

        print("Loading and aggregating GDELT daily data...")
        gdelt_monthly = daily_to_monthly_features(
            gdelt_path,
            keep_cols=[
                "total_events",
                "conflict_war_count",
                "econ_policy_count",
                "government_political_count",
                "civil_unrest_protest_count",
                "diplomatic_relations_count",
                "avg_goldstein_scale",
                "avg_tone",
            ]
        )
        print(gdelt_monthly.head())
        print("-" * 30)

        print("Building epidemics/pandemics monthly flags...")
        epidemics_monthly = build_pandemic_monthly(epi_path, monthly_index)
        print(epidemics_monthly.head())
        print("-" * 30)

        print("Building natural disaster monthly flags...")
        disasters_monthly = build_disaster_monthly(nd_path, monthly_index)
        print(disasters_monthly.head())
        print("-" * 30)


        # --- 4. Merge DataFrames ---
        print("Merging SAHM and Monthly Gold data...")
        
        # Join the two dataframes on their index (the date)
        # 'how='inner'' ensures we only keep dates where *both* datasets have data
        merged_df = sahm_df.join(gold_monthly_avg, how='inner')

        # join monthly GDELT, epidemics+pandemics, and disasters
        merged_df = merged_df.join(gdelt_monthly, how='left')
        merged_df = merged_df.join(epidemics_monthly, how='left')
        merged_df = merged_df.join(disasters_monthly, how='left')
        
        # fill missing event features with 0, keep rows with both SAHM + Gold
        core_cols = ['SAHMREALTIME', 'Gold_Price_Monthly_Avg']
        event_cols = [c for c in merged_df.columns if c not in core_cols]

        merged_df[event_cols] = merged_df[event_cols].fillna(0)
        merged_df = merged_df.dropna(subset=core_cols)
        
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
        output_filename = "SAHM_vs_Gold_with_Events_Monthly.csv"
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