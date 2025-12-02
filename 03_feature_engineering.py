import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
DATA_DIR = 'data'
MASTER_FILE = os.path.join(DATA_DIR, 'master_dataset.csv')
DISASTER_FILE = os.path.join(DATA_DIR, 'natural_disasters.csv')
EPIDEMIC_FILE = os.path.join(DATA_DIR, 'epidemic_and_pandemics.csv')
GDELT_FILE = os.path.join(DATA_DIR, 'gdelt_daily_world_2013_present.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'model_ready_dataset.csv')

def load_master():
    print("Loading Master Dataset...")
    df = pd.read_csv(MASTER_FILE, index_col=0, parse_dates=True)
    return df

def process_disasters(target_index):
    """Creates a daily binary flag for active disasters."""
    print("Processing Natural Disasters...")
    if not os.path.exists(DISASTER_FILE):
        print(f"Warning: {DISASTER_FILE} not found. Skipping.")
        return pd.DataFrame(index=target_index)

    df_dis = pd.read_csv(DISASTER_FILE)
    
    # Helper to clean dates
    def make_date(row, prefix):
        try:
            y, m, d = int(row.get(f'{prefix} Year', 0)), int(row.get(f'{prefix} Month', 1)), int(row.get(f'{prefix} Day', 1))
            return pd.Timestamp(year=y, month=max(1, m), day=max(1, d))
        except:
            return pd.NaT

    df_dis['Start_Date'] = df_dis.apply(lambda r: make_date(r, 'Start'), axis=1)
    df_dis['End_Date'] = df_dis.apply(lambda r: make_date(r, 'End'), axis=1)
    
    # Fill missing end dates with start date (1-day disaster)
    df_dis['End_Date'] = df_dis['End_Date'].fillna(df_dis['Start_Date'])
    df_dis.dropna(subset=['Start_Date'], inplace=True)

    # Create Daily Flags
    # We create a Series of 0s indexed by our master dates
    disaster_flags = pd.Series(0, index=target_index, name='Disaster_Flag')

    for _, row in df_dis.iterrows():
        # Set value to 1 for all days between Start and End
        if pd.notna(row['Start_Date']) and pd.notna(row['End_Date']):
            # Clip dates to ensure we don't try to write outside our master index range
            # (handling this logic via intersection)
            mask = (disaster_flags.index >= row['Start_Date']) & (disaster_flags.index <= row['End_Date'])
            disaster_flags.loc[mask] = 1

    return disaster_flags.to_frame()

def process_epidemics(target_index):
    """Creates a daily flag for epidemic starts."""
    print("Processing Epidemics...")
    if not os.path.exists(EPIDEMIC_FILE):
        print(f"Warning: {EPIDEMIC_FILE} not found. Skipping.")
        return pd.DataFrame(index=target_index)

    df_epi = pd.read_csv(EPIDEMIC_FILE)
    df_epi['Date'] = pd.to_datetime(df_epi['Date'], errors='coerce')
    df_epi.dropna(subset=['Date'], inplace=True)

    # Since epidemics are events, we might just flag the start date
    # Or create a 'Pandemic_Active' flag if we had end dates. 
    # For now, let's flag the start day.
    epi_flags = pd.Series(0, index=target_index, name='Epidemic_Start_Flag')
    
    # Mark 1 on days where an epidemic started
    valid_dates = df_epi['Date'][df_epi['Date'].isin(target_index)]
    epi_flags.loc[valid_dates] = 1
    
    return epi_flags.to_frame()

def process_gdelt(target_index):
    """Merges GDELT sentiment data."""
    print("Processing GDELT Data...")
    if not os.path.exists(GDELT_FILE):
        print(f"Warning: {GDELT_FILE} not found. Skipping.")
        return pd.DataFrame(index=target_index)

    df_gdelt = pd.read_csv(GDELT_FILE)
    
    # Find the date column (it might be 'SQLDATE', 'Date', or similar)
    date_col = [c for c in df_gdelt.columns if 'date' in c.lower()]
    if not date_col:
        print("Warning: No date column found in GDELT. Skipping.")
        return pd.DataFrame(index=target_index)
        
    df_gdelt[date_col[0]] = pd.to_datetime(df_gdelt[date_col[0]])
    df_gdelt.set_index(date_col[0], inplace=True)
    
    # Keep useful numeric columns (Sentiment, Conflict counts)
    keep_cols = ['avg_tone', 'conflict_war_count', 'civil_unrest_protest_count']
    available_cols = [c for c in keep_cols if c in df_gdelt.columns]
    
    df_gdelt = df_gdelt[available_cols]
    
    # GDELT might have missing days; reindex to match master
    df_gdelt = df_gdelt.reindex(target_index).fillna(0) # Fill missing days with 0 activity
    
    return df_gdelt

def main():
    # 1. Load Base Data
    master_df = load_master()
    target_index = master_df.index
    
    # 2. Generate Features
    disaster_df = process_disasters(target_index)
    epidemic_df = process_epidemics(target_index)
    gdelt_df = process_gdelt(target_index)
    
    # 3. Merge All
    print("Merging features...")
    final_df = pd.concat([master_df, disaster_df, epidemic_df, gdelt_df], axis=1)
    
    # 4. Final Cleanup
    # Fill any remaining NaNs (e.g. from macro gaps) with 0 or forward fill
    final_df.fillna(method='ffill', inplace=True) # Forward fill prices/macro
    final_df.fillna(0, inplace=True) # Fill events with 0
    
    # 5. Save
    final_df.to_csv(OUTPUT_FILE)
    print("-" * 30)
    print(f"Feature Engineering Complete!")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Final Shape: {final_df.shape}")
    print("Columns:", list(final_df.columns))
    print("-" * 30)
    
    # Quick Stat Check
    print(f"Days with Active Disaster: {final_df['Disaster_Flag'].sum()}")
    print(f"Days with Epidemic Start:  {final_df['Epidemic_Start_Flag'].sum()}")

if __name__ == "__main__":
    main()