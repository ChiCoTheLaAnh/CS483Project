import os

import pandas as pd

from config import load_config


def load_master(cfg):
    print("Loading Master Dataset...")
    df = pd.read_csv(cfg.master_path, index_col=0, parse_dates=True)
    return df


def process_disasters(cfg, target_index):
    print("Processing Natural Disasters...")
    if not os.path.exists(cfg.disaster_path):
        print(f"Warning: {cfg.disaster_path} not found. Skipping.")
        return pd.DataFrame(index=target_index)

    df_dis = pd.read_csv(cfg.disaster_path)

    def make_date(row, prefix):
        try:
            y, m, d = int(row.get(f'{prefix} Year', 0)), int(row.get(f'{prefix} Month', 1)), int(row.get(f'{prefix} Day', 1))
            return pd.Timestamp(year=y, month=max(1, m), day=max(1, d))
        except Exception:
            return pd.NaT

    df_dis['Start_Date'] = df_dis.apply(lambda r: make_date(r, 'Start'), axis=1)
    df_dis['End_Date'] = df_dis.apply(lambda r: make_date(r, 'End'), axis=1)

    df_dis['End_Date'] = df_dis['End_Date'].fillna(df_dis['Start_Date'])
    df_dis.dropna(subset=['Start_Date'], inplace=True)

    disaster_flags = pd.Series(0, index=target_index, name='Disaster_Flag')

    for _, row in df_dis.iterrows():
        if pd.notna(row['Start_Date']) and pd.notna(row['End_Date']):
            mask = (disaster_flags.index >= row['Start_Date']) & (disaster_flags.index <= row['End_Date'])
            disaster_flags.loc[mask] = 1

    return disaster_flags.to_frame()


def process_epidemics(cfg, target_index):
    print("Processing Epidemics...")
    if not os.path.exists(cfg.epidemic_path):
        print(f"Warning: {cfg.epidemic_path} not found. Skipping.")
        return pd.DataFrame(index=target_index)

    df_epi = pd.read_csv(cfg.epidemic_path)
    df_epi['Date'] = pd.to_datetime(df_epi['Date'], errors='coerce')
    df_epi.dropna(subset=['Date'], inplace=True)

    epi_flags = pd.Series(0, index=target_index, name='Epidemic_Start_Flag')

    valid_dates = df_epi['Date'][df_epi['Date'].isin(target_index)]
    epi_flags.loc[valid_dates] = 1

    return epi_flags.to_frame()


def process_gdelt(cfg, target_index):
    print("Processing GDELT Data...")
    if not os.path.exists(cfg.gdelt_path):
        print(f"Warning: {cfg.gdelt_path} not found. Skipping.")
        return pd.DataFrame(index=target_index)

    df_gdelt = pd.read_csv(cfg.gdelt_path)

    date_col = [c for c in df_gdelt.columns if 'date' in c.lower()]
    if not date_col:
        print("Warning: No date column found in GDELT. Skipping.")
        return pd.DataFrame(index=target_index)

    df_gdelt[date_col[0]] = pd.to_datetime(df_gdelt[date_col[0]])
    df_gdelt.set_index(date_col[0], inplace=True)

    keep_cols = ['avg_tone', 'conflict_war_count', 'civil_unrest_protest_count']
    available_cols = [c for c in keep_cols if c in df_gdelt.columns]

    df_gdelt = df_gdelt[available_cols]

    df_gdelt = df_gdelt.reindex(target_index).fillna(0)

    return df_gdelt


def main(cfg=None):
    cfg = cfg or load_config()
    master_df = load_master(cfg)
    target_index = master_df.index

    disaster_df = process_disasters(cfg, target_index)
    epidemic_df = process_epidemics(cfg, target_index)
    gdelt_df = process_gdelt(cfg, target_index)

    print("Merging features...")
    final_df = pd.concat([master_df, disaster_df, epidemic_df, gdelt_df], axis=1)

    final_df.fillna(method='ffill', inplace=True)
    final_df.fillna(0, inplace=True)

    final_df.to_csv(cfg.model_ready_path)
    print("-" * 30)
    print("Feature Engineering Complete!")
    print(f"Saved to: {cfg.model_ready_path}")
    print(f"Final Shape: {final_df.shape}")
    print("Columns:", list(final_df.columns))
    print("-" * 30)

    print(f"Days with Active Disaster: {final_df['Disaster_Flag'].sum()}")
    print(f"Days with Epidemic Start:  {final_df['Epidemic_Start_Flag'].sum()}")


if __name__ == "__main__":
    main()
