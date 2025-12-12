import os
import time
from datetime import datetime


def run_script(script_name):
    print(f"\n{'='*50}")
    print(f">>> RUNNING: {script_name}")
    print(f"{'='*50}")
    result = os.system(f"python {script_name}")
    if result != 0:
        print(f"ERROR: {script_name} failed. Stopping execution.")
        exit(1)
    time.sleep(1)


def main():
    print("--- STARTING FINAL PROJECT PIPELINE ---")

    run_id = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.environ["RUN_ID"] = run_id
    print(f"RUN_ID set to {run_id}. Outputs will be grouped under outputs/{run_id}/")

    run_script("01_data_pipeline.py")
    run_script("02_baseline_random_walk.py")
    run_script("02_2_statistical_diagnosis.py")
    run_script("03_feature_engineering.py")
    run_script("09_final_regime_boost.py")

    print("\n" + "="*50)
    print("PROJECT COMPLETE.")
    print("All results and charts have been saved to this folder.")
    print("="*50)


if __name__ == "__main__":
    main()
