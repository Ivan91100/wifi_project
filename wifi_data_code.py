import pandas as pd
import glob
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === CONFIGURATION ===
# Define the list of MAC addresses to exclude (computers not of interest)
EXCLUDE_MACS = [
    'dc:fb:48:68:be:e4',
    'dc:fb:48:8c:71:fc',
    'dc:fb:48:2a:52:e0',
    'dc:fb:48:f5:c6:c2',
    'dc:fb:48:e3:ab:78',
    'dc:fb:48:00:51:90',
    'dc:fb:48:75:d8:42',
    'dc:fb:48:de:86:8d',
    'dc:fb:48:dd:c6:0b',
    'dc:fb:48:55:d5:78',
    'dc:fb:48:c2:1d:64',
    '40:ec:99:f9:34:a6',
    '40:ec:99:8e:3f:63',
    '40:ec:99:1f:3e:75'
]

# === PARAMETERS ===
RSSI_MIN = -90  # Minimum acceptable RSSI
RSSI_MAX = 0     # Maximum acceptable RSSI
AGG_INTERVAL = '10min'  # Aggregation interval (you can change to '1H', '5min', etc.)
OUTPUT_FILE = r"C:\Users\ivane\OneDrive - ISEP\CSV_file\filtered_data.csv"

# STEP 1: Load, Clean, and Add Fingerprint

def load_and_clean_data(path, exclude_macs):
    """
    Loads and cleans CSV data.
    """
    # If path is a folder:
    csv_files = sorted(glob.glob(f"{path}/*.csv"))

    # If path is a direct file (uncomment this line and comment the one above if you have a single file)
    # csv_files = [path]

    if not csv_files:
        print("[!] No CSV files found. Check the path.")
        return None

    print(f"Loading {len(csv_files)} CSV file(s)...")
    df = pd.concat((pd.read_csv(file, sep=";", decimal=".") for file in csv_files), ignore_index=True)

    # Convert 'datetime' field
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Remove duplicates based on 'src' and 'datetime'
    df = df.drop_duplicates(subset=["src", "datetime"])

    # Filter RSSI (using configurable min and max values)
    df = df[(df["rssi"] >= RSSI_MIN) & (df["rssi"] <= RSSI_MAX)]

    # Normalize RSSI for potential future analyses
    df["rssi_norm"] = (df["rssi"] - df["rssi"].mean()) / df["rssi"].std()

    # Exclude specified MAC addresses
    df = df[~df["src"].isin(exclude_macs)]

    # Sort by date
    df = df.sort_values(by="datetime").reset_index(drop=True)

    print(f"✅ Cleaning complete: {len(df):,} valid entries.")
    return df


def compute_fingerprint(row):
    """
    Computes a fingerprint for each record by combining
    several fields to mitigate MAC address randomization.
    """
    parts = []
    for col in ['src_vendor', 'oui', 'randomized', 'ch_freq', 'seq_num', 'FCfield', 'dot11elt']:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]).strip())
    fingerprint_str = '_'.join(parts)
    fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    return fingerprint


def add_fingerprint(df):
    """
    Adds a 'fingerprint' column to the DataFrame.
    """
    df['fingerprint'] = df.apply(compute_fingerprint, axis=1)
    print("✅ Fingerprints added.")
    return df


# === STEP 2: Aggregate Data to Extract Features ===

def prepare_aggregated_features(df, interval=AGG_INTERVAL):
    """
    Aggregates data by time interval to extract useful features.
    """
    agg_df = df.groupby(pd.Grouper(key='datetime', freq=interval)).agg(
        unique_devices=('fingerprint', 'nunique'),
        total_requests=('fingerprint', 'count'),
        avg_rssi=('rssi', 'mean'),
        std_rssi=('rssi', 'std'),
        occupancy=('occupancy', 'mean')  # Assuming 'occupancy' column is present
    ).reset_index()

    agg_df = agg_df.dropna(subset=['occupancy'])
    print("✅ Aggregated features prepared.")
    return agg_df


# === Extra Plots ===

def plot_device_activity_over_time(df):
    """
    Plots the number of unique devices detected over time.
    """
    device_counts = df.groupby(pd.Grouper(key='datetime', freq='1H'))['fingerprint'].nunique()
    plt.figure(figsize=(12, 6))
    device_counts.plot()
    plt.xlabel('Time')
    plt.ylabel('Number of Unique Devices')
    plt.title('Unique Devices Detected Over Time')
    plt.grid(True)
    plt.show()


def plot_rssi_distribution(df):
    """
    Plots the distribution of RSSI values.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df['rssi'], bins=50, kde=True)
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Frequency')
    plt.title('Distribution of RSSI Values')
    plt.grid(True)
    plt.show()


# === MAIN ===

if __name__ == "__main__":
    # === Define your path ===

    # Example: Folder path (if you have multiple CSV files)
    path = r"C:\Users\ivane\OneDrive - ISEP\CSV_file"

    # Example: Single file path
    # path = r"C:\Users\ivane\OneDrive\Bureau\CSV_file\your_file.csv"

    # === Run processing pipeline ===
    df_clean = load_and_clean_data(path, EXCLUDE_MACS)

    if df_clean is not None:
        df_fp = add_fingerprint(df_clean)

        # Plot device activity over time
        plot_device_activity_over_time(df_fp)

        # Plot RSSI distribution
        plot_rssi_distribution(df_fp)

        # Aggregate features
        agg_df = prepare_aggregated_features(df_fp)

        # Print the first few rows of aggregated features
        print("\nAggregated Features (sample):")
        print(agg_df.head())

        # ✅ Export the filtered data (MAC excluded and with fingerprint)
        df_fp.to_csv(OUTPUT_FILE, index=False, sep=";", decimal=".")
        print(f"✅ Filtered data saved to {OUTPUT_FILE}")

    else:
        print("❌ No data loaded. Exiting.")
