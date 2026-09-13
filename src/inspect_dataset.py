from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_DIR = Path("data/raw/KETI")

SENSOR_FILES = {
    "temperature": "temperature.csv",
    "humidity": "humidity.csv",
    "co2": "co2.csv",
    "light": "light.csv",
    "pir": "pir.csv",
}


# --------------------------------------------------
# Analyze one sensor file
# --------------------------------------------------

def analyze_sensor_file(file_path):
    df = pd.read_csv(
        file_path,
        header=None,
        names=["timestamp", "value"]
    )

    # Remove invalid rows
    df = df.dropna(subset=["timestamp", "value"])

    # Convert types
    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df = df.dropna()

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Time differences
    time_diff = df["timestamp"].diff().dropna()

    # Basic statistics
    stats = {
        "rows": len(df),
        "start_timestamp": int(df["timestamp"].min()),
        "end_timestamp": int(df["timestamp"].max()),
        "unique_timestamps": df["timestamp"].nunique(),
        "duplicate_timestamps": int(
            df["timestamp"].duplicated().sum()
        ),
        "median_interval_sec": (
            float(time_diff.median())
            if len(time_diff) > 0
            else None
        ),
        "mean_interval_sec": (
            float(time_diff.mean())
            if len(time_diff) > 0
            else None
        ),
        "min_value": float(df["value"].min()),
        "max_value": float(df["value"].max()),
        "mean_value": float(df["value"].mean()),
        "zero_percentage": float(
            (df["value"] == 0).mean() * 100
        ),
    }

    return stats


# --------------------------------------------------
# Convert Unix timestamp
# --------------------------------------------------

def format_timestamp(timestamp):
    return datetime.fromtimestamp(
        timestamp
    ).strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if not DATASET_DIR.exists():
        print(f"Dataset directory not found:")
        print(DATASET_DIR)
        return

    room_dirs = sorted(
        [
            directory
            for directory in DATASET_DIR.iterdir()
            if directory.is_dir()
        ]
    )

    print("=" * 70)
    print("ASK THE SENSORS - DATASET INSPECTION")
    print("=" * 70)

    print(f"\nDataset: {DATASET_DIR}")
    print(f"Number of rooms: {len(room_dirs)}")

    total_readings = {
        sensor: 0
        for sensor in SENSOR_FILES
    }

    print("\n" + "=" * 70)
    print("ROOM DETAILS")
    print("=" * 70)

    for room_dir in room_dirs:

        room_id = room_dir.name

        print(f"\nROOM {room_id}")
        print("-" * 50)

        for sensor_type, filename in SENSOR_FILES.items():

            file_path = room_dir / filename

            if not file_path.exists():
                print(f"{sensor_type:15} FILE MISSING")
                continue

            stats = analyze_sensor_file(file_path)

            total_readings[sensor_type] += stats["rows"]

            start = format_timestamp(
                stats["start_timestamp"]
            )

            end = format_timestamp(
                stats["end_timestamp"]
            )

            print(f"\n{sensor_type}")

            print(
                f"  rows:                 {stats['rows']:,}"
            )

            print(
                f"  unique timestamps:    "
                f"{stats['unique_timestamps']:,}"
            )

            print(
                f"  duplicate timestamps: "
                f"{stats['duplicate_timestamps']:,}"
            )

            print(
                f"  start:                {start}"
            )

            print(
                f"  end:                  {end}"
            )

            print(
                f"  median interval:      "
                f"{stats['median_interval_sec']:.2f} sec"
            )

            print(
                f"  mean interval:        "
                f"{stats['mean_interval_sec']:.2f} sec"
            )

            print(
                f"  min value:            "
                f"{stats['min_value']:.4f}"
            )

            print(
                f"  max value:            "
                f"{stats['max_value']:.4f}"
            )

            print(
                f"  mean value:           "
                f"{stats['mean_value']:.4f}"
            )

            print(
                f"  zero percentage:      "
                f"{stats['zero_percentage']:.2f}%"
            )

    # --------------------------------------------------
    # Dataset summary
    # --------------------------------------------------

    print("\n\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"\nRooms: {len(room_dirs)}")

    print("\nTotal readings:")

    for sensor_type, count in total_readings.items():

        print(
            f"  {sensor_type:15}: {count:,}"
        )


if __name__ == "__main__":
    main()