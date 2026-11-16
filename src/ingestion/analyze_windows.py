import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sensor.db"


def load_sensor_windows():
    """Load sensor-window statistics from SQLite."""

    query = """
        SELECT
            temperature_mean,
            humidity_mean,
            co2_mean,
            light_mean,
            pir_active_ratio
        FROM sensor_windows
    """

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


def main():
    """Print descriptive statistics for sensor windows."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    df = load_sensor_windows()

    print("=" * 60)
    print("SENSOR WINDOW DISTRIBUTIONS")
    print("=" * 60)

    for column in df.columns:
        print(f"\n{column}")
        print(
            df[column].describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ]
            )
        )


if __name__ == "__main__":
    main()