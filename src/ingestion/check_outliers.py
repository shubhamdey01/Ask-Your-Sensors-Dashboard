import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sensor.db"

LIMIT = 20


def query_database(conn, query):
    """Run a SQL query and return the result as a DataFrame."""
    return pd.read_sql_query(query, conn, params=(LIMIT,))


def print_section(title, data):
    """Print a formatted diagnostic section."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(data.to_string(index=False))


def main():
    """Display extreme temperature and CO2 windows."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    queries = {
        "EXTREME TEMPERATURE WINDOWS": """
            SELECT
                room_id,
                window_start,
                temperature_mean,
                temperature_min,
                temperature_max,
                temperature_count
            FROM sensor_windows
            WHERE temperature_mean IS NOT NULL
            ORDER BY temperature_mean DESC
            LIMIT ?
        """,
        "LOW TEMPERATURE WINDOWS": """
            SELECT
                room_id,
                window_start,
                temperature_mean,
                temperature_min,
                temperature_max,
                temperature_count
            FROM sensor_windows
            WHERE temperature_mean IS NOT NULL
            ORDER BY temperature_mean ASC
            LIMIT ?
        """,
        "EXTREME CO2 WINDOWS": """
            SELECT
                room_id,
                window_start,
                co2_mean,
                co2_min,
                co2_max,
                co2_count
            FROM sensor_windows
            WHERE co2_mean IS NOT NULL
            ORDER BY co2_mean DESC
            LIMIT ?
        """,
        "LOW CO2 WINDOWS": """
            SELECT
                room_id,
                window_start,
                co2_mean,
                co2_min,
                co2_max,
                co2_count
            FROM sensor_windows
            WHERE co2_mean IS NOT NULL
            ORDER BY co2_mean ASC
            LIMIT ?
        """,
    }

    with sqlite3.connect(DB_PATH) as conn:
        for title, query in queries.items():
            data = query_database(conn, query)
            print_section(title, data)


if __name__ == "__main__":
    main()