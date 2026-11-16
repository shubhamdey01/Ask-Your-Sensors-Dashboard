import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sensor.db"


def query_database(conn, query):
    """Run a SQL query and return the result as a DataFrame."""
    return pd.read_sql_query(query, conn)


def main():
    """Run basic validation checks on the sensor database."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    with sqlite3.connect(DB_PATH) as conn:

        count = query_database(
            conn,
            """
            SELECT COUNT(*) AS total_windows
            FROM sensor_windows
            """,
        )

        print("\nTotal windows:")
        print(count)

        room_counts = query_database(
            conn,
            """
            SELECT
                room_id,
                COUNT(*) AS windows
            FROM sensor_windows
            GROUP BY room_id
            ORDER BY room_id
            """,
        )

        print("\nWindows per room:")
        print(room_counts.to_string(index=False))

        sample = query_database(
            conn,
            """
            SELECT *
            FROM sensor_windows
            WHERE room_id = '413'
            ORDER BY window_start
            LIMIT 10
            """,
        )

        print("\nSample records from room 413:")
        print(sample.to_string(index=False))

        missing = query_database(
            conn,
            """
            SELECT
                COUNT(*) AS total,

                SUM(
                    CASE WHEN temperature_mean IS NULL
                    THEN 1 ELSE 0 END
                ) AS temperature_missing,

                SUM(
                    CASE WHEN humidity_mean IS NULL
                    THEN 1 ELSE 0 END
                ) AS humidity_missing,

                SUM(
                    CASE WHEN co2_mean IS NULL
                    THEN 1 ELSE 0 END
                ) AS co2_missing,

                SUM(
                    CASE WHEN light_mean IS NULL
                    THEN 1 ELSE 0 END
                ) AS light_missing,

                SUM(
                    CASE WHEN pir_mean IS NULL
                    THEN 1 ELSE 0 END
                ) AS pir_missing

            FROM sensor_windows
            """,
        )

        print("\nMissing sensor windows:")
        print(missing.to_string(index=False))

        pir = query_database(
            conn,
            """
            SELECT
                MIN(pir_active_ratio) AS min_activity,
                MAX(pir_active_ratio) AS max_activity,
                AVG(pir_active_ratio) AS avg_activity
            FROM sensor_windows
            WHERE pir_active_ratio IS NOT NULL
            """,
        )

        print("\nPIR activity:")
        print(pir.to_string(index=False))


if __name__ == "__main__":
    main()