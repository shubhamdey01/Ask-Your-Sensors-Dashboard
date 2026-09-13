from pathlib import Path
import sqlite3
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sensor.db"


conn = sqlite3.connect(DB_PATH)


# ------------------------------------------------------------
# 1. Basic database information
# ------------------------------------------------------------

count = pd.read_sql_query(
    "SELECT COUNT(*) AS total_windows FROM sensor_windows",
    conn
)

print("\nTotal windows:")
print(count)


# ------------------------------------------------------------
# 2. Windows per room
# ------------------------------------------------------------

room_counts = pd.read_sql_query(
    """
    SELECT
        room_id,
        COUNT(*) AS windows
    FROM sensor_windows
    GROUP BY room_id
    ORDER BY room_id
    """,
    conn
)

print("\nWindows per room:")
print(room_counts.to_string(index=False))


# ------------------------------------------------------------
# 3. Show sample records
# ------------------------------------------------------------

sample = pd.read_sql_query(
    """
    SELECT *
    FROM sensor_windows
    WHERE room_id = '413'
    ORDER BY window_start
    LIMIT 10
    """,
    conn
)

print("\nSample records from room 413:")
print(sample.to_string(index=False))


# ------------------------------------------------------------
# 4. Check missing sensor values
# ------------------------------------------------------------

missing = pd.read_sql_query(
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
    conn
)

print("\nMissing sensor windows:")
print(missing.to_string(index=False))


# ------------------------------------------------------------
# 5. Check PIR activity
# ------------------------------------------------------------

pir = pd.read_sql_query(
    """
    SELECT
        MIN(pir_active_ratio) AS min_activity,
        MAX(pir_active_ratio) AS max_activity,
        AVG(pir_active_ratio) AS avg_activity
    FROM sensor_windows
    WHERE pir_active_ratio IS NOT NULL
    """,
    conn
)

print("\nPIR activity:")
print(pir.to_string(index=False))


conn.close()