from pathlib import Path
import sqlite3
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sensor.db"

conn = sqlite3.connect(DB_PATH)


# ------------------------------------------------------------
# Temperature extremes
# ------------------------------------------------------------

print("=" * 70)
print("EXTREME TEMPERATURE WINDOWS")
print("=" * 70)

temperature = pd.read_sql_query(
    """
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
    LIMIT 20
    """,
    conn
)

print(temperature.to_string(index=False))


# ------------------------------------------------------------
# Very low temperature
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LOW TEMPERATURE WINDOWS")
print("=" * 70)

temperature_low = pd.read_sql_query(
    """
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
    LIMIT 20
    """,
    conn
)

print(temperature_low.to_string(index=False))


# ------------------------------------------------------------
# CO2 extremes
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXTREME CO2 WINDOWS")
print("=" * 70)

co2 = pd.read_sql_query(
    """
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
    LIMIT 20
    """,
    conn
)

print(co2.to_string(index=False))


# ------------------------------------------------------------
# Very low CO2
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LOW CO2 WINDOWS")
print("=" * 70)

co2_low = pd.read_sql_query(
    """
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
    LIMIT 20
    """,
    conn
)

print(co2_low.to_string(index=False))


conn.close()