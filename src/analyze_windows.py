from pathlib import Path
import sqlite3
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sensor.db"

conn = sqlite3.connect(DB_PATH)


# ------------------------------------------------------------
# Basic sensor distributions
# ------------------------------------------------------------

query = """
SELECT
    temperature_mean,
    humidity_mean,
    co2_mean,
    light_mean,
    pir_active_ratio
FROM sensor_windows
"""

df = pd.read_sql_query(query, conn)

conn.close()


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
                0.99
            ]
        )
    )