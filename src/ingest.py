from pathlib import Path
import sqlite3
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "KETI"
DB_PATH = PROJECT_ROOT / "data" / "sensor.db"


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = "5min"

SENSORS = [
    "temperature",
    "humidity",
    "co2",
    "light",
    "pir",
]


# ============================================================
# DATABASE
# ============================================================

def create_database(conn):
    """Create the SQLite table for aggregated sensor windows."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_windows (
            window_id INTEGER PRIMARY KEY AUTOINCREMENT,

            room_id TEXT NOT NULL,
            window_start TEXT NOT NULL,
            window_end TEXT NOT NULL,

            temperature_mean REAL,
            temperature_min REAL,
            temperature_max REAL,
            temperature_std REAL,
            temperature_count INTEGER,

            humidity_mean REAL,
            humidity_min REAL,
            humidity_max REAL,
            humidity_std REAL,
            humidity_count INTEGER,

            co2_mean REAL,
            co2_min REAL,
            co2_max REAL,
            co2_std REAL,
            co2_count INTEGER,

            light_mean REAL,
            light_min REAL,
            light_max REAL,
            light_std REAL,
            light_count INTEGER,

            pir_mean REAL,
            pir_min REAL,
            pir_max REAL,
            pir_std REAL,
            pir_active_count INTEGER,
            pir_active_ratio REAL,
            pir_count INTEGER
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sensor_windows_room_time
        ON sensor_windows(room_id, window_start)
    """)

    conn.commit()


# ============================================================
# SENSOR LOADING
# ============================================================

def load_sensor_csv(room_dir, sensor):
    """
    Load one sensor CSV.

    CSV format:
        Unix timestamp, value

    There is no header in the original dataset.
    """

    file_path = room_dir / f"{sensor}.csv"

    if not file_path.exists():
        return None

    df = pd.read_csv(
        file_path,
        header=None,
        names=["timestamp", "value"]
    )

    # Convert Unix timestamp to datetime
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    ).dt.tz_convert(
        "America/Los_Angeles"
    ).dt.tz_localize(None)

    # Make sure values are numeric
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=["timestamp", "value"]
    )

    return df


# ============================================================
# SENSOR AGGREGATION
# ============================================================

def aggregate_sensor(df, sensor):
    """
    Aggregate one sensor into 5-minute windows.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.set_index("timestamp")

    grouped = df["value"].resample(WINDOW_SIZE)

    result = pd.DataFrame(index=grouped.mean().index)

    result[f"{sensor}_mean"] = grouped.mean()
    result[f"{sensor}_min"] = grouped.min()
    result[f"{sensor}_max"] = grouped.max()
    result[f"{sensor}_std"] = grouped.std()
    result[f"{sensor}_count"] = grouped.count()

    # PIR needs some additional information
    if sensor == "pir":

        result["pir_active_count"] = grouped.apply(
            lambda x: (x > 0).sum()
        )

        result["pir_active_ratio"] = grouped.apply(
            lambda x: (x > 0).mean()
        )

    return result


# ============================================================
# ROOM PROCESSING
# ============================================================

def process_room(room_dir):
    """
    Process all five sensors for one room.
    """

    room_id = room_dir.name

    print(f"Processing room {room_id}...")

    sensor_results = []

    for sensor in SENSORS:

        df = load_sensor_csv(room_dir, sensor)

        if df is None:
            print(f"  WARNING: {sensor}.csv not found")
            continue

        aggregated = aggregate_sensor(
            df,
            sensor
        )

        sensor_results.append(aggregated)

    if not sensor_results:
        return pd.DataFrame()

    # Combine sensors by timestamp.
    # Outer join preserves windows where only some sensors
    # have readings.
    result = pd.concat(
        sensor_results,
        axis=1,
        join="outer"
    )

    result = result.reset_index()

    result["room_id"] = room_id

    # Window end = start + 5 minutes
    result["window_end"] = (
        result["timestamp"]
        + pd.Timedelta(minutes=5)
    )

    result = result.rename(
        columns={
            "timestamp": "window_start"
        }
    )

    # Put identifiers first
    identifier_columns = [
        "room_id",
        "window_start",
        "window_end",
    ]

    other_columns = [
        col for col in result.columns
        if col not in identifier_columns
    ]

    result = result[
        identifier_columns + other_columns
    ]

    return result


# ============================================================
# DATABASE INSERT
# ============================================================

def insert_room_data(conn, df):
    """Insert aggregated room data into SQLite."""

    if df.empty:
        return 0

    df = df.copy()

    # SQLite stores datetime values as text
    df["window_start"] = df["window_start"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    df["window_end"] = df["window_end"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # SQLite expects NULL instead of pandas NaN
    df = df.where(pd.notna(df), None)

    columns = list(df.columns)

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    column_names = ", ".join(columns)

    sql = f"""
        INSERT INTO sensor_windows
        ({column_names})
        VALUES ({placeholders})
    """

    values = [
        tuple(row)
        for row in df.itertuples(index=False, name=None)
    ]

    conn.executemany(sql, values)
    conn.commit()

    return len(values)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("KETI SENSOR DATA INGESTION")
    print("=" * 60)

    print(f"Raw data: {RAW_DATA_DIR}")
    print(f"Database: {DB_PATH}")
    print(f"Window:   {WINDOW_SIZE}")
    print()

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"KETI dataset not found: {RAW_DATA_DIR}"
        )

    # Create data directory if necessary
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(DB_PATH)

    create_database(conn)

    # Find room directories
    room_dirs = sorted(
        [
            p for p in RAW_DATA_DIR.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name
    )

    print(f"Rooms found: {len(room_dirs)}")
    print()

    total_rows = 0

    for room_dir in room_dirs:

        room_data = process_room(room_dir)

        inserted = insert_room_data(
            conn,
            room_data
        )

        total_rows += inserted

        print(
            f"  → {inserted:,} windows inserted"
        )

    # Final statistics
    cursor = conn.execute(
        "SELECT COUNT(*) FROM sensor_windows"
    )

    database_rows = cursor.fetchone()[0]

    conn.close()

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(f"Rooms processed:  {len(room_dirs)}")
    print(f"Rows inserted:    {total_rows:,}")
    print(f"Rows in database: {database_rows:,}")
    print(f"Database:         {DB_PATH}")


if __name__ == "__main__":
    main()