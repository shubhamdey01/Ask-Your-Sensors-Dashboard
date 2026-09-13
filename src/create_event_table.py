import sqlite3
import pandas as pd


DB_PATH = "./data/sensor.db"
EVENTS_FILE = "./data/semantic_events.csv"


def main():

    print("=" * 70)
    print("CREATING SEMANTIC EVENT TABLE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load semantic events
    # --------------------------------------------------------

    print("\nLoading semantic events...")

    events = pd.read_csv(EVENTS_FILE)

    print(f"Events loaded: {len(events):,}")

    # --------------------------------------------------------
    # Open SQLite
    # --------------------------------------------------------

    conn = sqlite3.connect(DB_PATH)

    # --------------------------------------------------------
    # Create table
    # --------------------------------------------------------

    conn.execute("""
        DROP TABLE IF EXISTS semantic_events
    """)

    conn.execute("""
        CREATE TABLE semantic_events (

            event_id INTEGER PRIMARY KEY,

            room_id TEXT NOT NULL,

            event_start TEXT NOT NULL,

            event_end TEXT NOT NULL,

            window_count INTEGER,

            temperature_mean REAL,
            temperature_min REAL,
            temperature_max REAL,

            humidity_mean REAL,
            humidity_min REAL,
            humidity_max REAL,

            co2_mean REAL,
            co2_min REAL,
            co2_max REAL,

            light_mean REAL,
            light_min REAL,
            light_max REAL,

            pir_mean REAL,
            pir_min REAL,
            pir_max REAL,

            pir_active_ratio REAL,

            reasons TEXT
        )
    """)

    # --------------------------------------------------------
    # Insert events
    # --------------------------------------------------------

    columns = [
        "event_id",
        "room_id",
        "event_start",
        "event_end",
        "window_count",

        "temperature_mean",
        "temperature_min",
        "temperature_max",

        "humidity_mean",
        "humidity_min",
        "humidity_max",

        "co2_mean",
        "co2_min",
        "co2_max",

        "light_mean",
        "light_min",
        "light_max",

        "pir_mean",
        "pir_min",
        "pir_max",

        "pir_active_ratio",

        "reasons"
    ]

    # Keep only columns that exist in the CSV
    available_columns = [
        column
        for column in columns
        if column in events.columns
    ]

    events[available_columns].to_sql(
        "semantic_events",
        conn,
        if_exists="append",
        index=False
    )

    # --------------------------------------------------------
    # Indexes
    # --------------------------------------------------------

    conn.execute("""
        CREATE INDEX idx_semantic_events_room_time
        ON semantic_events(room_id, event_start)
    """)

    conn.execute("""
        CREATE INDEX idx_semantic_events_reasons
        ON semantic_events(reasons)
    """)

    conn.commit()

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    count = conn.execute("""
        SELECT COUNT(*)
        FROM semantic_events
    """).fetchone()[0]

    print(f"\nRows inserted: {count:,}")

    print("\nSample events:")

    rows = conn.execute("""
        SELECT
            event_id,
            room_id,
            event_start,
            event_end,
            reasons
        FROM semantic_events
        ORDER BY event_id
        LIMIT 5
    """).fetchall()

    for row in rows:
        print(row)

    conn.close()

    print("\n" + "=" * 70)
    print("SEMANTIC EVENT TABLE CREATED")
    print("=" * 70)


if __name__ == "__main__":
    main()