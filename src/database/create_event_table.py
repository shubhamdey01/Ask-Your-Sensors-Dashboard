import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DB_PATH = PROJECT_ROOT / "data" / "sensor.db"
EVENTS_FILE = PROJECT_ROOT / "data" / "semantic_events.csv"


EVENT_COLUMNS = [
    "event_id",
    "room_id",
    "event_start",
    "event_end",
    "duration_minutes",
    "window_count",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "humidity_mean",
    "co2_mean",
    "co2_min",
    "co2_max",
    "light_mean",
    "pir_active_ratio",
    "reasons",
]


def create_table(conn):
    """Create the semantic events table."""

    conn.execute("DROP TABLE IF EXISTS semantic_events")

    conn.execute(
        """
        CREATE TABLE semantic_events (
            event_id INTEGER PRIMARY KEY,
            room_id TEXT NOT NULL,
            event_start TEXT NOT NULL,
            event_end TEXT NOT NULL,
            duration_minutes REAL,
            window_count INTEGER,

            temperature_mean REAL,
            temperature_min REAL,
            temperature_max REAL,

            humidity_mean REAL,

            co2_mean REAL,
            co2_min REAL,
            co2_max REAL,

            light_mean REAL,

            pir_active_ratio REAL,

            reasons TEXT
        )
        """
    )


def insert_events(conn, events):
    """Insert semantic events into SQLite."""

    missing_columns = [
        column
        for column in EVENT_COLUMNS
        if column not in events.columns
    ]

    if missing_columns:
        raise ValueError(
            "semantic_events.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    events[EVENT_COLUMNS].to_sql(
        "semantic_events",
        conn,
        if_exists="append",
        index=False,
    )


def create_indexes(conn):
    """Create indexes used by semantic event queries."""

    conn.execute(
        """
        CREATE INDEX idx_semantic_events_room_time
        ON semantic_events(room_id, event_start)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_semantic_events_reasons
        ON semantic_events(reasons)
        """
    )


def verify_table(conn):
    """Print basic information from the created table."""

    count = conn.execute(
        "SELECT COUNT(*) FROM semantic_events"
    ).fetchone()[0]

    print(f"\nRows inserted: {count:,}")

    rows = conn.execute(
        """
        SELECT
            event_id,
            room_id,
            event_start,
            event_end,
            duration_minutes,
            reasons
        FROM semantic_events
        ORDER BY event_id
        LIMIT 5
        """
    ).fetchall()

    print("\nSample events:")

    for row in rows:
        print(row)


def main():
    """Rebuild and verify the SQLite semantic events table."""

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Semantic events file not found: {EVENTS_FILE}"
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    print("=" * 70)
    print("CREATING SEMANTIC EVENT TABLE")
    print("=" * 70)

    events = pd.read_csv(EVENTS_FILE)

    print(f"\nEvents loaded: {len(events):,}")

    with sqlite3.connect(DB_PATH) as conn:
        create_table(conn)
        insert_events(conn, events)
        create_indexes(conn)
        conn.commit()

        verify_table(conn)

    print("\n" + "=" * 70)
    print("SEMANTIC EVENT TABLE CREATED")
    print("=" * 70)


if __name__ == "__main__":
    main()