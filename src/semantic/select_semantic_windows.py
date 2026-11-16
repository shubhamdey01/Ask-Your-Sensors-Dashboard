import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sensor.db"
OUTPUT_PATH = PROJECT_ROOT / "data" / "semantic_candidates.csv"


QUERY = """
    SELECT
        window_id,
        room_id,
        window_start,
        window_end,

        temperature_mean,
        temperature_min,
        temperature_max,
        temperature_count,

        humidity_mean,

        co2_mean,
        co2_min,
        co2_max,
        co2_count,

        light_mean,

        pir_active_ratio,
        pir_count

    FROM sensor_windows
"""


NUMERIC_COLUMNS = [
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "temperature_count",
    "humidity_mean",
    "co2_mean",
    "co2_min",
    "co2_max",
    "co2_count",
    "light_mean",
    "pir_active_ratio",
    "pir_count",
]


def load_sensor_windows():
    """Load aggregated sensor windows from SQLite."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(QUERY, conn)


def prepare_numeric_columns(df):
    """Convert sensor columns to numeric values."""

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def calculate_baselines(df):
    """Calculate room-specific anomaly thresholds."""

    co2_stats = (
        df[df["co2_mean"].notna()]
        .groupby("room_id")["co2_mean"]
        .quantile([0.05, 0.95])
        .unstack()
        .rename(
            columns={
                0.05: "co2_p05",
                0.95: "co2_p95",
            }
        )
    )

    temperature_valid = (
        df["temperature_mean"].notna()
        & df["temperature_mean"].between(10, 40)
    )

    temperature_stats = (
        df[temperature_valid]
        .groupby("room_id")["temperature_mean"]
        .quantile([0.05, 0.95])
        .unstack()
        .rename(
            columns={
                0.05: "temperature_p05",
                0.95: "temperature_p95",
            }
        )
    )

    light_stats = (
        df[df["light_mean"].notna()]
        .groupby("room_id")["light_mean"]
        .quantile(0.95)
        .rename("light_p95")
        .to_frame()
    )

    df = df.merge(
        co2_stats,
        on="room_id",
        how="left",
    )

    df = df.merge(
        temperature_stats,
        on="room_id",
        how="left",
    )

    df = df.merge(
        light_stats,
        on="room_id",
        how="left",
    )

    return df, temperature_valid


def detect_events(df, temperature_valid):
    """Identify semantic candidate windows."""

    df["pir_event"] = (
        df["pir_active_ratio"].fillna(0) > 0
    )

    df["co2_high_event"] = (
        df["co2_mean"].notna()
        & (df["co2_mean"] > df["co2_p95"])
    )

    df["co2_low_event"] = (
        df["co2_mean"].notna()
        & (df["co2_mean"] < df["co2_p05"])
    )

    df["temperature_invalid"] = (
        df["temperature_mean"].notna()
        & ~df["temperature_mean"].between(10, 40)
    )

    df["temperature_high_event"] = (
        temperature_valid
        & (
            df["temperature_mean"]
            > df["temperature_p95"]
        )
    )

    df["temperature_low_event"] = (
        temperature_valid
        & (
            df["temperature_mean"]
            < df["temperature_p05"]
        )
    )

    df["light_high_event"] = (
        df["light_mean"].notna()
        & (df["light_mean"] > df["light_p95"])
    )

    df["semantic_candidate"] = (
        df["pir_event"]
        | df["co2_high_event"]
        | df["co2_low_event"]
        | df["temperature_high_event"]
        | df["temperature_low_event"]
        | df["light_high_event"]
    )

    return df


def create_event_reason(row):
    """Create a comma-separated reason string for one window."""

    reasons = []

    conditions = [
        ("pir_activity", row["pir_event"]),
        ("high_co2", row["co2_high_event"]),
        ("low_co2", row["co2_low_event"]),
        ("high_temperature", row["temperature_high_event"]),
        ("low_temperature", row["temperature_low_event"]),
        ("high_light", row["light_high_event"]),
    ]

    for reason, condition in conditions:
        if condition:
            reasons.append(reason)

    return ",".join(reasons)


def print_summary(df, candidates):
    """Print semantic-selection statistics."""

    total_windows = len(df)
    candidate_count = len(candidates)

    print("\nSemantic candidates:")
    print(f"{candidate_count:,}")

    print("\nReduction:")
    print(
        f"{total_windows:,} → {candidate_count:,}"
    )

    print(
        f"Reduction: "
        f"{100 * (1 - candidate_count / total_windows):.2f}%"
    )

    reason_counts = {
        "PIR activity": df["pir_event"].sum(),
        "High CO2": df["co2_high_event"].sum(),
        "Low CO2": df["co2_low_event"].sum(),
        "High temperature": df["temperature_high_event"].sum(),
        "Low temperature": df["temperature_low_event"].sum(),
        "High light": df["light_high_event"].sum(),
    }

    print("\nEvent counts:")

    for reason, count in reason_counts.items():
        print(f"  {reason:<20}: {count:,}")

    print("\nInvalid temperature windows:")
    print(f"  {df['temperature_invalid'].sum():,}")

    room_counts = (
        candidates
        .groupby("room_id")
        .size()
        .sort_values(ascending=False)
    )

    print("\nTop rooms by semantic candidates:")
    print(room_counts.head(15).to_string())

    sample_columns = [
        "window_id",
        "room_id",
        "window_start",
        "temperature_mean",
        "co2_mean",
        "light_mean",
        "pir_active_ratio",
        "event_reason",
    ]

    print("\nSample semantic candidates:")
    print(
        candidates[sample_columns]
        .head(20)
        .to_string(index=False)
    )


def main():
    """Select sensor windows that contain semantic events."""

    print("=" * 70)
    print("SEMANTIC WINDOW SELECTION")
    print("=" * 70)

    df = load_sensor_windows()

    print(f"Total windows: {len(df):,}")

    df = prepare_numeric_columns(df)

    df, temperature_valid = calculate_baselines(df)

    df = detect_events(
        df,
        temperature_valid,
    )

    df["event_reason"] = df.apply(
        create_event_reason,
        axis=1,
    )

    candidates = df[
        df["semantic_candidate"]
    ].copy()

    print_summary(
        df,
        candidates,
    )

    candidates.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved candidates to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()