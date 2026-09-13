from pathlib import Path
import sqlite3
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sensor.db"


# ============================================================
# LOAD DATA
# ============================================================

conn = sqlite3.connect(DB_PATH)

query = """
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

df = pd.read_sql_query(query, conn)

conn.close()


print("=" * 70)
print("SEMANTIC WINDOW SELECTION")
print("=" * 70)

print(f"Total windows: {len(df):,}")


# ============================================================
# DATA TYPES
# ============================================================

numeric_columns = [
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

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 1. PIR ACTIVITY
# ============================================================

df["pir_event"] = (
    df["pir_active_ratio"].fillna(0) > 0
)


# ============================================================
# 2. ROOM-SPECIFIC CO2 BASELINES
# ============================================================

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

df = df.merge(
    co2_stats,
    on="room_id",
    how="left"
)

df["co2_high_event"] = (
    df["co2_mean"].notna()
    & (df["co2_mean"] > df["co2_p95"])
)

df["co2_low_event"] = (
    df["co2_mean"].notna()
    & (df["co2_mean"] < df["co2_p05"])
)


# ============================================================
# 3. ROOM-SPECIFIC TEMPERATURE BASELINES
# ============================================================

# Ignore obviously corrupted temperatures when calculating
# room-specific baselines.

valid_temperature = (
    df["temperature_mean"].notna()
    & (df["temperature_mean"] >= 10)
    & (df["temperature_mean"] <= 40)
)

temperature_stats = (
    df[valid_temperature]
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

df = df.merge(
    temperature_stats,
    on="room_id",
    how="left"
)


df["temperature_invalid"] = (
    df["temperature_mean"].notna()
    & (
        (df["temperature_mean"] < 10)
        | (df["temperature_mean"] > 40)
    )
)


df["temperature_high_event"] = (
    valid_temperature
    & (
        df["temperature_mean"]
        > df["temperature_p95"]
    )
)


df["temperature_low_event"] = (
    valid_temperature
    & (
        df["temperature_mean"]
        < df["temperature_p05"]
    )
)


# ============================================================
# 4. LIGHT BASELINE
# ============================================================

light_stats = (
    df[df["light_mean"].notna()]
    .groupby("room_id")["light_mean"]
    .quantile([0.95])
    .unstack()
    .rename(
        columns={
            0.95: "light_p95"
        }
    )
)

df = df.merge(
    light_stats,
    on="room_id",
    how="left"
)

df["light_high_event"] = (
    df["light_mean"].notna()
    & (df["light_mean"] > df["light_p95"])
)


# ============================================================
# 5. COMBINE EVENT CONDITIONS
# ============================================================

df["semantic_candidate"] = (
    df["pir_event"]
    | df["co2_high_event"]
    | df["co2_low_event"]
    | df["temperature_high_event"]
    | df["temperature_low_event"]
    | df["light_high_event"]
)


# ============================================================
# EVENT REASONS
# ============================================================

df["event_reason"] = ""

conditions = [
    df["pir_event"],
    df["co2_high_event"],
    df["co2_low_event"],
    df["temperature_high_event"],
    df["temperature_low_event"],
    df["light_high_event"],
]

names = [
    "pir_activity",
    "high_co2",
    "low_co2",
    "high_temperature",
    "low_temperature",
    "high_light",
]

for condition, name in zip(conditions, names):

    df.loc[
        condition,
        "event_reason"
    ] = df.loc[
        condition,
        "event_reason"
    ].apply(
        lambda x: (
            f"{x},{name}"
            if x
            else name
        )
    )


# ============================================================
# SUMMARY
# ============================================================

candidates = df[
    df["semantic_candidate"]
].copy()


print("\nSemantic candidates:")
print(f"{len(candidates):,}")


print("\nReduction:")
print(
    f"{len(df):,} → {len(candidates):,}"
)

print(
    f"Reduction: "
    f"{100 * (1 - len(candidates) / len(df)):.2f}%"
)


# ============================================================
# REASON COUNTS
# ============================================================

print("\nEvent counts:")

reason_counts = {
    "PIR activity": df["pir_event"].sum(),
    "High CO2": df["co2_high_event"].sum(),
    "Low CO2": df["co2_low_event"].sum(),
    "High temperature": df["temperature_high_event"].sum(),
    "Low temperature": df["temperature_low_event"].sum(),
    "High light": df["light_high_event"].sum(),
}

for reason, count in reason_counts.items():
    print(
        f"  {reason:<20}: {count:,}"
    )


# ============================================================
# INVALID TEMPERATURE
# ============================================================

print("\nInvalid temperature windows:")
print(
    f"  {df['temperature_invalid'].sum():,}"
)


# ============================================================
# CANDIDATES PER ROOM
# ============================================================

room_counts = (
    candidates
    .groupby("room_id")
    .size()
    .sort_values(ascending=False)
)

print("\nTop rooms by semantic candidates:")

print(
    room_counts.head(15).to_string()
)


# ============================================================
# SAMPLE CANDIDATES
# ============================================================

print("\nSample semantic candidates:")

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

print(
    candidates[
        sample_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE CANDIDATES
# ============================================================

output_path = (
    PROJECT_ROOT
    / "data"
    / "semantic_candidates.csv"
)

candidates.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved candidates to:"
    f"\n{output_path}"
)