from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "semantic_candidates.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "semantic_events.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=[
        "window_start",
        "window_end"
    ]
)

print("=" * 70)
print("GROUPING SEMANTIC WINDOWS INTO EVENTS")
print("=" * 70)

print(f"Candidate windows: {len(df):,}")


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["room_id", "window_start"]
).reset_index(drop=True)


# ============================================================
# IDENTIFY EVENT BOUNDARIES
# ============================================================

previous_room = df["room_id"].shift(1)

previous_time = df["window_start"].shift(1)


time_gap = (
    df["window_start"]
    - previous_time
).dt.total_seconds() / 60


new_event = (
    (df["room_id"] != previous_room)
    | (time_gap > 5)
)


# Event number within the entire dataset
df["event_id"] = new_event.cumsum()


# ============================================================
# GROUP EVENTS
# ============================================================

events = (
    df.groupby(
        ["room_id", "event_id"],
        as_index=False
    )
    .agg(
        event_start=("window_start", "min"),
        event_end=("window_end", "max"),

        window_count=("window_id", "count"),

        temperature_mean=("temperature_mean", "mean"),
        temperature_min=("temperature_min", "min"),
        temperature_max=("temperature_max", "max"),

        humidity_mean=("humidity_mean", "mean"),

        co2_mean=("co2_mean", "mean"),
        co2_min=("co2_min", "min"),
        co2_max=("co2_max", "max"),

        light_mean=("light_mean", "mean"),
        light_min=("light_mean", "min"),
        light_max=("light_mean", "max"),

        pir_active_ratio=("pir_active_ratio", "mean"),

        reasons=("event_reason", lambda x:
                 ",".join(sorted(set(
                     ",".join(x.dropna()).split(",")
                 ))))
    )
)


# ============================================================
# EVENT DURATION
# ============================================================

events["duration_minutes"] = (
    events["event_end"]
    - events["event_start"]
).dt.total_seconds() / 60


# ============================================================
# REMOVE INTERNAL EVENT ID
# ============================================================

events = events.drop(
    columns=["event_id"]
)


# ============================================================
# SUMMARY
# ============================================================

print(
    f"\nSemantic events: {len(events):,}"
)

print(
    f"Reduction: "
    f"{100 * (1 - len(events) / len(df)):.2f}%"
)

print(
    f"\nAverage windows per event: "
    f"{len(df) / len(events):.2f}"
)


# ============================================================
# EVENT SIZE DISTRIBUTION
# ============================================================

print("\nEvent window distribution:")

print(
    events["window_count"].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


# ============================================================
# SAMPLE EVENTS
# ============================================================

print("\nSample events:")

sample_columns = [
    "room_id",
    "event_start",
    "event_end",
    "duration_minutes",
    "window_count",
    "temperature_mean",
    "co2_mean",
    "light_mean",
    "pir_active_ratio",
    "reasons",
]

print(
    events[
        sample_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

# Assign a stable identifier to every semantic event
events["event_id"] = range(1, len(events) + 1)

events.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved to:"
    f"\n{OUTPUT_PATH}"
)