from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "semantic_candidates.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "semantic_events.csv"


REQUIRED_COLUMNS = [
    "window_id",
    "room_id",
    "window_start",
    "window_end",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "humidity_mean",
    "co2_mean",
    "co2_min",
    "co2_max",
    "light_mean",
    "pir_active_ratio",
    "event_reason",
]


def load_candidates():
    """Load semantic candidate windows."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Semantic candidates file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=[
            "window_start",
            "window_end",
        ],
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "semantic_candidates.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    return df


def identify_event_groups(df):
    """Identify boundaries between consecutive semantic events."""

    df = df.sort_values(
        ["room_id", "window_start"]
    ).reset_index(drop=True)

    previous_room = df["room_id"].shift()
    previous_time = df["window_start"].shift()

    time_gap_minutes = (
        df["window_start"] - previous_time
    ).dt.total_seconds() / 60

    new_event = (
        df["room_id"].ne(previous_room)
        | time_gap_minutes.gt(5)
    )

    df["_event_group"] = new_event.cumsum()

    return df


def combine_reasons(values):
    """Combine unique event reasons into one comma-separated string."""

    reasons = set()

    for value in values.dropna():
        reasons.update(
            reason
            for reason in str(value).split(",")
            if reason
        )

    return ",".join(sorted(reasons))


def group_events(df):
    """Combine consecutive candidate windows into semantic events."""

    return (
        df.groupby(
            ["room_id", "_event_group"],
            as_index=False,
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

            pir_active_ratio=("pir_active_ratio", "mean"),

            reasons=("event_reason", combine_reasons),
        )
    )


def add_event_metadata(events):
    """Add duration and stable event identifiers."""

    events["duration_minutes"] = (
        events["event_end"]
        - events["event_start"]
    ).dt.total_seconds() / 60

    events = events.sort_values(
        ["room_id", "event_start"]
    ).reset_index(drop=True)

    events["event_id"] = range(
        1,
        len(events) + 1,
    )

    columns = [
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

    return events[columns]


def print_summary(candidates, events):
    """Print event-grouping statistics."""

    print(
        f"\nSemantic events: {len(events):,}"
    )

    print(
        f"Reduction: "
        f"{100 * (1 - len(events) / len(candidates)):.2f}%"
    )

    print(
        f"\nAverage windows per event: "
        f"{len(candidates) / len(events):.2f}"
    )

    print("\nEvent window distribution:")

    print(
        events["window_count"].describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    sample_columns = [
        "event_id",
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

    print("\nSample events:")

    print(
        events[sample_columns]
        .head(20)
        .to_string(index=False)
    )


def main():
    """Group semantic candidate windows into semantic events."""

    print("=" * 70)
    print("GROUPING SEMANTIC WINDOWS INTO EVENTS")
    print("=" * 70)

    candidates = load_candidates()

    print(
        f"Candidate windows: {len(candidates):,}"
    )

    candidates = identify_event_groups(candidates)

    events = group_events(candidates)

    events = add_event_metadata(events)

    print_summary(
        candidates,
        events,
    )

    events.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()