from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "semantic_events.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "semantic_events_text.csv"


def temperature_text(value):
    """Describe temperature while ignoring obviously invalid values."""

    if pd.isna(value):
        return "Temperature data was unavailable."

    if value < 10 or value > 40:
        return "Temperature data was unavailable or invalid."

    return f"average temperature was {value:.2f} degrees Celsius"


def humidity_text(value):
    """Describe humidity."""

    if pd.isna(value):
        return "Humidity data was unavailable."

    return f"average humidity was {value:.2f}%"


def co2_text(value):
    """Describe CO2."""

    if pd.isna(value):
        return "CO2 data was unavailable."

    return f"average CO2 was {value:.1f} ppm"


def light_text(value):
    """Describe light level."""

    if pd.isna(value):
        return "Light data was unavailable."

    return f"average light level was {value:.2f}"


def pir_text(value):
    """Describe PIR activity."""

    if pd.isna(value):
        return "PIR activity data was unavailable."

    percentage = value * 100

    if percentage == 0:
        return "no PIR activity was detected"

    return (
        f"PIR activity was detected in "
        f"{percentage:.1f}% of the observations"
    )


def reason_text(reasons):
    """Convert event reasons into natural-language descriptions."""

    descriptions = []

    reason_descriptions = {
        "high_temperature": (
            "temperature was unusually high relative to "
            "the normal conditions of this room"
        ),
        "low_temperature": (
            "temperature was unusually low relative to "
            "the normal conditions of this room"
        ),
        "high_co2": (
            "CO2 was unusually high relative to "
            "the normal conditions of this room"
        ),
        "low_co2": (
            "CO2 was unusually low relative to "
            "the normal conditions of this room"
        ),
        "high_light": (
            "light level was unusually high relative to "
            "the normal conditions of this room"
        ),
        "pir_activity": (
            "PIR activity indicated possible occupancy or movement"
        ),
        "invalid_temperature": (
            "temperature data contained values considered invalid"
        ),
    }

    reasons = str(reasons)

    for reason, description in reason_descriptions.items():
        if reason in reasons:
            descriptions.append(description)

    if not descriptions:
        return "an unusual sensor condition was detected"

    return "; ".join(descriptions)


def create_description(row):
    """Create one semantic description from a semantic event."""

    room = row["room_id"]
    start = row["event_start"].strftime("%Y-%m-%d %H:%M")
    end = row["event_end"].strftime("%Y-%m-%d %H:%M")

    duration = row["duration_minutes"]
    reasons = reason_text(row["reasons"])

    temperature = temperature_text(row["temperature_mean"])
    humidity = humidity_text(row["humidity_mean"])
    co2 = co2_text(row["co2_mean"])
    light = light_text(row["light_mean"])
    pir = pir_text(row["pir_active_ratio"])

    return (
        f"Room {room} showed an environmental anomaly "
        f"from {start} to {end}. "
        f"The event lasted approximately "
        f"{duration:.0f} minutes. "
        f"The anomaly was identified because "
        f"{reasons}. "
        f"{temperature}. "
        f"{humidity}. "
        f"{co2}. "
        f"{light}. "
        f"{pir}."
    )


def main():
    """Generate natural-language descriptions for semantic events."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Semantic events file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["event_start", "event_end"],
    )

    required_columns = {
        "event_id",
        "room_id",
        "event_start",
        "event_end",
        "duration_minutes",
        "reasons",
        "temperature_mean",
        "humidity_mean",
        "co2_mean",
        "light_mean",
        "pir_active_ratio",
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "semantic_events.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    print("=" * 70)
    print("CREATING SEMANTIC DESCRIPTIONS")
    print("=" * 70)
    print(f"Events: {len(df):,}")

    df["semantic_text"] = df.apply(
        create_description,
        axis=1,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nExample semantic descriptions:\n")

    for text in df["semantic_text"].head(10):
        print("-" * 70)
        print(text)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()