from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "semantic_events.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "semantic_events_text.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=[
        "event_start",
        "event_end"
    ]
)

print("=" * 70)
print("CREATING SEMANTIC DESCRIPTIONS")
print("=" * 70)

print(f"Events: {len(df):,}")


# ============================================================
# FORMAT HELPERS
# ============================================================

def temperature_text(row):
    """Describe temperature while ignoring obviously invalid values."""

    value = row["temperature_mean"]

    if pd.isna(value):
        return "Temperature data was unavailable."

    # Ignore obviously corrupted values
    if value < 10 or value > 40:
        return "Temperature data was unavailable or invalid."

    return (
        f"average temperature was {value:.2f} degrees Celsius"
    )


def humidity_text(row):
    """Describe humidity."""

    value = row["humidity_mean"]

    if pd.isna(value):
        return "Humidity data was unavailable."

    return f"average humidity was {value:.2f}%"


def co2_text(row):
    """Describe CO2."""

    value = row["co2_mean"]

    if pd.isna(value):
        return "CO2 data was unavailable."

    return f"average CO2 was {value:.1f} ppm"


def light_text(row):
    """Describe light level."""

    value = row["light_mean"]

    if pd.isna(value):
        return "Light data was unavailable."

    return f"average light level was {value:.2f}"


def pir_text(row):
    """Describe PIR activity."""

    value = row["pir_active_ratio"]

    if pd.isna(value):
        return "PIR activity data was unavailable"

    percentage = value * 100

    if percentage == 0:
        return "no PIR activity was detected"

    return (
        f"PIR activity was detected in "
        f"{percentage:.1f}% of the observations"
    )


# ============================================================
# CREATE REASON TEXT
# ============================================================

def reason_text(reasons):
    """
    Convert machine-generated event reasons into
    natural-language descriptions.
    """

    reasons = str(reasons)

    descriptions = []

    if "high_temperature" in reasons:
        descriptions.append(
            "temperature was unusually high relative to "
            "the normal conditions of this room"
        )

    if "low_temperature" in reasons:
        descriptions.append(
            "temperature was unusually low relative to "
            "the normal conditions of this room"
        )

    if "high_co2" in reasons:
        descriptions.append(
            "CO2 was unusually high relative to "
            "the normal conditions of this room"
        )

    if "low_co2" in reasons:
        descriptions.append(
            "CO2 was unusually low relative to "
            "the normal conditions of this room"
        )

    if "high_light" in reasons:
        descriptions.append(
            "light level was unusually high relative to "
            "the normal conditions of this room"
        )

    if "pir_activity" in reasons:
        descriptions.append(
            "PIR activity indicated possible occupancy or movement"
        )

    if "invalid_temperature" in reasons:
        descriptions.append(
            "temperature data contained values considered invalid"
        )

    if not descriptions:
        return "an unusual sensor condition was detected"

    return "; ".join(descriptions)


# ============================================================
# CREATE SEMANTIC TEXT
# ============================================================

descriptions = []


for _, row in df.iterrows():

    room = row["room_id"]

    start = row["event_start"].strftime(
        "%Y-%m-%d %H:%M"
    )

    end = row["event_end"].strftime(
        "%Y-%m-%d %H:%M"
    )

    duration = row["duration_minutes"]

    reasons = reason_text(row["reasons"])

    temperature = temperature_text(row)

    humidity = humidity_text(row)

    co2 = co2_text(row)

    light = light_text(row)

    pir = pir_text(row)

    description = (
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

    descriptions.append(description)


df["semantic_text"] = descriptions


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY EXAMPLES
# ============================================================

print("\nExample semantic descriptions:\n")

for text in df["semantic_text"].head(10):

    print("-" * 70)
    print(text)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Saved to:\n{OUTPUT_PATH}"
)