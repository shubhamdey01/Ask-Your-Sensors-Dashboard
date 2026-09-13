import re
import sqlite3
import pandas as pd
from pathlib import Path
import streamlit as st

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "data" / "sensor.db"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"


# ============================================================
# MODELS / DATABASE CONNECTIONS
# ============================================================

@st.cache_resource
def get_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cuda"
    )


@st.cache_resource
def get_qdrant_client():

    return QdrantClient(
        path=str(QDRANT_PATH)
    )


embedding_model = get_embedding_model()

qdrant = get_qdrant_client()

COLLECTION_NAME = "sensor_events"


# ============================================================
# SQLITE HELPER
# ============================================================

def sqlite_query(sql, params=()):
    """Execute a SQLite query and return a list of dictionaries."""

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


# ============================================================
# BASIC STRUCTURED QUERIES
# ============================================================

def average_temperature(room_id):
    """Return average/min/max temperature for a room."""

    sql = """
        SELECT
            room_id,
            AVG(temperature_mean) AS average_temperature,
            MIN(temperature_min) AS minimum_temperature,
            MAX(temperature_max) AS maximum_temperature
        FROM sensor_windows
        WHERE room_id = ?
          AND temperature_mean IS NOT NULL
          AND temperature_mean BETWEEN 10 AND 40
        GROUP BY room_id
    """

    return sqlite_query(sql, (room_id,))


def absolute_high_temperature_events(room_id=None):
    """
    Return windows with the highest absolute temperatures.

    This answers questions such as:
    'Which rooms had the highest temperatures?'
    """

    if room_id:

        sql = """
            SELECT
                room_id,
                window_start,
                window_end,
                temperature_mean,
                temperature_min,
                temperature_max
            FROM sensor_windows
            WHERE room_id = ?
              AND temperature_mean IS NOT NULL
              AND temperature_mean BETWEEN 10 AND 40
            ORDER BY temperature_mean DESC
            LIMIT 20
        """

        return sqlite_query(sql, (room_id,))

    sql = """
        SELECT
            room_id,
            window_start,
            window_end,
            temperature_mean,
            temperature_min,
            temperature_max
        FROM sensor_windows
        WHERE temperature_mean IS NOT NULL
          AND temperature_mean BETWEEN 10 AND 40
        ORDER BY temperature_mean DESC
        LIMIT 20
    """

    return sqlite_query(sql)


def absolute_high_co2_events(room_id=None):
    """
    Return windows with the highest absolute CO2.

    This answers questions such as:
    'Which rooms had the highest CO2?'
    """

    if room_id:

        sql = """
            SELECT
                room_id,
                window_start,
                window_end,
                co2_mean,
                co2_min,
                co2_max
            FROM sensor_windows
            WHERE room_id = ?
              AND co2_mean IS NOT NULL
            ORDER BY co2_mean DESC
            LIMIT 20
        """

        return sqlite_query(sql, (room_id,))

    sql = """
        SELECT
            room_id,
            window_start,
            window_end,
            co2_mean,
            co2_min,
            co2_max
        FROM sensor_windows
        WHERE co2_mean IS NOT NULL
        ORDER BY co2_mean DESC
        LIMIT 20
    """

    return sqlite_query(sql)


# ============================================================
# ROOM-RELATIVE ANOMALIES
# ============================================================

def room_relative_anomalies(
    column,
    room_id=None,
    percentile=0.95,
    limit=20
):
    """
    Find unusually high values relative to each room's
    own normal distribution.

    This uses the same percentile-based idea used when
    semantic events were created.

    Example:
        temperature -> 95th percentile per room
        CO2         -> 95th percentile per room
    """

    sql = f"""
        SELECT
            room_id,
            window_start,
            window_end,
            {column} AS value
        FROM sensor_windows
        WHERE {column} IS NOT NULL
    """

    params = ()

    if room_id:
        sql += " AND room_id = ?"
        params = (room_id,)

    rows = sqlite_query(sql, params)

    if not rows:
        return []

    df = pd.DataFrame(rows)

    # Remove invalid temperatures
    if column == "temperature_mean":
        df = df[
            (df["value"] >= 10) &
            (df["value"] <= 40)
        ]

    if df.empty:
        return []

    # Calculate room-specific percentile
    thresholds = (
        df.groupby("room_id")["value"]
        .quantile(percentile)
        .rename("room_threshold")
        .reset_index()
    )

    df = df.merge(
        thresholds,
        on="room_id",
        how="left"
    )

    # Keep values above each room's own threshold
    df = df[
        df["value"] >= df["room_threshold"]
    ]

    df = df.sort_values(
        ["value"],
        ascending=False
    ).head(limit)

    return df.to_dict("records")


def unusually_high_temperature_events(
    room_id=None,
    limit=20
):
    """
    Find temperatures that are unusually high
    relative to the normal conditions of each room.
    """

    return room_relative_anomalies(
        column="temperature_mean",
        room_id=room_id,
        percentile=0.95,
        limit=limit
    )


def unusually_high_co2_events(
    room_id=None,
    limit=20
):
    """
    Find CO2 values that are unusually high
    relative to the normal conditions of each room.
    """

    return room_relative_anomalies(
        column="co2_mean",
        room_id=room_id,
        percentile=0.95,
        limit=limit
    )


def unusually_high_light_events(
    room_id=None,
    limit=20
):
    """
    Find light levels that are unusually high
    relative to each room's normal conditions.
    """

    return room_relative_anomalies(
        column="light_mean",
        room_id=room_id,
        percentile=0.95,
        limit=limit
    )


# ============================================================
# OCCUPANCY / PIR
# ============================================================

def occupancy_events(room_id=None, limit=20):
    """
    Return grouped semantic events containing PIR activity.

    Duration is calculated from event_start and event_end because
    duration_minutes is not stored in the SQLite semantic_events table.
    """

    if room_id:

        sql = """
            SELECT
                event_id,
                room_id,
                event_start,
                event_end,
                ROUND(
                    (julianday(event_end) - julianday(event_start))
                    * 1440.0,
                    1
                ) AS duration_minutes,
                window_count,
                pir_active_ratio,
                temperature_mean,
                humidity_mean,
                co2_mean,
                light_mean,
                reasons
            FROM semantic_events
            WHERE room_id = ?
              AND reasons LIKE '%pir_activity%'
            ORDER BY event_start
            LIMIT ?
        """

        return sqlite_query(
            sql,
            (room_id, limit)
        )

    sql = """
        SELECT
            event_id,
            room_id,
            event_start,
            event_end,
            ROUND(
                (julianday(event_end) - julianday(event_start))
                * 1440.0,
                1
            ) AS duration_minutes,
            window_count,
            pir_active_ratio,
            temperature_mean,
            humidity_mean,
            co2_mean,
            light_mean,
            reasons
        FROM semantic_events
        WHERE reasons LIKE '%pir_activity%'
        ORDER BY event_start
        LIMIT ?
    """

    return sqlite_query(
        sql,
        (limit,)
    )


# ============================================================
# QDRANT SEMANTIC SEARCH
# ============================================================

def semantic_search(query, limit=5):
    """Search semantic sensor events using vector similarity."""

    vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit
    )

    output = []

    for point in results.points:

        payload = point.payload

        output.append({
            "score": point.score,
            "event_id": payload.get("event_id"),
            "room_id": payload.get("room_id"),
            "event_start": payload.get("event_start"),
            "event_end": payload.get("event_end"),
            "duration_minutes": payload.get(
                "duration_minutes"
            ),
            "reasons": payload.get("reasons"),
            "semantic_text": payload.get(
                "semantic_text"
            )
        })

    return output


# ============================================================
# ROOM EXTRACTION
# ============================================================

def extract_room(query):
    """Extract room number such as 'room 776' or 'room 621E'."""

    match = re.search(
        r"\broom\s+([0-9]+[a-zA-Z]?)\b",
        query,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# QUERY ROUTER
# ============================================================

def route_query(query):
    """
    Decide which database source should answer the question.
    """

    q = query.lower()

    room_id = extract_room(q)

    # --------------------------------------------------------
    # OCCUPANCY / PIR
    # --------------------------------------------------------

    occupancy_words = [
        "occupied",
        "occupancy",
        "human activity",
        "people",
        "person",
        "presence",
        "movement",
        "pir"
    ]

    if any(word in q for word in occupancy_words):

        return {
            "source": "sqlite",
            "query_type": "occupancy",
            "room_id": room_id
        }


    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    temperature_words = [
        "temperature",
        "temp",
        "warm",
        "warmer",
        "hot",
        "hotter"
    ]

    if any(word in q for word in temperature_words):

        # ----------------------------------------------------
        # Absolute temperature questions
        # ----------------------------------------------------

        if any(
            word in q
            for word in [
                "average",
                "mean"
            ]
        ):

            return {
                "source": "sqlite",
                "query_type": "average_temperature",
                "room_id": room_id
            }


        if any(
            word in q
            for word in [
                "highest",
                "maximum",
                "hottest"
            ]
        ):

            return {
                "source": "sqlite",
                "query_type": "absolute_temperature",
                "room_id": room_id
            }


        if any(
            word in q
            for word in [
                "lowest",
                "minimum",
                "coldest"
            ]
        ):

            return {
                "source": "sqlite",
                "query_type": "absolute_temperature",
                "room_id": room_id
            }


        # ----------------------------------------------------
        # Room-relative temperature anomaly
        # ----------------------------------------------------

        if any(
            word in q
            for word in [
                "unusual",
                "unusually",
                "warm",
                "warmer",
                "hot",
                "hotter",
                "high"
            ]
        ):

            return {
                "source": "hybrid",
                "query_type": "high_temperature",
                "room_id": room_id
            }

    # --------------------------------------------------------
    # CO2
    # --------------------------------------------------------

    if "co2" in q or "carbon dioxide" in q:

        if any(
            word in q
            for word in [
                "average",
                "mean"
            ]
        ):

            return {
                "source": "sqlite",
                "query_type": "absolute_co2",
                "room_id": room_id
            }

        if any(
            word in q
            for word in [
                "unusual",
                "unusually",
                "high",
                "elevated"
            ]
        ):

            return {
                "source": "hybrid",
                "query_type": "high_co2",
                "room_id": room_id
            }


    # --------------------------------------------------------
    # LIGHT
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "light",
            "lighting",
            "bright",
            "brightness"
        ]
    ):

        # Absolute lighting question
        if any(
            word in q
            for word in [
                "highest",
                "maximum",
                "brightest"
            ]
        ):

            return {
                "source": "sqlite",
                "query_type": "absolute_light",
                "room_id": room_id
            }

        # Otherwise room-relative semantic anomaly
        return {
            "source": "hybrid",
            "query_type": "high_light",
            "room_id": room_id
        }


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return {
        "source": "qdrant",
        "query_type": "semantic",
        "room_id": room_id
    }


# ============================================================
# QUERY EXECUTION
# ============================================================

def execute_query(query):
    """Route and execute a natural-language sensor query."""

    route = route_query(query)

    source = route["source"]
    query_type = route["query_type"]
    room_id = route["room_id"]


    # --------------------------------------------------------
    # AVERAGE TEMPERATURE
    # --------------------------------------------------------

    if query_type == "average_temperature":

        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results": average_temperature(room_id)
        }


    # --------------------------------------------------------
    # ABSOLUTE TEMPERATURE
    # --------------------------------------------------------

    if query_type == "absolute_temperature":

        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results":
                absolute_high_temperature_events(room_id)
        }


    # --------------------------------------------------------
    # HIGH TEMPERATURE
    # --------------------------------------------------------

    if query_type == "high_temperature":

        return {
            "source": "hybrid",
            "query_type": query_type,
            "structured_results":
                unusually_high_temperature_events(room_id),
            "semantic_results":
                semantic_search(query, limit=5)
        }


    # --------------------------------------------------------
    # OCCUPANCY
    # --------------------------------------------------------

    if query_type == "occupancy":

        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results":
                occupancy_events(
                    room_id=room_id,
                    limit=20
                )
        }


    # --------------------------------------------------------
    # CO2
    # --------------------------------------------------------

    if query_type == "absolute_co2":

        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results":
                absolute_high_co2_events(room_id)
        }


    if query_type == "high_co2":

        return {
            "source": "hybrid",
            "query_type": query_type,
            "structured_results":
                unusually_high_co2_events(room_id),
            "semantic_results":
                semantic_search(query, limit=5)
        }


    # --------------------------------------------------------
    # LIGHT
    # --------------------------------------------------------

    if query_type == "absolute_light":

        results = room_relative_anomalies(
            column="light_mean",
            room_id=room_id,
            percentile=1.0,
            limit=20
        )

        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results": results
        }


    if query_type == "high_light":

        return {
            "source": "hybrid",
            "query_type": query_type,
            "structured_results":
                unusually_high_light_events(room_id),
            "semantic_results":
                semantic_search(query, limit=5)
        }


    # --------------------------------------------------------
    # GENERAL SEMANTIC SEARCH
    # --------------------------------------------------------

    return {
        "source": "qdrant",
        "query_type": "semantic",
        "semantic_results":
            semantic_search(query, limit=5)
    }

