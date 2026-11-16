import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DB_PATH = PROJECT_ROOT / "data" / "sensor.db"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "sensor_events"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def get_embedding_model():
    """Load and cache the embedding model."""

    return SentenceTransformer(
        EMBEDDING_MODEL,
        device="cuda",
    )


@st.cache_resource
def get_qdrant_client():
    """Open and cache the local Qdrant client."""

    return QdrantClient(
        path=str(QDRANT_PATH),
    )


def sqlite_query(sql, params=()):
    """Execute a SQLite query and return dictionaries."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            sql,
            params,
        ).fetchall()

    return [dict(row) for row in rows]


def average_temperature(room_id=None):
    """Return average, minimum, and maximum valid temperature."""

    sql = """
        SELECT
            AVG(temperature_mean) AS average_temperature,
            MIN(temperature_min) AS minimum_temperature,
            MAX(temperature_max) AS maximum_temperature
        FROM sensor_windows
        WHERE temperature_mean IS NOT NULL
          AND temperature_mean BETWEEN 10 AND 40
    """

    params = ()

    if room_id:
        sql += " AND room_id = ?"
        params = (room_id,)

    return sqlite_query(sql, params)


def absolute_temperature_events(
    room_id=None,
    ascending=False,
    limit=20,
):
    """Return windows with the highest or lowest absolute temperature."""

    order = "ASC" if ascending else "DESC"

    sql = f"""
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
    """

    params = ()

    if room_id:
        sql += " AND room_id = ?"
        params = (room_id,)

    sql += f"""
        ORDER BY temperature_mean {order}
        LIMIT ?
    """

    return sqlite_query(
        sql,
        (*params, limit),
    )


def absolute_high_co2_events(room_id=None, limit=20):
    """Return windows with the highest absolute CO2 values."""

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
    """

    params = ()

    if room_id:
        sql += " AND room_id = ?"
        params = (room_id,)

    sql += """
        ORDER BY co2_mean DESC
        LIMIT ?
    """

    return sqlite_query(
        sql,
        (*params, limit),
    )


def absolute_high_light_events(room_id=None, limit=20):
    """Return windows with the highest absolute light values."""

    sql = """
        SELECT
            room_id,
            window_start,
            window_end,
            light_mean,
            light_min,
            light_max
        FROM sensor_windows
        WHERE light_mean IS NOT NULL
    """

    params = ()

    if room_id:
        sql += " AND room_id = ?"
        params = (room_id,)

    sql += """
        ORDER BY light_mean DESC
        LIMIT ?
    """

    return sqlite_query(
        sql,
        (*params, limit),
    )


def room_relative_anomalies(
    column,
    room_id=None,
    percentile=0.95,
    limit=20,
):
    """Find unusually high values relative to each room's baseline."""

    allowed_columns = {
        "temperature_mean",
        "co2_mean",
        "light_mean",
    }

    if column not in allowed_columns:
        raise ValueError(
            f"Unsupported sensor column: {column}"
        )

    if not 0 < percentile <= 1:
        raise ValueError(
            "Percentile must be between 0 and 1."
        )

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

    rows = sqlite_query(
        sql,
        params,
    )

    if not rows:
        return []

    df = pd.DataFrame(rows)

    if column == "temperature_mean":
        df = df[
            df["value"].between(10, 40)
        ]

    if df.empty:
        return []

    thresholds = (
        df.groupby("room_id")["value"]
        .quantile(percentile)
        .rename("room_threshold")
        .reset_index()
    )

    df = df.merge(
        thresholds,
        on="room_id",
        how="left",
    )

    df = df[
        df["value"] >= df["room_threshold"]
    ]

    return (
        df.sort_values(
            "value",
            ascending=False,
        )
        .head(limit)
        .to_dict("records")
    )


def unusually_high_temperature_events(
    room_id=None,
    limit=20,
):
    """Find unusually high temperatures relative to each room."""

    return room_relative_anomalies(
        column="temperature_mean",
        room_id=room_id,
        percentile=0.95,
        limit=limit,
    )


def unusually_high_co2_events(
    room_id=None,
    limit=20,
):
    """Find unusually high CO2 relative to each room."""

    return room_relative_anomalies(
        column="co2_mean",
        room_id=room_id,
        percentile=0.95,
        limit=limit,
    )


def unusually_high_light_events(
    room_id=None,
    limit=20,
):
    """Find unusually high light levels relative to each room."""

    return room_relative_anomalies(
        column="light_mean",
        room_id=room_id,
        percentile=0.95,
        limit=limit,
    )


def occupancy_events(room_id=None, limit=20):
    """Return semantic events containing PIR activity."""

    sql = """
        SELECT
            event_id,
            room_id,
            event_start,
            event_end,
            ROUND(
                (
                    julianday(event_end)
                    - julianday(event_start)
                ) * 1440.0,
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
    """

    params = ()

    if room_id:
        sql += " AND room_id = ?"
        params = (room_id,)

    sql += """
        ORDER BY event_start
        LIMIT ?
    """

    return sqlite_query(
        sql,
        (*params, limit),
    )


def semantic_search(query, limit=5):
    """Search semantic sensor events using vector similarity."""

    embedding_model = get_embedding_model()
    qdrant = get_qdrant_client()

    vector = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit,
    )

    return [
        {
            "score": point.score,
            "event_id": point.payload.get("event_id"),
            "room_id": point.payload.get("room_id"),
            "event_start": point.payload.get("event_start"),
            "event_end": point.payload.get("event_end"),
            "duration_minutes": point.payload.get(
                "duration_minutes"
            ),
            "reasons": point.payload.get("reasons"),
            "semantic_text": point.payload.get(
                "semantic_text"
            ),
        }
        for point in results.points
    ]


def extract_room(query):
    """Extract a room identifier such as 776 or 621E."""

    match = re.search(
        r"\broom\s+([0-9]+[a-zA-Z]?)\b",
        query,
        re.IGNORECASE,
    )

    return match.group(1) if match else None


def route_query(query):
    """Determine the appropriate source and query type."""

    q = query.lower()
    room_id = extract_room(q)

    occupancy_words = [
        "occupied",
        "occupancy",
        "human activity",
        "people",
        "person",
        "presence",
        "movement",
        "pir",
    ]

    if any(word in q for word in occupancy_words):
        return {
            "source": "sqlite",
            "query_type": "occupancy",
            "room_id": room_id,
        }

    temperature_words = [
        "temperature",
        "temp",
        "warm",
        "warmer",
        "hot",
        "hotter",
    ]

    if any(word in q for word in temperature_words):

        if any(word in q for word in ["average", "mean"]):
            return {
                "source": "sqlite",
                "query_type": "average_temperature",
                "room_id": room_id,
            }

        if any(
            word in q
            for word in [
                "highest",
                "maximum",
                "hottest",
            ]
        ):
            return {
                "source": "sqlite",
                "query_type": "absolute_temperature_high",
                "room_id": room_id,
            }

        if any(
            word in q
            for word in [
                "lowest",
                "minimum",
                "coldest",
            ]
        ):
            return {
                "source": "sqlite",
                "query_type": "absolute_temperature_low",
                "room_id": room_id,
            }

        if any(
            word in q
            for word in [
                "unusual",
                "unusually",
                "warm",
                "warmer",
                "hot",
                "hotter",
                "high",
            ]
        ):
            return {
                "source": "hybrid",
                "query_type": "high_temperature",
                "room_id": room_id,
            }

    if "co2" in q or "carbon dioxide" in q:

        if any(word in q for word in ["average", "mean"]):
            return {
                "source": "sqlite",
                "query_type": "absolute_co2",
                "room_id": room_id,
            }

        if any(
            word in q
            for word in [
                "unusual",
                "unusually",
                "high",
                "elevated",
            ]
        ):
            return {
                "source": "hybrid",
                "query_type": "high_co2",
                "room_id": room_id,
            }

    light_words = [
        "light",
        "lighting",
        "bright",
        "brightness",
    ]

    if any(word in q for word in light_words):

        if any(
            word in q
            for word in [
                "highest",
                "maximum",
                "brightest",
            ]
        ):
            return {
                "source": "sqlite",
                "query_type": "absolute_light",
                "room_id": room_id,
            }

        return {
            "source": "hybrid",
            "query_type": "high_light",
            "room_id": room_id,
        }

    return {
        "source": "qdrant",
        "query_type": "semantic",
        "room_id": room_id,
    }


def execute_query(query):
    """Route and execute a natural-language sensor query."""

    route = route_query(query)

    query_type = route["query_type"]
    room_id = route["room_id"]

    if query_type == "average_temperature":
        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results": average_temperature(room_id),
        }

    if query_type == "absolute_temperature_high":
        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results": absolute_temperature_events(
                room_id=room_id,
                ascending=False,
            ),
        }

    if query_type == "absolute_temperature_low":
        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results": absolute_temperature_events(
                room_id=room_id,
                ascending=True,
            ),
        }

    if query_type == "high_temperature":
        return {
            "source": "hybrid",
            "query_type": query_type,
            "structured_results":
                unusually_high_temperature_events(
                    room_id
                ),
            "semantic_results":
                semantic_search(query),
        }

    if query_type == "occupancy":
        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results":
                occupancy_events(
                    room_id=room_id,
                ),
        }

    if query_type == "absolute_co2":
        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results":
                absolute_high_co2_events(
                    room_id=room_id,
                ),
        }

    if query_type == "high_co2":
        return {
            "source": "hybrid",
            "query_type": query_type,
            "structured_results":
                unusually_high_co2_events(
                    room_id
                ),
            "semantic_results":
                semantic_search(query),
        }

    if query_type == "absolute_light":
        return {
            "source": "sqlite",
            "query_type": query_type,
            "structured_results":
                absolute_high_light_events(
                    room_id=room_id,
                ),
        }

    if query_type == "high_light":
        return {
            "source": "hybrid",
            "query_type": query_type,
            "structured_results":
                unusually_high_light_events(
                    room_id
                ),
            "semantic_results":
                semantic_search(query),
        }

    return {
        "source": "qdrant",
        "query_type": "semantic",
        "semantic_results":
            semantic_search(query),
    }