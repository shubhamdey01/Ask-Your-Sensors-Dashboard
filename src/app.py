import streamlit as st

from query_engine import (
    execute_query,
    get_embedding_model,
    get_qdrant_client
)

from ask_sensors import (
    format_evidence,
    ask_ollama
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Ask the Sensors",
    page_icon="📡",
    layout="wide"
)

# ============================================================
# INITIALIZE BACKEND
# ============================================================

if "backend_ready" not in st.session_state:

    with st.status(
        "Initializing Ask the Sensors...",
        expanded=True
    ) as status:

        status.write(
            "Loading semantic search model..."
        )

        get_embedding_model()

        status.write(
            "Connecting to sensor vector database..."
        )

        get_qdrant_client()

        status.update(
            label="Ask the Sensors is ready!",
            state="complete",
            expanded=False
        )

        st.session_state["backend_ready"] = True


# ============================================================
# TITLE
# ============================================================

st.title("📡 Ask the Sensors")

st.markdown(
    """
    **Natural-language analytics for smart-building sensor data**

    Ask questions about temperature, humidity, CO₂, lighting,
    and occupancy. The system combines structured SQL queries
    with semantic vector search when appropriate.
    """
)


st.divider()


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader("💡 Example Questions")

example_questions = [
    "What was the average temperature in room 776?",
    "Which rooms were unusually warm?",
    "When was room 776 occupied?",
    "Which rooms had unusually high CO2?",
    "Which rooms had the highest CO2?",
    "Which rooms had unusually bright lighting?"
]


cols = st.columns(3)

for i, example in enumerate(example_questions):

    with cols[i % 3]:

        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True
        ):

            st.session_state["question"] = example


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask a sensor question:",
    value=st.session_state.get(
        "question",
        ""
    ),
    placeholder=(
        "e.g. Which rooms were unusually warm?"
    )
)


ask_button = st.button(
    "🔍 Ask Sensors",
    type="primary",
    use_container_width=True
)


# ============================================================
# QUERY TYPE LABELS
# ============================================================

QUERY_LABELS = {

    "average_temperature":
        "Average Temperature",

    "absolute_temperature":
        "Highest Temperature",

    "high_temperature":
        "High Temperature Anomaly",

    "occupancy":
        "Occupancy / PIR Activity",

    "absolute_co2":
        "Highest CO₂",

    "high_co2":
        "High CO₂ Anomaly",

    "absolute_light":
        "Highest Light Level",

    "high_light":
        "High Light Anomaly",

    "semantic":
        "Semantic Sensor Search"
}


SOURCE_LABELS = {

    "sqlite":
        "SQLite",

    "qdrant":
        "Qdrant",

    "hybrid":
        "SQLite + Qdrant"
}


# ============================================================
# PROCESS QUERY
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            # ------------------------------------------------
            # QUERY
            # ------------------------------------------------

            with st.spinner(
                "Querying sensor database..."
            ):

                result = execute_query(
                    question
                )

            # ------------------------------------------------
            # EVIDENCE
            # ------------------------------------------------

            evidence = format_evidence(
                result
            )

            # ------------------------------------------------
            # LLM
            # ------------------------------------------------

            with st.spinner(
                "Generating answer..."
            ):

                answer = ask_ollama(
                    question,
                    evidence,
                    result["query_type"]
                )


            # =================================================
            # ANSWER
            # =================================================

            st.subheader("💡 Answer")

            st.markdown(answer)


            st.divider()


            # =================================================
            # QUERY INFORMATION
            # =================================================

            st.subheader(
                "🔎 Query Information"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Data Source",
                    SOURCE_LABELS.get(
                        result["source"],
                        result["source"]
                    )
                )

            with col2:

                st.metric(
                    "Query Type",
                    QUERY_LABELS.get(
                        result["query_type"],
                        result["query_type"]
                    )
                )


            # =================================================
            # STRUCTURED RESULTS
            # =================================================

            if "structured_results" in result:

                st.divider()

                st.subheader(
                    "📊 Structured Database Results"
                )

                structured = result[
                    "structured_results"
                ]

                if structured:

                    st.dataframe(
                        structured,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No structured results found."
                    )


            # =================================================
            # SEMANTIC RESULTS
            # =================================================

            if "semantic_results" in result:

                st.divider()

                st.subheader(
                    "🧠 Semantic Search Results"
                )

                semantic = result[
                    "semantic_results"
                ]

                if semantic:

                    for i, item in enumerate(
                        semantic,
                        start=1
                    ):

                        room = item.get(
                            "room_id",
                            "Unknown"
                        )

                        with st.expander(
                            f"Result {i} — Room {room}"
                        ):

                            col1, col2 = st.columns(2)

                            with col1:

                                score = item.get(
                                    "score"
                                )

                                if score is not None:

                                    st.write(
                                        f"**Similarity score:** "
                                        f"{score:.4f}"
                                    )

                                st.write(
                                    f"**Event ID:** "
                                    f"{item.get('event_id')}"
                                )

                            with col2:

                                st.write(
                                    f"**Reasons:** "
                                    f"{item.get('reasons')}"
                                )

                                st.write(
                                    f"**Time:** "
                                    f"{item.get('event_start')} "
                                    f"→ "
                                    f"{item.get('event_end')}"
                                )

                            st.markdown(
                                "**Semantic Description**"
                            )

                            st.write(
                                item.get(
                                    "semantic_text",
                                    ""
                                )
                            )

                else:

                    st.info(
                        "No semantic results found."
                    )


        except Exception as e:

            st.error(
                f"Error while processing query: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Ask the Sensors | "
    "KETI Smart-Building Dataset | "
    "SQLite + Qdrant + SentenceTransformer + Qwen3 8B"
)