@"
# Ask the Sensors

An embedding-based sensor data retrieval and local LLM system.

## Architecture

Sensor Data
    ↓
Preprocessing
    ↓
SQLite + Qdrant
    ↓
Semantic Retrieval
    ↓
Local LLM
    ↓
Streamlit Dashboard
"@ | Out-File -Encoding utf8 README.md