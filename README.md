# Assignment 2: Making Your Model Talk

**Course:** AI Hands-on, NTUA  
**Domain:** Weather Prediction (Rain in Australia)  
**Name:** Pavlina Kogia  
**ID:** 09325011  

---

## 1. System Overview

This project extends Homework 1 by wrapping the best-performing model in a 
domain-aware conversational AI agent. The agent can:

- Answer factual and conceptual questions about weather, rainfall, and climate 
  in Australia using Retrieval-Augmented Generation (RAG)
- Predict whether it will rain tomorrow given a set of weather measurements, 
  using the Neural Network trained in HW1
- Provide summary statistics from the HW1 training dataset on demand

The agent is built with **LangGraph** and exposed via a **FastAPI** REST API 
with support for both standard and streaming responses.

---

## 2. Architecture

The agent is built with LangGraph's `create_react_agent`. It autonomously 
decides which tool to call (or whether to call any tool) based on the user's 
natural language input — no explicit commands are needed.

### Tools

| Tool | Trigger | Description |
|------|---------|-------------|
| `retrieval_tool` | Conceptual/factual questions | Searches the ChromaDB knowledge base using semantic similarity and returns the top-3 relevant passages as context for the LLM |
| `prediction_tool` | Prediction requests | Applies the full HW1 preprocessing pipeline and runs the Neural Network to predict RainTomorrow |
| `dataset_stats_tool` | Statistics questions | Queries the HW1 dataset and returns descriptive statistics for any numeric column |

### LangGraph Graph Structure

```
User Message
     │
     ▼
  [Agent Node] ──► decides tool
     │
     ├──► [retrieval_tool]
     ├──► [prediction_tool]
     ├──► [dataset_stats_tool]
     └──► (no tool) ──► Final Response
```

### Conversation Memory

Memory is maintained per session using a dictionary keyed by `session_id`. 
All messages — including tool calls and tool results — are stored in the 
session history and passed to the agent on every turn, enabling multi-turn 
conversations with full context.

---

## 3. Knowledge Base

Five documents were collected and stored in `data/documents/`:

| File | Topic | Why chosen |
|------|-------|------------|
| `rain_prediction.txt` | Weather forecasting methods | Core domain knowledge about how rain is predicted |
| `humidity_pressure.txt` | Humidity and atmospheric pressure | Key features in the HW1 dataset |
| `australia_climate.txt` | Climate of Australia | Geographic context for the dataset |
| `ml_weather.txt` | Machine learning in weather forecasting | Background on ML approaches in this domain |
| `rain_factors.txt` | Factors that cause rainfall | Explains the target variable |

### RAG Pipeline

- **Chunking:** `RecursiveCharacterTextSplitter` with `chunk_size=500`, `chunk_overlap=50`
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs locally)
- **Vector store:** ChromaDB, persisted to `data/vector_store/` — built once and loaded on subsequent startups
- **Retrieval:** Top-3 chunks by cosine similarity, concatenated into a single context string

Example questions the RAG system can answer:
- "What causes droughts in Australia?"
- "How does humidity affect rainfall?"
- "What is the El Niño-Southern Oscillation?"
- "How is machine learning used in weather forecasting?"

---

## 4. HW1 Model Integration

The best model from HW1 was a **PyTorch Neural Network** with:
- Architecture: Linear(64) → ReLU → Dropout(0.3) → Linear(32) → ReLU → Linear(1) → Sigmoid
- ROC-AUC: 0.8609
- F1 Score: 0.6351

The `prediction_tool` replicates the **exact** HW1 preprocessing pipeline:

1. Feature engineering: `TempRange = MaxTemp - MinTemp`, `HumidityDiff = Humidity3pm - Humidity9am`, `PressureDiff = Pressure3pm - Pressure9am`
2. Binary encoding: `RainToday` mapped to 0/1
3. One-Hot Encoding for all categorical features
4. Column alignment with `scaler.feature_names_in_`
5. StandardScaler transformation using the saved `models/scaler.pkl`

The tool accepts input as a JSON string with the following fields:

```
Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am,
WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, Month
```

---

## 5. Example Conversations

### Example 1 — RAG Retrieval

**User:** What factors cause rainfall in Australia?

**Agent:** Rainfall in Australia is influenced by several factors, including 
the Australian monsoon in northern parts of the country. The El Nino-Southern 
Oscillation also plays a significant role, leading to extensive droughts and 
wet periods.

---

### Example 2 — Prediction + Memory

**User:** Will it rain tomorrow in Albury with MinTemp=10.5, MaxTemp=22.3, 
Rainfall=0.0, Evaporation=5.0, Sunshine=8.0, WindGustDir=NW, WindGustSpeed=35, 
WindDir9am=N, WindDir3pm=NW, WindSpeed9am=10, WindSpeed3pm=20, Humidity9am=60, 
Humidity3pm=45, Pressure9am=1015.0, Pressure3pm=1012.0, Cloud9am=3, Cloud3pm=4, 
Temp9am=14.5, Temp3pm=20.8, RainToday=No, Month=6

**Agent:** The prediction for tomorrow in Albury is no rain, with a probability 
of rain at 17.7%.

**User:** What was the rain probability you just predicted?

**Agent:** The probability of rain was 17.7%.

---

### Example 3 — Dataset Statistics

**User:** What is the average humidity at 3pm in the dataset?

**Agent:** The average humidity at 3pm in the dataset is 52.78.

---

## 6. Installation & Execution

```bash
# 1. Clone the repository
git clone https://github.com/pavlinakogia/AI_Hands_On_SecondAssignment
cd AI_Hands_On_SecondAssignment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
echo GOOGLE_API_KEY=your_api_key_here > .env

# 4. Start the FastAPI server
uvicorn src.api:app --host 127.0.0.1 --port 8000

# 5. Open Swagger UI
# Navigate to: http://127.0.0.1:8000/docs
```

---

## 7. Example API Calls

### Standard chat endpoint

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "message": "What factors cause rainfall in Australia?",
        "session_id": "user_001"
    }
)
print(response.json())
# {"response": "Rainfall in Australia is influenced by..."}
```

### Streaming endpoint

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/chat/stream",
    json={
        "message": "Will it rain tomorrow in Albury?",
        "session_id": "user_001"
    },
    stream=True
)
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
```