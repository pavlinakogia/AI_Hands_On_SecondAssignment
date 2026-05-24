# Homework 2: Making Your Model Talk

**Course:** AI Hands-on, NTUA  
**Domain:** Weather Prediction (Rain in Australia)  
**Name:** Pavlina Kogia  
**ID:** 09325011  

---

## 1. System Overview

This project builds a domain-aware conversational AI agent that can:
- Answer factual questions about weather and rainfall using RAG
- Predict whether it will rain tomorrow using the trained model from HW1
- Provide summary statistics from the training dataset

The agent is exposed via a REST API built with FastAPI.

---

## 2. Architecture

The agent is built with LangGraph's `create_react_agent` and has access to three tools:

| Tool | Description |
|------|-------------|
| `retrieval_tool` | Searches the knowledge base using RAG to answer conceptual questions about weather and rainfall |
| `prediction_tool` | Uses the HW1 Neural Network model to predict RainTomorrow given weather measurements |
| `dataset_stats_tool` | Returns summary statistics for any numeric column in the training dataset |

The agent autonomously decides which tool to call based on the user's message. Conversation memory is maintained per session using a dictionary keyed by `session_id`.

---

## 3. Knowledge Base

Five documents were collected and stored in `data/documents/`:

| File | Topic |
|------|-------|
| `rain_prediction.txt` | Weather forecasting methods |
| `humidity_pressure.txt` | Humidity and atmospheric pressure |
| `australia_climate.txt` | Climate of Australia |
| `ml_weather.txt` | Machine learning in weather forecasting |
| `rain_factors.txt` | Factors that cause rainfall |

Documents were chunked (size=500, overlap=50) and embedded using
`sentence-transformers/all-MiniLM-L6-v2`. The vector store is persisted
with ChromaDB in `data/vector_store/`.

---

## 4. HW1 Model Integration

The best model from HW1 was a **Neural Network** (PyTorch) with ROC-AUC of 0.8609.

The `prediction_tool` applies the same preprocessing pipeline as HW1:
- Feature engineering: TempRange, HumidityDiff, PressureDiff
- RainToday binary encoding
- One-Hot Encoding for categorical features
- StandardScaler (loaded from `models/scaler.pkl`)

Input fields required:
`Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir,
WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm,
Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm,
Temp9am, Temp3pm, RainToday, Month`

---

## 5. Example Conversations

### RAG Retrieval
**User:** What factors cause rainfall in Australia?  
**Agent:** Rainfall in Australia is influenced by several factors, including
the Australian monsoon in northern parts of the country. The El Niño-Southern
Oscillation also plays a significant role, leading to extensive droughts and
wet periods.

### Prediction
**User:** Will it rain tomorrow in Albury with MinTemp=10.5, MaxTemp=22.3,
Humidity9am=60, Humidity3pm=45, RainToday=No, Month=6 (and other fields)?  
**Agent:** The prediction for tomorrow in Albury is no rain,
with a probability of 17.7%.

---

## 6. Installation & Execution

```bash
# 1. Clone the repository
git clone https://github.com/pavlinakogia/AI_hands_on_SecondAssignment
cd AI_hands_on_SecondAssignment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with your API key
echo GOOGLE_API_KEY=your_key_here > .env

# 4. Start the FastAPI server
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

---

## 7. Example API Call

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
```

Or with curl:
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Will it rain tomorrow?", "session_id": "user_001"}'
```