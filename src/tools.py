import os
import json
import joblib
import numpy as np
import pandas as pd
import torch  # noqa: F401 - required for pickle to load NNWrapper
from langchain.tools import tool
from src.rag import retrieve
from src.train_neural import NeuralNet, NNWrapper  # noqa: F401 - required for joblib
from dotenv import load_dotenv

load_dotenv()

_model = None
_scaler = None


def load_model():
    global _model, _scaler
    if _model is not None and _scaler is not None:
        return _model, _scaler
    _model = joblib.load('models/best_model.pkl')
    _scaler = joblib.load('models/scaler.pkl')
    return _model, _scaler


@tool
def retrieval_tool(query: str) -> str:
    """Use this tool to answer factual or conceptual questions about
    weather, rainfall, climate, or meteorology. Input is a natural
    language question."""
    return retrieve(query)


@tool
def prediction_tool(input_json: str) -> str:
    """Use this tool to predict whether it will rain tomorrow given
    weather measurements. Input must be a JSON string with these fields:
    Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
    WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am,
    WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
    Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, Month."""
    try:
        model, scaler = load_model()
        data = json.loads(input_json)
        df = pd.DataFrame([data])

        # Feature Engineering (ίδιο με HW1)
        if "MaxTemp" in df.columns and "MinTemp" in df.columns:
            df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
        if "Humidity3pm" in df.columns and "Humidity9am" in df.columns:
            df["HumidityDiff"] = df["Humidity3pm"] - df["Humidity9am"]
        if "Pressure3pm" in df.columns and "Pressure9am" in df.columns:
            df["PressureDiff"] = df["Pressure3pm"] - df["Pressure9am"]
        if "RainToday" in df.columns:
            df["RainToday"] = df["RainToday"].map({'No': 0, 'Yes': 1})

        # One-Hot Encoding
        df = pd.get_dummies(df)

        # Align με τα features του scaler
        expected_features = scaler.feature_names_in_
        df = df.reindex(columns=expected_features, fill_value=0)

        # Scaling
        X_scaled = scaler.transform(df)

        # Prediction
        proba = model.predict_proba(X_scaled)[0]
        prob_rain = proba[1]
        prediction = "Yes" if prob_rain >= 0.5 else "No"

        return (f"Prediction: RainTomorrow = {prediction} "
                f"(probability of rain: {prob_rain * 100:.1f}%)")

    except Exception as e:
        return f"Error making prediction: {str(e)}"


@tool
def dataset_stats_tool(column: str) -> str:
    """Use this tool to get summary statistics about the weather dataset
    used to train the prediction model. Input is a column name as a string.
    Available columns: MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
    WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm,
    Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm."""
    try:
        df = pd.read_csv('data/dataset.csv')

        if column not in df.columns:
            available = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            return f"Column '{column}' not found. Available numeric columns: {available}"

        stats = df[column].describe()
        return (
            f"Statistics for '{column}':\n"
            f"  Count:  {stats['count']:.0f}\n"
            f"  Mean:   {stats['mean']:.2f}\n"
            f"  Std:    {stats['std']:.2f}\n"
            f"  Min:    {stats['min']:.2f}\n"
            f"  Median: {df[column].median():.2f}\n"
            f"  Max:    {stats['max']:.2f}"
        )
    except Exception as e:
        return f"Error fetching stats: {str(e)}"
