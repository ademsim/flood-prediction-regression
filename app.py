import re
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Flood Risk Predictor")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "flood_model.pkl", "rb"))


model = load_model()
features = list(model.feature_names_in_)

st.title("Flood Risk Predictor")
st.write(
    "A linear regression model (trained on the Kaggle Flood Prediction dataset, Playground Series S4E5) estimates "
    "the probability of flooding from 20 risk factors. Each factor is scored from 0 (no risk) to 16 (very high risk)."
)

values = {}
col1, col2 = st.columns(2)
for i, name in enumerate(features):
    label = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    with (col1 if i % 2 == 0 else col2):
        values[name] = st.slider(label, 0, 16, 5)

if st.button("Predict flood risk"):
    X = pd.DataFrame([values])[features]
    proba = float(np.clip(model.predict(X)[0], 0, 1))
    if proba >= 0.6:
        st.error(f"High flood risk: **{proba:.1%}**")
    elif proba >= 0.45:
        st.warning(f"Moderate flood risk: **{proba:.1%}**")
    else:
        st.success(f"Low flood risk: **{proba:.1%}**")
    st.progress(proba)

st.caption(
    "Model: Linear Regression (validation R² ≈ 0.845 on the Kaggle data). In this synthetic dataset the flood "
    "probability grows almost linearly with the sum of all factor scores, so a simple linear model works well."
)
