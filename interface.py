import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP & TRAIN MODEL (The "Standard" Logic) ---
@st.cache_resource
def train_model():
    # Load Data
    # Ensure this CSV is in the same folder!
    df = pd.read_csv('venezuela_wdi_indicators.csv').sort_values('year')
    
    # Feature Engineering (Standard)
    # Note: No Inflation data is used here
    df['gdp_growth_1'] = df['gdp_growth_pct'].shift(1)
    df['gdp_roll3_valid'] = df['gdp_growth_1'].rolling(window=3).mean()
    
    # Sanctions Logic
    df['sanctions_score'] = df['year'].apply(lambda y: 0.2 if y >= 2024 else (1.0 if 2019 <= y < 2023 else (0.5 if 2017 <= y < 2019 else 0.0)))
    
    features = ['gdp_growth_1', 'gdp_roll3_valid', 'sanctions_score']
    df_final = df.dropna(subset=features + ['gdp_growth_pct']).copy()
    
    # Train Voting Regressor
    X = df_final[features]
    y = df_final['gdp_growth_pct']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Voting Model: Linear Regression (Trend) + Random Forest (Pattern Memory)
    model_lr = LinearRegression()
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model = VotingRegressor(estimators=[('lr', model_lr), ('rf', model_rf)], weights=[1, 1])
    model.fit(X_scaled, y)
    
    return model, scaler, features

model, scaler, feature_names = train_model()

# --- 2. THE DASHBOARD UI ---
st.title("🇻🇪 Venezuela GDP Forecaster")
st.caption("Model: Voting Regressor (RF + LR) | Features: Lag, Trend, Sanctions")

# Sidebar Controls
st.sidebar.header("Economic Scenarios")

# Input 1: Momentum
lag_input = st.sidebar.slider(
    "Previous Year Growth (%)", 
    min_value=-10.0, max_value=15.0, value=5.30, 
    help="GDP Growth in 2024"
)

# Input 2: Trend
trend_input = st.sidebar.slider(
    "3-Year Average (%)", 
    min_value=-10.0, max_value=15.0, value=5.77,
    help="Average growth of 2022-2024"
)

# Input 3: Sanctions
sanctions_input = st.sidebar.selectbox(
    "Sanctions Environment", 
    options=[0.0, 0.2, 0.5, 0.7, 1.0], 
    index=1, # Default to 0.2
    format_func=lambda x: f"Score {x} (Low/Boom)" if x==0.2 else (f"Score {x} (Strict)" if x==1.0 else f"Score {x}")
)

# --- 3. PREDICT ---
# We intentionally exclude Inflation here
input_data = pd.DataFrame([[lag_input, trend_input, sanctions_input]], columns=feature_names)
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# --- 4. DISPLAY RESULTS ---
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Forecast for 2025")
    st.metric(label="GDP Growth", value=f"{prediction:.2f}%", delta_color="normal")

with col2:
    st.write("### Analysis")
    if prediction < 0:
        st.error("📉 Recession")
        st.write("The model predicts a downturn, likely driven by historical volatility patterns.")
    elif prediction < 2.0:
        st.warning("⚠️ Stagnation")
        st.write("Growth is positive but weak. The Random Forest component is being cautious.")
    else:
        st.success("🚀 Boom")
        st.write("Strong recovery predicted.")

# Debugging / Explainability
with st.expander("Why is this result different from SVR?"):
    st.write("""
    **The Voting Model (Random Forest)** relies heavily on memory. 
    It looks at the inputs (5.3% growth, Low Sanctions) and compares them to similar years in the past. 
    Since Venezuela often crashed after growth spurts in the past, this model is naturally more pessimistic 
    than the SVR, which focuses on the 'slope' of recovery.
    """)
    