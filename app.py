import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="BMS Research Dashboard", layout="wide")
st.title("🛡️ Advanced Battery Thermal Management System")
st.markdown(f"**Researcher:** CHAMIKA SANKALPA | **Project:** Battery Thermal Management Research")

# 2. Load Model & Scalers (අලුත්ම ක්‍රමයට)
@st.cache_resource
def load_assets():
    # Deserialization Error එක මගහැරීමට සරලවම model එක load කිරීම
    model = tf.keras.models.load_model('bms_final_lstm_model.h5', compile=False)
    scaler_X = joblib.load('scaler_X_final.pkl')
    scaler_y = joblib.load('scaler_y_final.pkl')
    return model, scaler_X, scaler_y

# Error handling එකක් දාලා තියෙන්නේ site එක crash වෙන්නේ නැතිවෙන්න
try:
    model, scaler_X, scaler_y = load_assets()
    
    # 3. Sidebar Inputs
    st.sidebar.header("🕹️ Real-time Parameters")
    v_in = st.sidebar.slider("Voltage Input (V)", 10.0, 15.0, 12.8, 0.1)
    i_in = st.sidebar.slider("Current Input (A)", 0.0, 5.0, 1.5, 0.1)

    # 4. Session State for History
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=['Time', 'Voltage', 'Current', 'Temp'])

    # 5. Prediction Logic
    power = v_in * i_in
    # මූලික input array එක සෑදීම
    input_features = np.array([[v_in, i_in, 0.01, 0.01, power]])
    
    # Scaler එක පාවිච්චි කරලා transform කිරීම
    input_scaled = scaler_X.transform(input_features)
    
    # LSTM එකට අවශ්‍ය (1, 30, 5) shape එකට සකස් කිරීම
    input_seq = np.zeros((1, 30, 5))
    input_seq[0, :, :] = input_scaled

    # Prediction කිරීම
    prediction = model.predict(input_seq, verbose=0)
    temp_res = float(scaler_y.inverse_transform(prediction)[0][0])

    # History Update
    new_data = pd.DataFrame({
        'Time': [datetime.now().strftime("%H:%M:%S")], 
        'Voltage': [v_in], 
        'Current': [i_in], 
        'Temp': [temp_res]
    })
    st.session_state.history = pd.concat([st.session_state.history, new_data]).tail(10)

    # 6. Dashboard Metrics Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Voltage (V)", f"{v_in} V")
    with col2:
        st.metric("Current (A)", f"{i_in} A")
    with col3:
        status_color = "normal" if temp_res < 40 else "inverse"
        st.metric("Predicted Temp", f"{temp_res:.2f} °C", delta=f"{temp_res-30:.2f} °C", delta_color=status_color)

    st.markdown("---")

    # 7. Visuals
    chart_col, gauge_col = st.columns([2, 1])
    with chart_col:
        st.subheader("📈 Temperature Trend (Real-time)")
        st.line_chart(st.session_state.history.set_index('Time')['Temp'])

    with gauge_col:
        st.subheader("🌡️ Heat Level Gauge")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = temp_res,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Celsius"},
            gauge = {
                'axis': {'range': [None, 60]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 35], 'color': "lightgreen"},
                    {'range': [35, 45], 'color': "orange"},
                    {'range': [45, 60], 'color': "red"}],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 40}
            }))
        st.plotly_chart(fig, use_container_width=True)

    if temp_res > 40:
        st.error("🚨 CRITICAL ALERT: Battery temperature exceeding safe limits!")

except Exception as e:
    st.error(f"System Initializing... Please wait or Refresh. (Detail: {e})")
