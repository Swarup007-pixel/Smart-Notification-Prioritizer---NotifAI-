

import streamlit as st
import requests
import time

# Host URL configuration
FASTAPI_URL = "http://127.0.0.1:8000/get_model_data"

st.set_page_config(
    page_title="Notification Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Notification Priority Streaming Monitor")
st.write("Live system status interface processing downstream telemetry from your 2-stage inference pipeline.")
st.markdown("---")

# Sidebar Configuration Control Panel
st.sidebar.header("🔄 UI Sync Options")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh Loop", value=True)
refresh_rate = st.sidebar.slider("Sampling Interval (Seconds)", 1, 10, 2)

def fetch_pipeline_state():
    try:
        res = requests.get(FASTAPI_URL, timeout=2)
        if res.status_code == 200:
            return res.json()
        return {"status": "error", "message": f"Server answered with status code: {res.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Failed to establish cross-port network handshake with FastAPI backend server."}

# Data Retrieval execution
payload = fetch_pipeline_state()

if isinstance(payload, dict) and payload.get("status") == "error":
    st.error(payload.get("message"))
    
elif isinstance(payload, dict) and payload.get("status") == "idle":
    st.info(f"⏳ **Pipeline Status:** {payload.get('message')}")
    
else:
    # Safely extract values handling your list-of-dict data format
    data_item = payload[0] if isinstance(payload, list) and len(payload) > 0 else payload
    
    # 1. Top-Level Value Dashboard Matrix Rows
    m1, m2 = st.columns(2)
    with m1:
        prediction_val = data_item.get("final_prediction", 0.0)
        st.metric(label="🎯 Regressor Priority Weight", value=f"{prediction_val:.4f}")
    with m2:
        app_name_val = str(data_item.get("app_name", "Unknown")).upper()
        st.metric(label="📱 Source Application", value=app_name_val)

    st.markdown("---")

    # 2. Content Profile Layout Columns
    st.subheader("📝 Extracted Text Payload")
    text_val = data_item.get("notification_text", "No text provided.")
    st.info(f'"{text_val}"')

    st.markdown("---")
    
    # 3. Complete Payload Inspection Window
    with st.expander("📦 Inspect Raw Network Data Layer JSON"):
        st.json(payload)

# Handle Auto-Refresh System Loops
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()