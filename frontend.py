import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Notification Priority Monitor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Notification Priority Monitor")
st.write("Real-time notification priority prediction data")
st.markdown("---")

FASTAPI_URL = "http://127.0.0.1:8000/get_model_data"

try:
    response = requests.get(FASTAPI_URL, timeout=3)
    
    if response.status_code == 200:
        data = response.json()
        
        # Display data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="App Name",
                value=data.get("app_name", "N/A")
            )
        
        with col2:
            st.metric(
                label="Priority Score",
                value=f"{data.get('final_prediction', 0):.4f}"
            )
        
        with col3:
            st.metric(
                label="Status",
                value="Active"
            )
        
        st.markdown("---")
        st.subheader("Notification Text")
        st.write(data.get("notification_text", "No text available"))
        
        st.markdown("---")
        st.subheader("Raw JSON Response")
        st.json(data)
        
    else:
        st.error(f"Server error: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to FastAPI server. Make sure it's running on http://127.0.0.1:8000")
except Exception as e:
    st.error(f"Error: {str(e)}")
