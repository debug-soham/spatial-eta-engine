import os
import sys
# Add parent directory to path to enable importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# Imports from src
from src.spatial_features import coord_to_h3, get_hexagon_geojson
from src.explainability import explain_prediction
from src.dataset import haversine_distance

# Set Streamlit Page Config
st.set_page_config(
    page_title="Spatial-Temporal ETA Simulator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #ffffff;
    }
    div[data-testid="stSidebar"] {
        background-color: #161920;
    }
    .kpi-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #8a8d98;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00ffd0;
    }
    .delay-alert-high {
        background: rgba(239, 85, 59, 0.15);
        border: 1px solid #ef553b;
        color: #ff765e;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 15px;
    }
    .delay-alert-low {
        background: rgba(0, 255, 208, 0.1);
        border: 1px solid #00ffd0;
        color: #00ffd0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 15px;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE COORDINATE SETUP -----------------
# Prepopulate with a typical NYC route (restaurant in Midtown to building in Chelsea)
if 'pickup_lat' not in st.session_state:
    st.session_state.pickup_lat = 40.7589
if 'pickup_lng' not in st.session_state:
    st.session_state.pickup_lng = -73.9851
if 'dropoff_lat' not in st.session_state:
    st.session_state.dropoff_lat = 40.7440
if 'dropoff_lng' not in st.session_state:
    st.session_state.dropoff_lng = -74.0010
if 'active_target' not in st.session_state:
    st.session_state.active_target = 'Pickup'

# ----------------- SIDEBAR INTERACTIVE INPUTS -----------------
st.sidebar.title("Simulation Controls")
st.sidebar.markdown("Use the parameters below to control the operational environment.")

st.sidebar.subheader("Temporal Settings")
hour_input = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 12, step=1)
dow_input = st.sidebar.selectbox(
    "Day of Week",
    options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    index=0
)

st.sidebar.subheader("Operational Metrics")
prep_time = st.sidebar.slider("Merchant Prep Time (mins)", 5, 45, 15)
driver_ratio = st.sidebar.slider("Active Driver Density (Drivers / Orders)", 0.1, 2.0, 1.0, step=0.1)

st.sidebar.subheader("Weather Settings")
rain_level = st.sidebar.slider("Rain Level (mm/hr)", 0.0, 15.0, 0.0, step=0.5)

# Convert inputs to inference timestamp representation
now = datetime.now()
dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
target_day = now + timedelta(days=(dow_map[dow_input] - now.weekday()))
order_datetime = datetime(target_day.year, target_day.month, target_day.day, hour_input, 0, 0)
order_timestamp_str = order_datetime.strftime("%Y-%m-%d %H:%M:%S")

# ----------------- DASHBOARD BODY -----------------
st.title("Spatial-Temporal Delivery Delay & ETA Engine")
st.markdown("An interactive simulation platform utilizing spatial hexagon indexing (H3), cyclic time encodings, and gradient-boosted trees.")

col_map, col_metrics = st.columns([3, 2])

# ----------------- MAP VIEWPORT & CLICK HANDLER -----------------
with col_map:
    st.subheader("Route Simulator Map")
    st.markdown("Select an active target in the toggle below, then click on the map to set coordinates.")
    
    # Toggle Active Selection target
    active_col1, active_col2 = st.columns(2)
    with active_col1:
        if st.button("Set Pickup Coordinates (Red Marker)", type="primary" if st.session_state.active_target == 'Pickup' else "secondary"):
            st.session_state.active_target = 'Pickup'
            st.rerun()
    with active_col2:
        if st.button("Set Dropoff Coordinates (Blue Marker)", type="primary" if st.session_state.active_target == 'Dropoff' else "secondary"):
            st.session_state.active_target = 'Dropoff'
            st.rerun()

    st.caption(f"Currently setting **{st.session_state.active_target}** coords by clicking on the map below.")
    
    # Draw Folium Map
    center_lat = (st.session_state.pickup_lat + st.session_state.dropoff_lat) / 2
    center_lng = (st.session_state.pickup_lng + st.session_state.dropoff_lng) / 2
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles="cartodbpositron")
    
    # Draw Markers
    folium.Marker(
        location=[st.session_state.pickup_lat, st.session_state.pickup_lng],
        popup="Pickup",
        icon=folium.Icon(color="red", icon="cutlery", prefix="fa")
    ).add_to(m)
    
    folium.Marker(
        location=[st.session_state.dropoff_lat, st.session_state.dropoff_lng],
        popup="Dropoff",
        icon=folium.Icon(color="blue", icon="home", prefix="fa")
    ).add_to(m)
    
    # Draw Dashed connection line
    folium.PolyLine(
        locations=[
            [st.session_state.pickup_lat, st.session_state.pickup_lng],
            [st.session_state.dropoff_lat, st.session_state.dropoff_lng]
        ],
        color="#7f8c8d",
        weight=2.5,
        dash_array="5, 10"
    ).add_to(m)

    # Calculate and draw H3 Hexagons
    try:
        p_h3 = coord_to_h3(st.session_state.pickup_lat, st.session_state.pickup_lng, 8)
        d_h3 = coord_to_h3(st.session_state.dropoff_lat, st.session_state.dropoff_lng, 8)
        
        # Add GeoJSON overlays for H3 hexagons
        p_geojson = get_hexagon_geojson(p_h3, properties={"style": {"color": "#ef553b", "fillColor": "#ef553b", "fillOpacity": 0.2}})
        d_geojson = get_hexagon_geojson(d_h3, properties={"style": {"color": "#636efa", "fillColor": "#636efa", "fillOpacity": 0.2}})
        
        folium.GeoJson(
            p_geojson, 
            style_function=lambda x: {"color": "#e74c3c", "weight": 2, "fillColor": "#e74c3c", "fillOpacity": 0.15}
        ).add_to(m)
        
        folium.GeoJson(
            d_geojson, 
            style_function=lambda x: {"color": "#2980b9", "weight": 2, "fillColor": "#2980b9", "fillOpacity": 0.15}
        ).add_to(m)
    except Exception as ex:
        st.warning(f"Unable to render H3 overlay: {ex}")

    # Capture clicks
    map_data = st_folium(m, height=450, width=700, key="map")
    
    # Process click capture
    if map_data and map_data.get("last_clicked"):
        click = map_data["last_clicked"]
        lat, lng = click["lat"], click["lng"]
        
        if 'last_click_seen' not in st.session_state or st.session_state.last_click_seen != click:
            st.session_state.last_click_seen = click
            
            if st.session_state.active_target == 'Pickup':
                st.session_state.pickup_lat = lat
                st.session_state.pickup_lng = lng
                st.session_state.active_target = 'Dropoff'  # smart toggle to Dropoff
            else:
                st.session_state.dropoff_lat = lat
                st.session_state.dropoff_lng = lng
                st.session_state.active_target = 'Pickup'  # toggle back to Pickup
            
            st.rerun()

# ----------------- INFERENCE ENGINE -----------------
# 1. Attempt REST API call
# 2. Fallback to Local joblib prediction
api_url = "http://127.0.0.1:8000/predict_eta"
payload_req = {
    "pickup_latitude": st.session_state.pickup_lat,
    "pickup_longitude": st.session_state.pickup_lng,
    "dropoff_latitude": st.session_state.dropoff_lat,
    "dropoff_longitude": st.session_state.dropoff_lng,
    "order_timestamp": order_timestamp_str,
    "merchant_prep_time_min": float(prep_time),
    "active_driver_ratio": float(driver_ratio),
    "rain_level_mm_hr": float(rain_level)
}

prediction_success = False
predicted_eta = 0.0
delay_prob = 0.0
is_high_risk = False
attributions = {}
distance_km = 0.0
pickup_cell_tag = ""
dropoff_cell_tag = ""

with st.spinner("Calculating route metrics..."):
    try:
        response = requests.post(api_url, json=payload_req, timeout=1.5)
        if response.status_code == 200:
            res_data = response.json()
            predicted_eta = res_data["predicted_eta_min"]
            delay_prob = res_data["delay_probability"]
            is_high_risk = res_data["is_high_delay_risk"]
            distance_km = res_data["distance_km"]
            pickup_cell_tag = res_data["pickup_h3_cell"]
            dropoff_cell_tag = res_data["dropoff_h3_cell"]
            attributions = res_data["explainability"]["attributions_min"]
            prediction_success = True
            st.sidebar.success("Connected to Live FastAPI Engine")
    except Exception:
        # Fallback to local load
        model_path = "models/eta_pipeline.joblib"
        if os.path.exists(model_path):
            st.sidebar.warning("API offline. Running local inference fallback.")
            try:
                pipeline_payload = joblib.load(model_path)
                preprocessor = pipeline_payload['preprocessor']
                regressor = pipeline_payload['regressor']
                classifier = pipeline_payload['classifier']
                
                # Math distance
                distance_km = haversine_distance(
                    st.session_state.pickup_lat, st.session_state.pickup_lng,
                    st.session_state.dropoff_lat, st.session_state.dropoff_lng
                )
                distance_km = float(np.clip(distance_km, 0.2, 15.0))
                
                pickup_cell_tag = coord_to_h3(st.session_state.pickup_lat, st.session_state.pickup_lng, 8)
                dropoff_cell_tag = coord_to_h3(st.session_state.dropoff_lat, st.session_state.dropoff_lng, 8)
                
                input_df = pd.DataFrame([{
                    'pickup_latitude': st.session_state.pickup_lat,
                    'pickup_longitude': st.session_state.pickup_lng,
                    'dropoff_latitude': st.session_state.dropoff_lat,
                    'dropoff_longitude': st.session_state.dropoff_lng,
                    'order_timestamp': order_timestamp_str,
                    'distance_km': distance_km,
                    'merchant_prep_time_min': float(prep_time),
                    'active_driver_ratio': float(driver_ratio),
                    'rain_level_mm_hr': float(rain_level)
                }])
                
                input_processed = preprocessor.transform(input_df)
                predicted_eta = float(regressor.predict(input_processed)[0])
                predicted_eta = max(predicted_eta, 5.0)
                
                delay_prob = float(classifier.predict_proba(input_processed)[0, 1])
                is_high_risk = delay_prob >= 0.50 or predicted_eta > 35.0
                
                exp_payload = explain_prediction(pipeline_payload, input_df)
                attributions = exp_payload["attributions_min"]
                prediction_success = True
            except Exception as e_local:
                st.sidebar.error(f"Local inference error: {e_local}")
        else:
            st.sidebar.error("Pipeline file not found. Please run training first!")

# ----------------- DISPLAY METRICS & ATTRIBUTIONS -----------------
with col_metrics:
    st.subheader("Delivery Estimates")
    
    if prediction_success:
        # Delay Risk alert card
        if is_high_risk:
            st.markdown('<div class="delay-alert-high">WARNING: HIGH DELAY RISK ENCOUNTERED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="delay-alert-low">STATUS: EXPECTED ON TIME</div>', unsafe_allow_html=True)
            
        # KPI Grid
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Predicted ETA</div>
                <div class="kpi-value">{predicted_eta:.1f}m</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Delay Prob</div>
                <div class="kpi-value">{delay_prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        kpi_col3, kpi_col4 = st.columns(2)
        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Distance</div>
                <div class="kpi-value" style="color: #636efa;">{distance_km:.2f} km</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">H3 pickup</div>
                <div class="kpi-value" style="font-size: 1.1rem; color: #f1c40f; padding-top: 15px; font-family: monospace;">
                    {pickup_cell_tag}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Explainability Bar Chart
        if attributions:
            groups = list(attributions.keys())
            values = list(attributions.values())
            
            # Sort for clean display (excluding baseline contribution)
            plot_df = pd.DataFrame({'Feature Group': groups, 'Impact': values})
            plot_df = plot_df[plot_df['Feature Group'] != 'Interactions & Base Overhead'].sort_values(by='Impact', ascending=True)
            
            base_offset = attributions.get('Interactions & Base Overhead', 0.0)
            
            fig_colors = ['#ef553b' if x >= 0 else '#636efa' for x in plot_df['Impact']]
            
            fig = go.Figure(go.Bar(
                x=plot_df['Impact'],
                y=plot_df['Feature Group'],
                orientation='h',
                marker_color=fig_colors,
                text=[f"{val:+.1f} min" for val in plot_df['Impact']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=dict(text=f"Local Feature Attribution (Base baseline: {base_offset:.1f} mins)", font=dict(size=12, color="#ffffff")),
                xaxis=dict(title="ETA impact in minutes", gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#8a8d98")),
                yaxis=dict(tickfont=dict(color="#ffffff")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                height=260
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Set coordinates and adjust sidebar controls. If the system is started for the first time, make sure to execute the model training script.")
