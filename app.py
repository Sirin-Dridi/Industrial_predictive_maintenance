import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS (Cornflower Blue, Dark Plum Text, Mustard Boxes & Sidebar)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main App Background (Cornflower Blue) */
    .stApp {
        background-color: #75A2EA;
    }
    
    /* Header & Main Titles (Dark Chocolate Brown) */
    .main-title {
        color: #372118;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #372118;
        font-size: 1.05rem;
        font-weight: 600;
        opacity: 0.9;
        margin-bottom: 1.5rem;
    }
    
    /* Primary Action Buttons (Muted Orange with Mustard Hover) */
    .stButton>button {
        background-color: #E86A33;
        color: #FFFFFF;
        font-weight: bold;
        font-size: 1.1rem;
        border-radius: 6px;
        border: none;
        padding: 0.65rem 1.2rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #D99B26; /* Warm Mustard Hover */
        color: #372118;
        box-shadow: 0 4px 10px rgba(55, 33, 24, 0.2);
    }

    /* Metric Cards: Bold Numbers & Labels */
    [data-testid="stMetricValue"] {
        color: #372118 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #22120C !important;
        font-weight: 700 !important;
    }

    /* Sidebar Background & Text Styling */
    [data-testid="stSidebar"] {
        background-color: #D99B26 !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
        color: #372118 !important;
    }

    /* Progress Bar (Dull Yellow / Mustard Accent) */
    .stProgress > div > div > div > div {
        background-color: #D99B26;
    }
    
    /* Custom Alert / Notification Rectangles (Background set to #D99B26) */
    .stAlert, [data-testid="stNotification"] {
        background-color: #D99B26 !important;
        color: #3B1F2B !important;
        border: 1px solid #372118 !important;
    }
    .stAlert p, .stAlert div, [data-testid="stNotification"] p {
        color: #3B1F2B !important;
        font-weight: 700 !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F5EE;
        color: #372118;
        font-weight: bold;
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #D99B26 !important;
        color: #3B1F2B !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD ARTIFACTS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/random_forest_pdm.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception:
    st.error("Model artifacts missing. Ensure trained .pkl files exist inside the models/ folder.")
    st.stop()

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Machine Health Monitoring & Diagnostics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Sensor Telemetry & Predictive Risk Analysis | Industrial AI</div>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------------------------------------------------------
# SIDEBAR & PRESET SIMULATOR
# -----------------------------------------------------------------------------
st.sidebar.markdown("<h2>Telemetry Controls</h2>", unsafe_allow_html=True)

preset = st.sidebar.selectbox(
    "Quick Test Presets:",
    ["Custom Inputs", "Normal Operation", "High Tool Wear Warning", "Overheating Danger"]
)

if preset == "Normal Operation":
    def_air, def_proc, def_speed, def_torque, def_wear = 300.0, 310.0, 1500, 38.0, 45
elif preset == "High Tool Wear Warning":
    def_air, def_proc, def_speed, def_torque, def_wear = 300.0, 311.5, 1400, 52.0, 215
elif preset == "Overheating Danger":
    def_air, def_proc, def_speed, def_torque, def_wear = 305.0, 320.0, 2800, 65.0, 180
else:
    def_air, def_proc, def_speed, def_torque, def_wear = 300.0, 310.0, 1500, 40.0, 100

air_temp = st.sidebar.number_input("Air Temperature [K]", value=def_air, step=0.5)
proc_temp = st.sidebar.number_input("Process Temperature [K]", value=def_proc, step=0.5)
speed = st.sidebar.number_input("Rotational Speed [rpm]", value=def_speed, step=25)
torque = st.sidebar.number_input("Torque [Nm]", value=def_torque, step=1.0)
tool_wear = st.sidebar.number_input("Tool Wear [min]", value=def_wear, step=5)

# Feature Calculations
power_kw = (torque * (speed * (2 * 3.14159 / 60))) / 1000.0
temp_diff = proc_temp - air_temp

# -----------------------------------------------------------------------------
# MODEL INFERENCE
# -----------------------------------------------------------------------------
raw_inputs = np.array([[air_temp, proc_temp, speed, torque, tool_wear, power_kw * 1000, temp_diff]])
scaled_inputs = scaler.transform(raw_inputs)

prediction = model.predict(scaled_inputs)[0]
failure_prob = model.predict_proba(scaled_inputs)[0][1]

# -----------------------------------------------------------------------------
# DASHBOARD LAYOUT
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<h3 style='color: #372118;'>Operational Telemetry</h3>", unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Air Temperature", f"{air_temp:.1f} K")
        st.metric("Process Temperature", f"{proc_temp:.1f} K", delta=f"{temp_diff:.1f} K Diff")
        st.metric("Tool Wear", f"{tool_wear} min")
    with m2:
        st.metric("Rotational Speed", f"{speed} RPM")
        st.metric("Torque", f"{torque:.1f} Nm")
        st.metric("Calculated Power", f"{power_kw:.2f} kW")

with col2:
    st.markdown("<h3 style='color: #372118;'>Diagnostic Results</h3>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='color: #372118; font-weight: bold;'>Estimated Failure Risk Level: {failure_prob:.1%}</p>", unsafe_allow_html=True)
    st.progress(float(failure_prob))
    
    if prediction == 1:
        st.error("CRITICAL ALERT: Machine Failure Imminent!")
        st.warning("Action Required: Schedule immediate tool replacement and maintenance check.")
    else:
        st.success("NOMINAL: System Operating Normally")
        st.info("Status: Operational metrics sit within safe parameters.")

# -----------------------------------------------------------------------------
# ADVANCED ANALYTICS & GRAPH SECTION
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("<h3 style='color: #372118;'>Diagnostic Visualization & Trend Analysis</h3>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([" Metric Scale Comparison", " 24-Hour Telemetry Trend", " Operating Limits vs Current"])

with tab1:
    chart_data = pd.DataFrame({
        'Parameter': ['Air Temp (K)', 'Process Temp (K)', 'Speed (RPM/10)', 'Torque (Nm)', 'Tool Wear (min)'],
        'Value': [air_temp, proc_temp, speed / 10, torque, tool_wear]
    }).set_index('Parameter')
    
    st.bar_chart(chart_data)

with tab2:
    # Generate simulated 24-hour historical telemetry data ending at the current inputs
    hours = [f"-{i}h" for i in range(24, 0, -1)] + ["Now"]
    
    # Simulating gradual buildup towards current user input parameters
    hist_temp = np.linspace(proc_temp - 5.0, proc_temp, 25) + np.random.normal(0, 0.2, 25)
    hist_wear = np.linspace(max(0, tool_wear - 120), tool_wear, 25)
    hist_torque = np.linspace(torque - 8.0, torque, 25) + np.random.normal(0, 0.5, 25)
    
    trend_df = pd.DataFrame({
        'Time': hours,
        'Process Temp (K)': hist_temp,
        'Tool Wear (min)': hist_wear,
        'Torque (Nm)': hist_torque
    }).set_index('Time')
    
    st.line_chart(trend_df)

with tab3:
    # Threshold stress ratios (Normalized between 0% and 100% of maximum recommended limit)
    # Recommended Limits: Temp Diff = 12K, Speed = 2800 RPM, Torque = 60 Nm, Wear = 200 min
    stress_levels = pd.DataFrame({
        'Metric': ['Temp Difference', 'Rotational Speed', 'Torque Stress', 'Tool Wear Stress'],
        'Stress Load (%)': [
            min(100.0, (temp_diff / 12.0) * 100),
            min(100.0, (speed / 2800.0) * 100),
            min(100.0, (torque / 60.0) * 100),
            min(100.0, (tool_wear / 200.0) * 100)
        ]
    }).set_index('Metric')
    
    st.bar_chart(stress_levels)