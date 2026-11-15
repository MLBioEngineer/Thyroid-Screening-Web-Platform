"""
Interactive Multimodal Thyroid Screening Web Application
Department of Biomedical Engineering, Jashore University of Science and Technology (JUST)
Authors: Sakhawat Hosen Sifat (221901) & Jannatun Khatun (221913)
Supervisors: Dr. Hassan Mohammad Al-Emran | Co-supervisor: Md. Faisal Mina
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from scipy import signal

# Add local path
sys.path.insert(0, os.path.abspath("."))
from src.feature_extractor import (extract_tremor_features, extract_neck_features,
                                   extract_patient_features, SAMPLING_RATE_TREMOR)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="JUST BME | Non-invasive Thyroid Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2b46 0%, #1a4a75 50%, #2980b9 100%);
        padding: 24px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #d1e5f7;
        font-size: 0.98rem;
        margin-bottom: 0px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-normal {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
    }
    .badge-hyper {
        background-color: #f5f3ff;
        border: 2px solid #8b5cf6;
        color: #5b21b6;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
    }
    .badge-hypo {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        color: #92400e;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
    }
    .disclaimer-box {
        background-color: #fff1f2;
        border-left: 5px solid #f43f5e;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #881337;
        margin-top: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model artifacts safely
@st.cache_resource
def load_models_and_metadata():
    models_dir = "models"
    with open(os.path.join(models_dir, "primary_extra_trees.pkl"), "rb") as f:
        primary_et = pickle.load(f)
    with open(os.path.join(models_dir, "ann_top40.pkl"), "rb") as f:
        ann_model = pickle.load(f)
    with open(os.path.join(models_dir, "random_forest.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(models_dir, "svm.pkl"), "rb") as f:
        svm_model = pickle.load(f)
    with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, "feature_columns.json"), "r") as f:
        feature_cols = json.load(f)
    with open(os.path.join(models_dir, "selected_features_top60.json"), "r") as f:
        selected_60 = json.load(f)
    with open(os.path.join(models_dir, "selected_features_top40.json"), "r") as f:
        selected_40 = json.load(f)
    with open(os.path.join(models_dir, "benchmark_results.json"), "r") as f:
        benchmarks = json.load(f)
    with open(os.path.join(models_dir, "selector_60.pkl"), "rb") as f:
        selector_60 = pickle.load(f)
    with open(os.path.join(models_dir, "selector_40.pkl"), "rb") as f:
        selector_40 = pickle.load(f)

    test_cohort = pd.read_csv(os.path.join(models_dir, "heldout_test_set.csv"))

    return {
        "primary_et": primary_et,
        "ann_model": ann_model,
        "rf_model": rf_model,
        "svm_model": svm_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "selected_60": selected_60,
        "selected_40": selected_40,
        "benchmarks": benchmarks,
        "selector_60": selector_60,
        "selector_40": selector_40,
        "test_cohort": test_cohort
    }

# Load assets
try:
    artifacts = load_models_and_metadata()
except Exception as e:
    st.error(f"Error loading trained models: {e}. Please run `python src/model_trainer.py` first.")
    st.stop()

# Header Presentation Banner
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1>Low-Cost and Non-invasive Thyroid Screening Platform</h1>
            <p><strong>Department of Biomedical Engineering, Jashore University of Science and Technology (JUST)</strong></p>
            <p>Researchers: Sakhawat Hosen Sifat (221901) & Jannatun Khatun (221913) &nbsp;|&nbsp; Supervisor: Dr. Hassan Mohammad Al-Emran &nbsp;|&nbsp; Co-supervisor: Md. Faisal Mina</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Controls
st.sidebar.image("poster_assets/image2.jpeg", width=140)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Module:",
    [
        "🩺 Live Thyroid Screening",
        "📈 Sensor Signal Visualizer",
        "🔬 Project & Hardware Overview",
        "📊 Dataset Profile & Cohort",
        "🏆 Benchmark Model Comparison",
        "📚 References & Clinical Scope"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Active Model")
selected_model_name = st.sidebar.selectbox(
    "Classifier Architecture:",
    ["ExtraTrees (Primary v9, 60 features)", "Compact ANN (ANN_top40, 40 features)", "Random Forest", "Support Vector Machine (SVM)"]
)

# Helper for inference
def run_screening_inference(tremor_df, neck_df, age, gender, model_choice):
    # Extract features
    feats = extract_patient_features([tremor_df], [neck_df], age=age, gender=gender)
    feature_cols = artifacts["feature_cols"]

    # Align to 416 feature columns
    x_vec = np.zeros(len(feature_cols))
    for idx, c in enumerate(feature_cols):
        x_vec[idx] = feats.get(c, 0.0)

    # Scale
    x_scaled = artifacts["scaler"].transform([x_vec])

    # Select features & predict based on model
    if "ExtraTrees" in model_choice:
        x_in = artifacts["selector_60"].transform(x_scaled)
        model = artifacts["primary_et"]
        used_feats = artifacts["selected_60"]
    elif "Compact ANN" in model_choice:
        x_in = artifacts["selector_40"].transform(x_scaled)
        model = artifacts["ann_model"]
        used_feats = artifacts["selected_40"]
    elif "Random Forest" in model_choice:
        x_in = artifacts["selector_60"].transform(x_scaled)
        model = artifacts["rf_model"]
        used_feats = artifacts["selected_60"]
    else:
        x_in = artifacts["selector_40"].transform(x_scaled)
        model = artifacts["svm_model"]
        used_feats = artifacts["selected_40"]

    pred_label = model.predict(x_in)[0]
    classes = list(model.classes_)
    probs = model.predict_proba(x_in)[0]
    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}

    return pred_label, prob_dict, feats, used_feats


# ==========================================================
# MODULE 1: LIVE THYROID SCREENING (INFERENCE)
# ==========================================================
if menu == "🩺 Live Thyroid Screening":
    st.subheader("🩺 Interactive Patient Thyroid Screening Engine")
    st.markdown("Upload or select patient tremor & neck distance recordings to perform real-time exploratory screening.")

    col_input, col_meta = st.columns([2, 1])

    with col_meta:
        st.markdown("##### 👤 Patient Demographics")
        age_in = st.slider("Patient Age (Years):", min_value=5, max_value=85, value=35)
        gender_in = st.selectbox("Patient Gender:", ["Female", "Male"])

        data_source = st.radio(
            "Select Data Mode:",
            ["Preset Test Cohort (Held-out Patients)", "Upload CSV Files (Custom ESP32 Data)", "Demo Default (TH054)"]
        )

    with col_input:
        tremor_df = None
        neck_df = None
        selected_pid = "DEMO"

        if data_source == "Preset Test Cohort (Held-out Patients)":
            test_df = artifacts["test_cohort"]
            pids = test_df['pid'].tolist()
            labels = test_df['label'].tolist()
            tshs = test_df['tsh'].tolist()
            options = [f"{pids[i]} — Actual: {labels[i]} (TSH: {tshs[i]})" for i in range(len(pids))]
            sel_opt = st.selectbox("Select Held-Out Patient:", options)
            idx = options.index(sel_opt)
            selected_pid = pids[idx]
            actual_label = labels[idx]
            actual_tsh = tshs[idx]

            # Find matching files in Thyroid_Data
            import glob
            t_files = glob.glob(f"Thyroid_Data/**/{selected_pid}/tremor_*.csv", recursive=True)
            n_files = glob.glob(f"Thyroid_Data/**/{selected_pid}/neck_*.csv", recursive=True)

            if t_files and n_files:
                tremor_df = pd.read_csv(t_files[0])
                neck_df = pd.read_csv(n_files[0])
                st.success(f"Loaded {len(tremor_df)} IMU samples and {len(neck_df)} ToF samples for patient **{selected_pid}**.")
            else:
                st.warning(f"Using cached feature vector for {selected_pid}.")

        elif data_source == "Upload CSV Files (Custom ESP32 Data)":
            up_tremor = st.file_uploader("Upload Tremor CSV (`tremor_*.csv`):", type=["csv"])
            up_neck = st.file_uploader("Upload Neck Distance CSV (`neck_*.csv`):", type=["csv"])

            if up_tremor is not None and up_neck is not None:
                tremor_df = pd.read_csv(up_tremor)
                neck_df = pd.read_csv(up_neck)
                st.success(f"Custom files loaded: Tremor ({len(tremor_df)} rows), Neck ({len(neck_df)} rows).")

        else:
            # Demo default TH054
            import glob
            t_files = glob.glob("Thyroid_Data/**/TH054/tremor_*.csv", recursive=True)
            n_files = glob.glob("Thyroid_Data/**/TH054/neck_*.csv", recursive=True)
            tremor_df = pd.read_csv(t_files[0])
            neck_df = pd.read_csv(n_files[0])
            selected_pid = "TH054"
            st.info("Loaded standard reference patient **TH054** (2 trials acquired at 100 Hz).")

    st.markdown("---")

    # Run Prediction
    if tremor_df is not None and neck_df is not None:
        pred_label, prob_dict, feats, used_feats = run_screening_inference(
            tremor_df, neck_df, age=age_in, gender=gender_in, model_choice=selected_model_name
        )

        st.markdown("### 📋 Screening Assessment Results")

        res_col1, res_col2, res_col3 = st.columns([1.5, 1.5, 1.2])

        with res_col1:
            st.markdown("##### Predicted Thyroid Status:")
            if pred_label == "Normal":
                st.markdown(f'<div class="badge-normal">🟢 NORMAL (EUTHYROID)<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {prob_dict["Normal"]*100:.1f}%</span></div>', unsafe_allow_html=True)
            elif pred_label == "Hyperthyroid":
                st.markdown(f'<div class="badge-hyper">🟣 HYPERTHYROIDISM<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {prob_dict["Hyperthyroid"]*100:.1f}%</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-hypo">🟠 HYPOTHYROIDISM<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {prob_dict["Hypothyroid"]*100:.1f}%</span></div>', unsafe_allow_html=True)

            if data_source == "Preset Test Cohort (Held-out Patients)":
                if pred_label == actual_label:
                    st.success(f"✅ Prediction matches clinical benchmark! Actual: **{actual_label}** (TSH: {actual_tsh})")
                else:
                    st.info(f"ℹ️ Actual reference label: **{actual_label}** (TSH: {actual_tsh})")

        with res_col2:
            st.markdown("##### Multimodal Class Probabilities:")
            for c_name in ["Normal", "Hyperthyroid", "Hypothyroid"]:
                p_val = prob_dict.get(c_name, 0.0)
                st.write(f"**{c_name}**: {p_val*100:.1f}%")
                st.progress(float(p_val))

        with res_col3:
            st.markdown("##### Clinical Triage Score:")
            conf = max(prob_dict.values())
            if pred_label == "Normal" and conf > 0.60:
                risk_text = "LOW RISK"
                risk_color = "#10b981"
                action_text = "Routine wellness monitoring; no acute screening anomalies detected."
            elif pred_label == "Normal":
                risk_text = "BORDERLINE"
                risk_color = "#3b82f6"
                action_text = "Borderline signal profile; recommend follow-up check."
            elif pred_label == "Hyperthyroid":
                risk_text = "ELEVATED RISK"
                risk_color = "#8b5cf6"
                action_text = "Fine tremor signatures detected. Recommend confirmatory serum TSH & FT4 laboratory test."
            else:
                risk_text = "ELEVATED RISK"
                risk_color = "#f59e0b"
                action_text = "Neck distance / hypothyroid tremor pattern detected. Recommend formal endocrine consultation."

            st.markdown(f"""
            <div style="border-left: 4px solid {risk_color}; padding-left: 12px;">
                <h4 style="color: {risk_color}; margin: 0;">{risk_text}</h4>
                <p style="font-size: 0.85rem; color: #475569; margin-top: 4px;">{action_text}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Key Biophysical Biomarkers
        st.markdown("##### 🔬 Extracted Biomarker Indicators")
        bio_c1, bio_c2, bio_c3, bio_c4 = st.columns(4)
        with bio_c1:
            st.metric("8–12 Hz Tremor Ratio", f"{feats.get('tremor_amag_ratio_8_12', 0.0)*100:.1f}%",
                      help="Elevated in thyrotoxic fine hand tremor")
        with bio_c2:
            st.metric("Tremor RMS Accel", f"{feats.get('tremor_amag_rms', 0.0):.2f} m/s²",
                      help="Root mean square acceleration")
        with bio_c3:
            st.metric("Neck Surface Distance", f"{feats.get('neck_mean_mm', 0.0):.1f} mm",
                      help="VL53L0X optical distance")
        with bio_c4:
            st.metric("Neck Distance Std", f"{feats.get('neck_std_mm', 0.0):.2f} mm",
                      help="Anterior neck surface displacement variability")

        # Explainable AI (XAI) feature importance
        with st.expander("🔍 Explainable AI (XAI) — Top Contributing Features"):
            st.markdown("Ranking of highest-impact predictors driving the primary tree ensemble prediction:")
            if hasattr(artifacts["primary_et"], "feature_importances_"):
                importances = artifacts["primary_et"].feature_importances_
                top_indices = np.argsort(importances)[::-1][:10]
                top_feats = [artifacts["selected_60"][i] for i in top_indices]
                top_scores = [importances[i] for i in top_indices]

                fig, ax = plt.subplots(figsize=(8, 3.5))
                y_pos = np.arange(len(top_feats))
                ax.barh(y_pos, top_scores, color='#2563eb', align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_feats, fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel("Relative Feature Importance (MDI)")
                ax.set_title("Top 10 Feature Attributions (v9 ExtraTrees)")
                plt.tight_layout()
                st.pyplot(fig)

        # Safety Disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Clinical Safety & Scope Advisory:</strong><br>
            The ESP32 prototype is an engineering research device designed for low-cost screening feasibility assessment.
            It does NOT measure thyroid volume, nor does it replace laboratory serum TSH/FT4 radioimmunoassays.
            Screening alerts are prospective decision-support indicators requiring validation by a qualified medical professional.
        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# MODULE 2: SENSOR SIGNAL VISUALIZER
# ==========================================================
elif menu == "📈 Sensor Signal Visualizer":
    st.subheader("📈 Multimodal Sensor Signal Visualizer")
    st.markdown("Interactive inspection of raw 6-axis IMU tremor waveforms, power spectra, and VL53L0X neck surface distance curves.")

    patient_opts = ["TH054", "TH001", "TH015", "TH087", "TH021"]
    sel_patient = st.selectbox("Select Patient Recording:", patient_opts)

    import glob
    t_files = glob.glob(f"Thyroid_Data/**/{sel_patient}/tremor_*.csv", recursive=True)
    n_files = glob.glob(f"Thyroid_Data/**/{sel_patient}/neck_*.csv", recursive=True)

    if t_files and n_files:
        df_tremor = pd.read_csv(t_files[0])
        df_neck = pd.read_csv(n_files[0])

        col_w1, col_w2 = st.columns(2)

        with col_w1:
            st.markdown("##### 🖐️ Hand Tremor Accelerometer (100 Hz)")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            t = np.arange(len(df_tremor)) / SAMPLING_RATE_TREMOR
            ax.plot(t, df_tremor.get('accel_x_m_s2', np.zeros(len(t))), label='Accel X', color='#ef4444', alpha=0.8, lw=1)
            ax.plot(t, df_tremor.get('accel_y_m_s2', np.zeros(len(t))), label='Accel Y', color='#10b981', alpha=0.8, lw=1)
            ax.plot(t, df_tremor.get('accel_z_m_s2', np.zeros(len(t))), label='Accel Z', color='#3b82f6', alpha=0.8, lw=1)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Acceleration (m/s²)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col_w2:
            st.markdown("##### 🔄 Hand Tremor Gyroscope (Angular Rate)")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(t, df_tremor.get('gyro_x_rad_s', np.zeros(len(t))), label='Gyro X', color='#f59e0b', alpha=0.8, lw=1)
            ax.plot(t, df_tremor.get('gyro_y_rad_s', np.zeros(len(t))), label='Gyro Y', color='#8b5cf6', alpha=0.8, lw=1)
            ax.plot(t, df_tremor.get('gyro_z_rad_s', np.zeros(len(t))), label='Gyro Z', color='#ec4899', alpha=0.8, lw=1)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Angular Velocity (rad/s)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("##### 📊 Tremor Power Spectral Density (Welch PSD)")
            # Compute Welch PSD of acceleration magnitude
            ax_data = df_tremor.get('accel_x_m_s2', np.zeros(len(t))).values
            ay_data = df_tremor.get('accel_y_m_s2', np.zeros(len(t))).values
            az_data = df_tremor.get('accel_z_m_s2', np.zeros(len(t))).values
            amag = np.sqrt(ax_data**2 + ay_data**2 + az_data**2)
            amag_detrend = amag - np.mean(amag)

            freqs, psd = signal.welch(amag_detrend, fs=SAMPLING_RATE_TREMOR, nperseg=256)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(freqs, psd, color='#1e3a8a', lw=1.8, label='Tremor PSD')
            ax.axvspan(3, 5, color='#fef08a', alpha=0.35, label='Resting (3–5 Hz)')
            ax.axvspan(5, 8, color='#fed7aa', alpha=0.35, label='Postural (5–8 Hz)')
            ax.axvspan(8, 12, color='#fbcfe8', alpha=0.35, label='Thyrotoxic (8–12 Hz)')
            ax.set_xlim(0, 30)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power Spectral Density")
            ax.set_title("Frequency Band Partitioning")
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col_s2:
            st.markdown("##### 📏 VL53L0X Neck Surface Distance Profile")
            dist = df_neck.get('distance_mm', pd.Series([50]*10)).values
            fig, ax = plt.subplots(figsize=(8, 3.5))
            samples = np.arange(1, len(dist) + 1)
            ax.plot(samples, dist, marker='o', color='#0284c7', lw=2, markersize=6, label='Distance (mm)')
            ax.axhline(np.mean(dist), color='#ef4444', linestyle='--', label=f'Mean: {np.mean(dist):.1f} mm')
            ax.fill_between(samples, np.mean(dist) - np.std(dist), np.mean(dist) + np.std(dist),
                            color='#bae6fd', alpha=0.4, label='±1 Std Dev')
            ax.set_xlabel("Sample Index (10 Consecutive Reads)")
            ax.set_ylabel("Distance (mm)")
            ax.set_title("Optical ToF Surface Distance Repeatability")
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)


# ==========================================================
# MODULE 3: PROJECT & HARDWARE OVERVIEW
# ==========================================================
elif menu == "🔬 Project & Hardware Overview":
    st.subheader("🔬 ESP32 Multimodal Hardware Prototype & Methodology")
    st.markdown("""
    This project evaluates a portable, low-cost multimodal sensing platform built on an **ESP32 microcontroller**
    to record hand tremor motion, anterior-neck surface distance, and voice acoustics.
    """)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("##### 🔌 Hardware System Architecture")
        st.image("poster_assets/image1.png", caption="System Components: ESP32, INMP441 Microphone, MPU6050 IMU, VL53L0X ToF Sensor, OLED, 5V Supply", use_container_width=True)
        st.markdown("""
        - **MPU6050 6-Axis IMU**: Captures 3-axis acceleration and 3-axis angular rates at 100 Hz to detect fine physiological and postural tremor.
        - **VL53L0X ToF Laser Sensor**: Measures optical distance to the anterior-neck surface across 10 reads (surface proximity only, does NOT measure internal thyroid volume).
        - **INMP441 I2S Microphone**: 16 kHz digital audio acquisition for vocal cord acoustic stability analysis.
        - **SSD1306 OLED Display**: 0.96-inch visual interface providing clinical instructions.
        """)

    with col_h2:
        st.markdown("##### ⚡ Schematic Circuit Diagram")
        st.image("poster_assets/image3.jpeg", caption="Circuit Wiring Diagram showing ESP32 I2C & I2S Interconnections", use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🔄 Methodology Workflow")
    st.image("poster_assets/image4.png", caption="8-Stage Methodology: Data Acquisition -> Signal Processing -> Feature Dataset -> Model Development -> Model Evaluation -> Screening -> UI -> Final Output", use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📸 Prototype Deployment & Field Operation")
    st.image("poster_assets/image5.png", caption="Field Prototype: Hand Tremor Ring Sensor, OLED Status Messages during Initial, Voice, Tremor, and Neck Reading Phases", use_container_width=True)


# ==========================================================
# MODULE 4: DATASET PROFILE & COHORT
# ==========================================================
elif menu == "📊 Dataset Profile & Cohort":
    st.subheader("📊 Dataset Profile & Primary Cohort Distribution")
    st.markdown("Overview of the 127 subjects collected at Jashore University of Science and Technology (JUST).")

    # Metrics row matching image7
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class="metric-card"><div class="metric-value">127</div><div class="metric-label">Subject Records</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card"><div class="metric-value">106</div><div class="metric-label">Provisional Labels</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card"><div class="metric-value">21</div><div class="metric-label">Unclassified</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class="metric-card"><div class="metric-value">308</div><div class="metric-label">Recording Groups</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.image("poster_assets/image7.png", caption="Dataset Profile & Explainable AI Benchmark Summary from Poster Presentation", use_container_width=True)

    st.markdown("---")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("##### 🏷️ Provisional TSH Label Distribution (106 Classified Subjects)")
        labels = ['Normal (41)', 'Hypothyroid (35)', 'Hyperthyroid (30)']
        counts = [41, 35, 30]
        colors = ['#10b981', '#f59e0b', '#8b5cf6']
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.axis('equal')
        st.pyplot(fig)

    with col_d2:
        st.markdown("##### 🎯 Primary v9 Supervised Cohort (70 Patients)")
        cohort_labels = ['Normal (28)', 'Hypothyroid (23)', 'Hyperthyroid (19)']
        cohort_counts = [28, 23, 19]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(cohort_labels, cohort_counts, color=['#10b981', '#f59e0b', '#8b5cf6'])
        ax.set_ylabel("Patient Count")
        ax.set_ylim(0, 35)
        for i, v in enumerate(cohort_counts):
            ax.text(i, v + 0.8, str(v), ha='center', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)


# ==========================================================
# MODULE 5: BENCHMARK MODEL COMPARISON
# ==========================================================
elif menu == "🏆 Benchmark Model Comparison":
    st.subheader("🏆 Model Benchmarks & Held-out Test Validation")
    st.markdown("Exact held-out evaluation metrics matching the primary v9 analysis presented on the poster.")

    # High-level poster stat cards matching image6
    st.image("poster_assets/image6.png", caption="Primary v9 Held-Out Test Validation Card from Poster Presentation", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Comprehensive Model Benchmark Table")

    benchmarks = artifacts["benchmarks"]
    b_df = pd.DataFrame(benchmarks)

    # Format percentages
    display_df = pd.DataFrame({
        "Model Name": b_df["model"],
        "Features Used": b_df["features"],
        "Test Accuracy": b_df["accuracy"].apply(lambda x: f"{x*100:.1f}%"),
        "Correct": b_df["test_correct"],
        "Macro-F1": b_df["macro_f1"].apply(lambda x: f"{x*100:.1f}%"),
        "Balanced Acc": b_df["balanced_acc"].apply(lambda x: f"{x*100:.1f}%"),
        "Macro AUC": b_df["macro_auc"].apply(lambda x: f"{x:.3f}"),
        "MCC": b_df["mcc"].apply(lambda x: f"{x:.3f}"),
        "Model Type": b_df["type"]
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("##### 🎯 Primary ExtraTrees Confusion Matrix (Held-out Test, n=11)")
        cm = np.array([
            [2, 1, 1],  # Normal (support 4): 2 correct, 1 hyper, 1 hypo
            [0, 2, 1],  # Hyper (support 3): 2 correct, 1 hypo
            [1, 0, 3]   # Hypo (support 4): 3 correct, 1 normal
        ])
        classes = ['Normal', 'Hyper', 'Hypo']
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted Label', ylabel='Actual Reference Label')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight='bold', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

    with col_b2:
        st.markdown("##### 📈 Model Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        models_sub = [b["model"] for b in benchmarks[:6]]
        accs = [b["accuracy"]*100 for b in benchmarks[:6]]
        colors = ['#1e3a8a', '#2563eb', '#0284c7', '#38bdf8', '#64748b', '#94a3b8']
        y_pos = np.arange(len(models_sub))
        ax.barh(y_pos, accs, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models_sub, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Held-Out Test Accuracy (%)")
        for i, v in enumerate(accs):
            ax.text(v + 1.5, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)


# ==========================================================
# MODULE 6: REFERENCES & CLINICAL SCOPE
# ==========================================================
elif menu == "📚 References & Clinical Scope":
    st.subheader("📚 References, Engineering Limitations & Clinical Scope")

    st.markdown("""
    #### 📖 Academic References
    1. **Chaganti, S. et al. (2022)**. *Thyroid Disease Prediction Using Selective Features and Machine Learning Techniques*. **Cancers**, 14(16), 3914. https://doi.org/10.3390/cancers14163914
    2. **Abedin, M. et al. (2026)**. *Explainable Machine Learning With Hybrid Feature Selection for Thyroid Disease Classification: A Case Study in Bangladesh*. **Engineering Reports**. https://doi.org/10.1002/eng2.70980

    ---

    #### ⚠️ Documented Engineering Limitations
    - **Small Test Cohort (n=11)**: 95% Wilson confidence interval spans 35.4–84.8%, highlighting the exploratory nature of the study.
    - **Acoustic Verification Challenge**: Strict automated speech recognition (ASR) required an exact "Hello" transcription. Zero voice samples passed strict phoneme verification in v9, leading to voice modality exclusion in the primary classifier.
    - **Provisional Labels**: 21 out of 127 subject records lacked completed TSH/FT4 laboratory panels.
    - **Optical Surface Distance Constraint**: The VL53L0X sensor records external anterior-neck surface distance; it cannot infer gland volume or nodule architecture.

    ---

    #### 💡 Potential Applications
    - Low-cost screening support for community health workers and rural outpatient clinics in Bangladesh.
    - Longitudinal triage and follow-up for patients undergoing medical thyroid therapy.
    - Multimodal bio-signal IoT research platform for future endocrine health studies.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "Low-Cost Multimodal Thyroid Screening System &bull; JUST Department of Biomedical Engineering &bull; 2026"
    "</div>",
    unsafe_allow_html=True
)
