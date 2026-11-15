"""
Interactive Multimodal Thyroid Screening Web Application
Department of Biomedical Engineering, Jashore University of Science and Technology (JUST)
Authors: Sakhawat Hosen Sifat (221901) & Jannatun Khatun (221913)
Supervisors: Dr. Hassan Mohammad Al-Emran | Co-supervisor: Md. Faisal Mina
"""

import os
import sys
import glob
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from scipy import signal

# Add local path
sys.path.insert(0, os.path.abspath("."))
from src.feature_extractor import (
    extract_tremor_features, extract_neck_features,
    extract_patient_features, SAMPLING_RATE_TREMOR, safe_float,
    get_series_by_aliases
)

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
        background: linear-gradient(135deg, #0b2238 0%, #153c61 50%, #21618c 100%);
        padding: 22px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 22px;
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
        font-size: 0.95rem;
        margin-bottom: 0px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        color: #0f172a !important;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e3a8a !important;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #334155 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 700;
    }
    .badge-normal {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46 !important;
        padding: 14px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
    }
    .badge-hyper {
        background-color: #f5f3ff;
        border: 2px solid #8b5cf6;
        color: #5b21b6 !important;
        padding: 14px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
    }
    .badge-hypo {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        color: #92400e !important;
        padding: 14px 20px;
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
        color: #881337 !important;
        margin-top: 18px;
    }
    .patient-info-box {
        background-color: #f1f5f9;
        border: 2px solid #0284c7;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        font-size: 0.95rem;
        color: #0f172a !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        line-height: 1.7;
    }
    .patient-info-box strong {
        color: #0369a1 !important;
    }
    .patient-info-box span {
        color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find patient sensor files reliably across all folder naming schemes
def find_patient_sensor_files(pid):
    num_match = re.search(r'\d+', str(pid))
    if not num_match:
        return [], []
    num = int(num_match.group())
    patterns = [f"TH{num:03d}", f"TH{num:02d}", f"TH{num}", f"TH {num:03d}", f"TH {num}"]
    t_files, n_files = [], []
    for pat in set(patterns):
        t_files.extend(glob.glob(f"Thyroid_Data/**/{pat}/tremor_*.csv", recursive=True))
        n_files.extend(glob.glob(f"Thyroid_Data/**/{pat}/neck_*.csv", recursive=True))
    return sorted(list(set(t_files))), sorted(list(set(n_files)))

# Helper to normalize patient ID
def normalize_pid(p):
    m = re.findall(r'TH\s*(\d+)', str(p), re.IGNORECASE)
    return f'TH{int(m[0]):03d}' if m else str(p).strip()

# Load Subject Info Excel safely
@st.cache_data
def load_subject_metadata():
    if os.path.exists("Subject Info Updated.xlsx"):
        df = pd.read_excel("Subject Info Updated.xlsx")
        df['pid'] = df['Patient ID'].apply(normalize_pid)
        def clean_tsh_val(v):
            if pd.isna(v): return np.nan
            s = str(v).strip()
            if s.startswith('<'): return 0.005
            if s.startswith('>'): return 200.0
            try: return float(s)
            except: return np.nan
        df['tsh_num'] = df['TSH'].apply(clean_tsh_val)
        def classify_tsh(t):
            if pd.isna(t): return "Unclassified"
            if t < 0.35: return "Hyperthyroid"
            elif t > 4.3: return "Hypothyroid"
            else: return "Normal"
        df['status'] = df['tsh_num'].apply(classify_tsh)
        return df
    return pd.DataFrame()

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

    # Optional models
    lr_model = None
    if os.path.exists(os.path.join(models_dir, "logistic_regression.pkl")):
        with open(os.path.join(models_dir, "logistic_regression.pkl"), "rb") as f:
            lr_model = pickle.load(f)

    gb_model = None
    if os.path.exists(os.path.join(models_dir, "gradient_boosting.pkl")):
        with open(os.path.join(models_dir, "gradient_boosting.pkl"), "rb") as f:
            gb_model = pickle.load(f)

    test_cohort = pd.read_csv(os.path.join(models_dir, "heldout_test_set.csv"))

    return {
        "primary_et": primary_et,
        "ann_model": ann_model,
        "rf_model": rf_model,
        "svm_model": svm_model,
        "lr_model": lr_model,
        "gb_model": gb_model,
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
    metadata_df = load_subject_metadata()
except Exception as e:
    st.error(f"Error loading trained models: {e}. Please ensure models are trained in `models/`.")
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
if os.path.exists("poster_assets/image2.jpeg"):
    st.sidebar.image("poster_assets/image2.jpeg", width=130)
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
model_options = [
    "ExtraTrees (Primary v9, 60 features) [Poster Match: 63.6%]",
    "Compact ANN (ANN_top40, 40 features)",
    "Multimodal Random Forest (60 features)",
    "Support Vector Machine (SVM RBF, 40 features)",
    "Gradient Boosting / XGBoost (40 features)",
    "Logistic Regression (40 features)"
]
selected_model_name = st.sidebar.selectbox("Classifier Architecture:", model_options)

# Helper for inference
def run_screening_inference(tremor_df, neck_df, age, gender, model_choice, cached_feats_dict=None):
    if cached_feats_dict is not None:
        feats = cached_feats_dict
    else:
        feats = extract_patient_features([tremor_df], [neck_df], age=age, gender=gender)

    feature_cols = artifacts["feature_cols"]

    # Align to 416 feature columns with strict NaN/inf sanitization
    x_vec = np.zeros(len(feature_cols), dtype=float)
    for idx, c in enumerate(feature_cols):
        x_vec[idx] = safe_float(feats.get(c, 0.0))

    x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    x_scaled = artifacts["scaler"].transform([x_vec])
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

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
    elif "Gradient Boosting" in model_choice and artifacts["gb_model"] is not None:
        x_in = artifacts["selector_40"].transform(x_scaled)
        model = artifacts["gb_model"]
        used_feats = artifacts["selected_40"]
    elif "Logistic Regression" in model_choice and artifacts["lr_model"] is not None:
        x_in = artifacts["selector_40"].transform(x_scaled)
        model = artifacts["lr_model"]
        used_feats = artifacts["selected_40"]
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
    st.markdown("Perform real-time multimodal screening using 6-axis IMU hand tremor and VL53L0X anterior-neck surface distance signals.")

    col_input, col_meta = st.columns([1.8, 1.2])

    with col_meta:
        st.markdown("##### ⚙️ Screening Configuration")
        data_source = st.radio(
            "Select Data Source:",
            [
                "Preset Held-Out Test Cohort (n=11)",
                "Browse Full Dataset Explorer (127 Patients)",
                "Upload Custom Sensor CSVs (ESP32 Live Data)",
                "Demo Sample Patient (TH054)"
            ]
        )

    with col_input:
        tremor_df = None
        neck_df = None
        selected_pid = "DEMO"
        actual_label = None
        actual_tsh = None
        patient_meta_info = {}
        cached_feat_dict = None

        if data_source == "Preset Held-Out Test Cohort (n=11)":
            test_df = artifacts["test_cohort"]
            pids = test_df['pid'].tolist()
            labels = test_df['label'].tolist()
            tshs = test_df['tsh'].tolist()
            options = [f"{pids[i]} — Clinical Ref: {labels[i]} (TSH: {tshs[i]})" for i in range(len(pids))]
            sel_opt = st.selectbox("Select Held-Out Patient:", options)
            idx = options.index(sel_opt)
            selected_pid = pids[idx]
            actual_label = labels[idx]
            actual_tsh = tshs[idx]

            # Find matching files with robust resolver
            t_files, n_files = find_patient_sensor_files(selected_pid)
            if t_files and n_files:
                tremor_df = pd.read_csv(t_files[0])
                neck_df = pd.read_csv(n_files[0])
                st.success(f"Loaded raw sensor streams for **{selected_pid}** ({len(tremor_df)} IMU reads, {len(neck_df)} ToF reads).")
            else:
                # Fallback to test_cohort feature row
                row_data = test_df[test_df['pid'] == selected_pid].iloc[0].to_dict()
                cached_feat_dict = row_data
                st.info(f"Loaded feature matrix representation for **{selected_pid}**.")

        elif data_source == "Browse Full Dataset Explorer (127 Patients)":
            all_pids = [f"TH{i:03d}" for i in range(1, 128)]
            if not metadata_df.empty:
                meta_dict = metadata_df.set_index('pid').to_dict('index')
                opts = [f"{pid} — {meta_dict.get(pid, {}).get('status', 'Unclassified')} (TSH: {meta_dict.get(pid, {}).get('TSH', 'N/A')})" for pid in all_pids]
                sel_opt = st.selectbox("Select Patient from 127 Cohort:", opts)
                selected_pid = all_pids[opts.index(sel_opt)]
            else:
                selected_pid = st.selectbox("Select Patient ID:", all_pids)

            t_files, n_files = find_patient_sensor_files(selected_pid)
            if t_files and n_files:
                tremor_df = pd.read_csv(t_files[0])
                neck_df = pd.read_csv(n_files[0])
                st.success(f"Loaded raw recordings for **{selected_pid}** ({len(tremor_df)} IMU samples, {len(neck_df)} ToF samples).")
            else:
                st.warning(f"No sensor CSVs found on disk for {selected_pid}.")

        elif data_source == "Upload Custom Sensor CSVs (ESP32 Live Data)":
            selected_pid = "CUSTOM_UPLOAD"
            st.markdown("Upload raw sensor CSV exports from the ESP32 data logger:")
            up_c1, up_c2 = st.columns(2)
            with up_c1:
                up_tremor = st.file_uploader("Hand Tremor CSV (`tremor_*.csv`):", type=["csv"], key="live_up_t")
            with up_c2:
                up_neck = st.file_uploader("Neck Distance CSV (`neck_*.csv`):", type=["csv"], key="live_up_n")

            if up_tremor is not None and up_neck is not None:
                try:
                    tremor_df = pd.read_csv(up_tremor)
                    neck_df = pd.read_csv(up_neck)
                    st.success(f"Custom files loaded: Tremor ({len(tremor_df)} rows), Neck ({len(neck_df)} rows).")
                except Exception as e:
                    st.error(f"Error reading uploaded CSVs: {e}")
            else:
                st.info("Please upload both tremor and neck CSV files to initiate screening.")

        else:
            # Demo TH054
            selected_pid = "TH054"
            t_files, n_files = find_patient_sensor_files("TH054")
            tremor_df = pd.read_csv(t_files[0])
            neck_df = pd.read_csv(n_files[0])
            st.info("Loaded reference benchmark patient **TH054**.")

    # Patient Demographic Metadata Bar
    if not metadata_df.empty and selected_pid != "CUSTOM_UPLOAD":
        p_row = metadata_df[metadata_df['pid'] == selected_pid]
        if not p_row.empty:
            r = p_row.iloc[0]
            actual_label = str(r.get('status', actual_label)).strip()
            tsh_raw = str(r.get('TSH', actual_tsh)).strip()
            p_age = safe_float(r.get('Age', 35), default=35)
            p_gender = str(r.get('Gender', 'Female')).strip()
            p_name = str(r.get('Patient Name', 'Anonymous')).strip()
            ft4_raw = str(r.get('FT4', '')).strip()
            wt_raw = str(r.get('Weight', '')).strip()

            p_ft4 = "Pending Lab" if (not ft4_raw or ft4_raw.lower() in ['nan', 'none', 'n/a']) else f"{ft4_raw} ng/dL"
            p_wt = "N/A" if (not wt_raw or wt_raw.lower() in ['nan', 'none', 'n/a']) else f"{wt_raw} kg"
            tsh_disp = "Pending Lab" if (not tsh_raw or tsh_raw.lower() in ['nan', 'none', 'n/a']) else f"{tsh_raw} mIU/L"

            # Badge color for reference status
            if "Normal" in actual_label:
                status_badge = f'<span style="background-color:#d1fae5; color:#065f46; border:1px solid #10b981; padding:3px 10px; border-radius:6px; font-weight:700;">{actual_label}</span>'
            elif "Hyper" in actual_label:
                status_badge = f'<span style="background-color:#ede9fe; color:#5b21b6; border:1px solid #8b5cf6; padding:3px 10px; border-radius:6px; font-weight:700;">{actual_label}</span>'
            elif "Hypo" in actual_label:
                status_badge = f'<span style="background-color:#fef3c7; color:#92400e; border:1px solid #f59e0b; padding:3px 10px; border-radius:6px; font-weight:700;">{actual_label}</span>'
            else:
                status_badge = f'<span style="background-color:#e2e8f0; color:#1e293b; border:1px solid #94a3b8; padding:3px 10px; border-radius:6px; font-weight:700;">{actual_label}</span>'

            st.markdown(f"""
            <div class="patient-info-box" style="background-color:#f1f5f9; border:2px solid #0284c7; border-radius:10px; padding:16px 20px; color:#0f172a; margin-bottom:16px;">
                <div style="font-size:1.05rem; font-weight:700; color:#0369a1; margin-bottom:8px;">
                    📋 Subject Demographics & Clinical Records
                </div>
                <div style="font-size:0.95rem; color:#0f172a; line-height:1.8;">
                    <span style="color:#0369a1; font-weight:700;">Patient ID:</span> <span style="color:#0f172a; font-weight:600;">{selected_pid}</span> &nbsp;|&nbsp;
                    <span style="color:#0369a1; font-weight:700;">Name:</span> <span style="color:#0f172a; font-weight:600;">{p_name}</span> &nbsp;|&nbsp;
                    <span style="color:#0369a1; font-weight:700;">Age:</span> <span style="color:#0f172a; font-weight:600;">{p_age:.0f} yrs</span> &nbsp;|&nbsp;
                    <span style="color:#0369a1; font-weight:700;">Gender:</span> <span style="color:#0f172a; font-weight:600;">{p_gender}</span> &nbsp;|&nbsp;
                    <span style="color:#0369a1; font-weight:700;">Weight:</span> <span style="color:#0f172a; font-weight:600;">{p_wt}</span> &nbsp;|&nbsp;
                    <span style="color:#0369a1; font-weight:700;">Serum TSH:</span> <span style="color:#0f172a; font-weight:600;">{tsh_disp}</span> &nbsp;|&nbsp;
                    <span style="color:#0369a1; font-weight:700;">FT4:</span> <span style="color:#0f172a; font-weight:600;">{p_ft4}</span> &nbsp;|&nbsp;
                    <span style="color:#0369a1; font-weight:700;">Reference Lab Status:</span> {status_badge}
                </div>
            </div>
            """, unsafe_allow_html=True)
            age_in = p_age
            gender_in = p_gender
        else:
            age_in = 35.0
            gender_in = "Female"
    else:
        with col_meta:
            age_in = st.slider("Patient Age (Years):", min_value=5, max_value=85, value=35)
            gender_in = st.selectbox("Patient Gender:", ["Female", "Male"])

    st.markdown("---")

    # Run Prediction Block
    has_valid_inputs = (tremor_df is not None and neck_df is not None) or (cached_feat_dict is not None)
    if has_valid_inputs:
        pred_label, prob_dict, feats, used_feats = run_screening_inference(
            tremor_df, neck_df, age=age_in, gender=gender_in,
            model_choice=selected_model_name, cached_feats_dict=cached_feat_dict
        )

        st.markdown("### 📋 Screening Assessment Results")

        res_col1, res_col2, res_col3 = st.columns([1.5, 1.5, 1.2])

        with res_col1:
            st.markdown("##### Predicted Thyroid Status:")
            if pred_label == "Normal":
                st.markdown(f'<div class="badge-normal">🟢 NORMAL (EUTHYROID)<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {prob_dict.get("Normal", 0.0)*100:.1f}%</span></div>', unsafe_allow_html=True)
            elif pred_label == "Hyperthyroid":
                st.markdown(f'<div class="badge-hyper">🟣 HYPERTHYROIDISM<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {prob_dict.get("Hyperthyroid", 0.0)*100:.1f}%</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-hypo">🟠 HYPOTHYROIDISM<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {prob_dict.get("Hypothyroid", 0.0)*100:.1f}%</span></div>', unsafe_allow_html=True)

            if actual_label and actual_label != "Unclassified":
                if pred_label == actual_label:
                    st.success(f"✅ Prediction matches clinical laboratory benchmark! Reference: **{actual_label}** (TSH: {actual_tsh})")
                else:
                    st.info(f"ℹ️ Clinical laboratory reference: **{actual_label}** (TSH: {actual_tsh})")

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
                action_text = "Normal bio-signal profile. Routine follow-up recommended."
            elif pred_label == "Normal":
                risk_text = "BORDERLINE"
                risk_color = "#3b82f6"
                action_text = "Mild signal variability. Recommend routine monitoring."
            elif pred_label == "Hyperthyroid":
                risk_text = "ELEVATED RISK"
                risk_color = "#8b5cf6"
                action_text = "Fine tremor signatures detected. Recommend confirmatory serum TSH & FT4 laboratory testing."
            else:
                risk_text = "ELEVATED RISK"
                risk_color = "#f59e0b"
                action_text = "Anterior neck distance / hypothyroid profile detected. Recommend clinical endocrine consultation."

            st.markdown(f"""
            <div style="border-left: 4px solid {risk_color}; padding-left: 12px;">
                <h4 style="color: {risk_color}; margin: 0;">{risk_text}</h4>
                <p style="font-size: 0.85rem; color: #475569; margin-top: 4px;">{action_text}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Key Biophysical Biomarkers
        st.markdown("##### 🔬 Extracted Biomarker Indicators")
        bio_c1, bio_c2, bio_c3, bio_c4, bio_c5 = st.columns(5)
        with bio_c1:
            st.metric("8–12 Hz Tremor Ratio", f"{feats.get('tremor_amag_ratio_8_12', 0.0)*100:.1f}%",
                      help="Elevated in thyrotoxic fine hand tremor")
        with bio_c2:
            st.metric("Dominant Tremor Freq", f"{feats.get('tremor_amag_dom_freq', 0.0):.1f} Hz",
                      help="Peak frequency of tremor motion")
        with bio_c3:
            st.metric("Tremor RMS Accel", f"{feats.get('tremor_amag_rms', 0.0):.2f} m/s²",
                      help="Root mean square acceleration")
        with bio_c4:
            st.metric("Neck Surface Distance", f"{feats.get('neck_mean_mm', 0.0):.1f} mm",
                      help="VL53L0X optical distance")
        with bio_c5:
            st.metric("Neck Displacement Std", f"{feats.get('neck_std_mm', 0.0):.2f} mm",
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
                plt.close(fig)

        # Safety Disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Clinical Safety & Scope Advisory:</strong><br>
            The ESP32 prototype is an engineering research platform designed for low-cost screening feasibility assessment.
            It does NOT measure internal thyroid gland volume or replace laboratory serum TSH/FT4 radioimmunoassays.
            Screening alerts are prospective decision-support indicators requiring validation by a qualified medical professional.
        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# MODULE 2: SENSOR SIGNAL VISUALIZER
# ==========================================================
elif menu == "📈 Sensor Signal Visualizer":
    st.subheader("📈 Multimodal Sensor Signal Visualizer")
    st.markdown("Interactive inspection of raw 6-axis IMU tremor waveforms, power spectra, and VL53L0X neck surface distance curves.")

    viz_source = st.radio("Signal Source:", ["Select from Cohort Patients (127 Subjects)", "Upload Custom Sensor Files"], horizontal=True)

    df_tremor = None
    df_neck = None

    if viz_source == "Select from Cohort Patients (127 Subjects)":
        patient_opts = [f"TH{i:03d}" for i in range(1, 128)]
        # Default to TH054
        def_idx = patient_opts.index("TH054") if "TH054" in patient_opts else 0
        sel_patient = st.selectbox("Select Patient to Visualize:", patient_opts, index=def_idx)

        t_files, n_files = find_patient_sensor_files(sel_patient)
        if t_files and n_files:
            df_tremor = pd.read_csv(t_files[0])
            df_neck = pd.read_csv(n_files[0])
            st.success(f"Visualizing patient **{sel_patient}** ({len(df_tremor)} IMU points, {len(df_neck)} ToF reads).")
        else:
            st.warning(f"No recordings found on disk for {sel_patient}.")
    else:
        st.markdown("Upload tremor and neck CSVs to visualize custom signals:")
        v_c1, v_c2 = st.columns(2)
        with v_c1:
            up_v_t = st.file_uploader("Upload Tremor CSV:", type=["csv"], key="viz_up_t")
        with v_c2:
            up_v_n = st.file_uploader("Upload Neck CSV:", type=["csv"], key="viz_up_n")

        if up_v_t is not None and up_v_n is not None:
            try:
                df_tremor = pd.read_csv(up_v_t)
                df_neck = pd.read_csv(up_v_n)
                st.success(f"Custom signals loaded: Tremor ({len(df_tremor)} samples), Neck ({len(df_neck)} samples).")
            except Exception as e:
                st.error(f"Error loading custom CSVs: {e}")

    if df_tremor is not None and df_neck is not None:
        ax_data = get_series_by_aliases(df_tremor, ['accel_x_m_s2', 'accel_x', 'acc_x', 'ax', 'accelerometer_x'])
        ay_data = get_series_by_aliases(df_tremor, ['accel_y_m_s2', 'accel_y', 'acc_y', 'ay', 'accelerometer_y'])
        az_data = get_series_by_aliases(df_tremor, ['accel_z_m_s2', 'accel_z', 'acc_z', 'az', 'accelerometer_z'], default_val=9.8)
        gx_data = get_series_by_aliases(df_tremor, ['gyro_x_rad_s', 'gyro_x', 'gyr_x', 'gx', 'gyroscope_x'])
        gy_data = get_series_by_aliases(df_tremor, ['gyro_y_rad_s', 'gyro_y', 'gyr_y', 'gy', 'gyroscope_y'])
        gz_data = get_series_by_aliases(df_tremor, ['gyro_z_rad_s', 'gyro_z', 'gyr_z', 'gz', 'gyroscope_z'])
        min_len = min(len(ax_data), len(ay_data), len(az_data), len(gx_data), len(gy_data), len(gz_data))
        if min_len < 1:
            min_len = 100
            ax_data, ay_data, az_data = np.zeros(min_len), np.zeros(min_len), np.full(min_len, 9.8)
            gx_data, gy_data, gz_data = np.zeros(min_len), np.zeros(min_len), np.zeros(min_len)
        t = np.arange(min_len) / SAMPLING_RATE_TREMOR

        col_w1, col_w2 = st.columns(2)

        with col_w1:
            st.markdown("##### 🖐️ Hand Tremor Accelerometer (100 Hz)")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(t, ax_data[:min_len], label='Accel X', color='#ef4444', alpha=0.8, lw=1)
            ax.plot(t, ay_data[:min_len], label='Accel Y', color='#10b981', alpha=0.8, lw=1)
            ax.plot(t, az_data[:min_len], label='Accel Z', color='#3b82f6', alpha=0.8, lw=1)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Acceleration (m/s²)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col_w2:
            st.markdown("##### 🔄 Hand Tremor Gyroscope (Angular Rate)")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(t, gx_data[:min_len], label='Gyro X', color='#f59e0b', alpha=0.8, lw=1)
            ax.plot(t, gy_data[:min_len], label='Gyro Y', color='#8b5cf6', alpha=0.8, lw=1)
            ax.plot(t, gz_data[:min_len], label='Gyro Z', color='#ec4899', alpha=0.8, lw=1)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Angular Velocity (rad/s)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("##### 📊 Tremor Power Spectral Density (Welch PSD)")
            amag = np.sqrt(ax_data[:min_len]**2 + ay_data[:min_len]**2 + az_data[:min_len]**2)
            amag = amag[np.isfinite(amag)]
            if len(amag) < 8:
                amag = np.zeros(10)

            amag_detrend = amag - np.mean(amag)
            nperseg = min(256, len(amag_detrend))
            freqs, psd = signal.welch(amag_detrend, fs=SAMPLING_RATE_TREMOR, nperseg=nperseg)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(freqs, psd, color='#1e3a8a', lw=1.8, label='Tremor PSD')
            ax.axvspan(3, 5, color='#fef08a', alpha=0.35, label='Resting (3–5 Hz)')
            ax.axvspan(5, 8, color='#fed7aa', alpha=0.35, label='Postural (5–8 Hz)')
            ax.axvspan(8, 12, color='#fbcfe8', alpha=0.35, label='Thyrotoxic (8–12 Hz)')
            ax.set_xlim(0, 30)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power Spectral Density")
            ax.set_title("Physiological Frequency Bands")
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col_s2:
            st.markdown("##### 📏 VL53L0X Neck Surface Distance Profile")
            dist = get_series_by_aliases(
                df_neck,
                ['distance_mm', 'distance', 'dist_mm', 'dist', 'neck_distance', 'neck_distance_mm', 'distance (mm)'],
                default_val=50.0
            )
            dist = dist[np.isfinite(dist)]
            if len(dist) == 0:
                dist = np.array([50.0] * 10)

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
            plt.close(fig)


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
        if os.path.exists("poster_assets/image1.png"):
            st.image("poster_assets/image1.png", caption="System Components: ESP32, INMP441 Microphone, MPU6050 IMU, VL53L0X ToF Sensor, OLED, 5V Supply", width="stretch")
        st.markdown("""
        - **MPU6050 6-Axis IMU**: Captures 3-axis acceleration and 3-axis angular rates at 100 Hz to detect fine physiological and postural tremor.
        - **VL53L0X ToF Laser Sensor**: Measures optical distance to the anterior-neck surface across 10 reads (surface proximity only, does NOT measure internal thyroid volume).
        - **INMP441 I2S Microphone**: 16 kHz digital audio acquisition for vocal cord acoustic stability analysis.
        - **SSD1306 OLED Display**: 0.96-inch visual interface providing clinical instructions.
        """)

    with col_h2:
        st.markdown("##### ⚡ Schematic Circuit Diagram")
        if os.path.exists("poster_assets/image3.jpeg"):
            st.image("poster_assets/image3.jpeg", caption="Circuit Wiring Diagram showing ESP32 I2C & I2S Interconnections", width="stretch")

    st.markdown("---")
    st.markdown("##### 🔄 Methodology Workflow")
    if os.path.exists("poster_assets/image4.png"):
        st.image("poster_assets/image4.png", caption="8-Stage Methodology: Data Acquisition -> Signal Processing -> Feature Dataset -> Model Development -> Model Evaluation -> Screening -> UI -> Final Output", width="stretch")

    st.markdown("---")
    st.markdown("##### 📸 Prototype Deployment & Field Operation")
    if os.path.exists("poster_assets/image5.png"):
        st.image("poster_assets/image5.png", caption="Field Prototype: Hand Tremor Ring Sensor, OLED Status Messages during Initial, Voice, Tremor, and Neck Reading Phases", width="stretch")


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

    if os.path.exists("poster_assets/image7.png"):
        st.image("poster_assets/image7.png", caption="Dataset Profile & Explainable AI Benchmark Summary from Poster Presentation", width="stretch")

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
        plt.close(fig)

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
        plt.close(fig)

    if not metadata_df.empty:
        st.markdown("---")
        st.markdown("##### 🔍 Full 127-Subject Directory Table")
        status_filter = st.multiselect(
            "Filter by Clinical Status:",
            ["Normal", "Hyperthyroid", "Hypothyroid", "Unclassified"],
            default=["Normal", "Hyperthyroid", "Hypothyroid", "Unclassified"],
            key="status_filter_multiselect"
        )
        filtered_meta = metadata_df[metadata_df['status'].isin(status_filter)]
        show_df = filtered_meta[['pid', 'Patient Name', 'Age', 'Gender', 'TSH', 'FT4', 'status']].rename(
            columns={'pid': 'Patient ID', 'status': 'Clinical Status'}
        ).copy()

        # Sanitize all columns to clean string format to guarantee zero PyArrow serialization issues
        show_df['Patient ID'] = show_df['Patient ID'].astype(str)
        show_df['Patient Name'] = show_df['Patient Name'].fillna("N/A").astype(str)
        show_df['Age'] = show_df['Age'].apply(lambda x: f"{x:.0f}" if pd.notna(x) and str(x).replace('.','',1).isdigit() else str(x) if pd.notna(x) else "N/A")
        show_df['Gender'] = show_df['Gender'].fillna("N/A").astype(str)
        show_df['TSH'] = show_df['TSH'].apply(lambda x: "N/A" if pd.isna(x) else str(x))
        show_df['FT4'] = show_df['FT4'].apply(lambda x: "N/A" if pd.isna(x) else str(x))
        show_df['Clinical Status'] = show_df['Clinical Status'].astype(str)

        st.dataframe(show_df, width="stretch", hide_index=True)


# ==========================================================
# MODULE 5: BENCHMARK MODEL COMPARISON
# ==========================================================
elif menu == "🏆 Benchmark Model Comparison":
    st.subheader("🏆 Model Benchmarks & Held-out Test Validation")
    st.markdown("Exact held-out evaluation metrics matching the primary v9 analysis presented on the poster.")

    if os.path.exists("poster_assets/image6.png"):
        st.image("poster_assets/image6.png", caption="Primary v9 Held-Out Test Validation Card from Poster Presentation", width="stretch")

    st.markdown("---")
    st.markdown("### 📊 Comprehensive Model Benchmark Table")

    benchmarks = artifacts["benchmarks"]
    b_df = pd.DataFrame(benchmarks)

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

    st.dataframe(display_df, width="stretch", hide_index=True)

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
        plt.close(fig)

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
        plt.close(fig)


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
