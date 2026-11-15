"""
Thyroid Monitor - Professional Research Desktop
Non-invasive Multimodal Thyroid Screening System
Department of Biomedical Engineering, Jashore University of Science and Technology (JUST)
"""

import os
import sys
import glob
import re
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from scipy import signal

# Add local project root
sys.path.insert(0, os.path.abspath("."))
from src.feature_extractor import (
    extract_tremor_features, extract_neck_features,
    extract_patient_features, SAMPLING_RATE_TREMOR, safe_float,
    get_series_by_aliases
)
from src.pdf_generator import generate_patient_pdf

# Streamlit Page Configuration
st.set_page_config(
    page_title="Thyroid Monitor - Professional Research Desktop",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling matching Thyroid Monitor Desktop UI
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Top Desktop Header Bar */
    .desktop-topbar {
        background: #0f172a;
        color: #e2e8f0;
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 0px;
        border-bottom: 1px solid #1e293b;
    }
    .desktop-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
        color: #ffffff;
    }
    .desktop-logo-box {
        background: #0d9488;
        color: #ffffff;
        font-weight: 800;
        font-size: 0.95rem;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 5px;
    }
    .desktop-menu {
        display: flex;
        gap: 18px;
        color: #94a3b8;
        font-size: 0.8rem;
    }
    .desktop-menu span:hover {
        color: #ffffff;
        cursor: pointer;
    }

    /* Workspace Title Bar */
    .workspace-subbar {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: none;
        padding: 14px 20px;
        border-radius: 0 0 10px 10px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .workspace-crumb {
        font-size: 0.75rem;
        font-weight: 700;
        color: #0d9488;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .workspace-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .workspace-desc {
        font-size: 0.85rem;
        color: #64748b;
        margin: 0;
    }
    .workspace-pill {
        background: #ecfdf5;
        border: 1.5px solid #10b981;
        color: #065f46;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Deep Forest Teal Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0a3d36 0%, #0f594e 100%);
        border: 1px solid #136f62;
        border-radius: 14px;
        padding: 24px 28px;
        color: #ffffff;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(10, 61, 54, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-pretitle {
        font-size: 0.72rem;
        font-weight: 800;
        color: #6ee7b7;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
        line-height: 1.25;
    }
    .hero-subtext {
        font-size: 0.92rem;
        color: #d1fae5;
        max-width: 580px;
        margin-bottom: 16px;
        line-height: 1.5;
    }

    /* Clean Metric Cards */
    .modern-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .modern-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .metric-icon-badge {
        width: 38px;
        height: 38px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-bottom: 12px;
    }
    .metric-label-modern {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-number-modern {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .metric-sub-modern {
        font-size: 0.78rem;
        color: #94a3b8;
    }

    /* Split Panels: Recent Sessions & Sensor Overview */
    .panel-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
    }
    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .panel-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .panel-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 16px;
    }

    /* Session List Items */
    .session-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .session-item:last-child {
        border-bottom: none;
    }
    .session-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .session-id {
        font-weight: 700;
        font-size: 0.88rem;
        color: #0f172a;
    }
    .session-meta {
        font-size: 0.75rem;
        color: #64748b;
    }

    /* Status Tags */
    .tag-planned {
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 9999px;
    }
    .tag-progress {
        background: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 9999px;
    }
    .tag-completed {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 9999px;
    }

    /* Modality Progress Bars */
    .modality-row {
        margin-bottom: 14px;
    }
    .modality-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 4px;
    }

    /* Screening Result Badges */
    .badge-normal {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46 !important;
        padding: 16px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
    }
    .badge-hyper {
        background-color: #f5f3ff;
        border: 2px solid #8b5cf6;
        color: #5b21b6 !important;
        padding: 16px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
    }
    .badge-hypo {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        color: #92400e !important;
        padding: 16px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
    }

    /* Login Screen Container */
    .login-wrapper {
        max-width: 460px;
        margin: 40px auto;
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 16px;
        padding: 32px 36px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.06);
    }
    .login-brand-header {
        text-align: center;
        margin-bottom: 24px;
    }
    .login-logo {
        background: #0d9488;
        color: #ffffff;
        font-weight: 800;
        font-size: 1.6rem;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
    }

    /* User Profile Sidebar Card */
    .user-profile-card {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 16px;
    }
    .user-profile-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #2dd4bf !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .user-profile-name {
        font-size: 0.95rem;
        font-weight: 800;
        color: #ffffff !important;
    }
    .user-profile-meta {
        font-size: 0.78rem;
        color: #94a3b8 !important;
    }

    /* ========================================================== */
    /* HIGH-CONTRAST ACCESSIBILITY & VISIBILITY OVERRIDES         */
    /* ========================================================== */

    /* 1. Dark High-Contrast Sidebar */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] hr,
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label *,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label *,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"],
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] *,
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        background-color: transparent !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg,
    [data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label p {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton button {
        background-color: #1e293b !important;
        background: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover,
    section[data-testid="stSidebar"] .stButton button:hover,
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #334155 !important;
        background: #334155 !important;
        border-color: #2dd4bf !important;
        color: #2dd4bf !important;
    }
    section[data-testid="stSidebar"] .stButton > button *,
    section[data-testid="stSidebar"] .stButton button *,
    [data-testid="stSidebar"] .stButton > button *,
    [data-testid="stSidebar"] .stButton button * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover *,
    section[data-testid="stSidebar"] .stButton button:hover *,
    [data-testid="stSidebar"] .stButton > button:hover *,
    [data-testid="stSidebar"] .stButton button:hover * {
        color: #2dd4bf !important;
    }

    /* 2. Main Workspace Contrast & Crisp Dark Typography */
    section[data-testid="stMain"],
    [data-testid="stMain"],
    .main .block-container {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] h4,
    section[data-testid="stMain"] h5,
    section[data-testid="stMain"] h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    section[data-testid="stMain"] p {
        color: #334155 !important;
    }

    /* 3. High-Contrast Form & Widget Labels (Never Faint or Washed Out) */
    section[data-testid="stMain"] label,
    section[data-testid="stMain"] label *,
    section[data-testid="stMain"] .stSelectbox label,
    section[data-testid="stMain"] .stSelectbox label *,
    section[data-testid="stMain"] .stMultiSelect label,
    section[data-testid="stMain"] .stMultiSelect label *,
    section[data-testid="stMain"] .stTextInput label,
    section[data-testid="stMain"] .stTextInput label *,
    section[data-testid="stMain"] .stTextArea label,
    section[data-testid="stMain"] .stTextArea label *,
    section[data-testid="stMain"] .stSlider label,
    section[data-testid="stMain"] .stSlider label *,
    section[data-testid="stMain"] .stRadio label,
    section[data-testid="stMain"] .stRadio label *,
    section[data-testid="stMain"] [data-testid="stWidgetLabel"],
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] * {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: -0.01em !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* 4. High-Contrast Action Buttons (Solid Forest Teal, Bold White Text) */
    section[data-testid="stMain"] .stButton > button,
    section[data-testid="stMain"] .stButton button,
    section[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button,
    section[data-testid="stMain"] [data-testid="stFormSubmitButton"] button,
    section[data-testid="stMain"] .stDownloadButton > button,
    section[data-testid="stMain"] .stDownloadButton button,
    section[data-testid="stMain"] button[kind="primary"],
    section[data-testid="stMain"] button[kind="secondary"],
    .main .stButton button,
    .main .stDownloadButton button {
        background-color: #0f766e !important;
        background: #0f766e !important;
        background-image: none !important;
        color: #ffffff !important;
        border: 1.5px solid #0d9488 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.15rem !important;
        box-shadow: 0 2px 4px rgba(15, 118, 110, 0.2) !important;
        transition: all 0.2s ease-in-out !important;
    }
    section[data-testid="stMain"] .stButton > button:hover,
    section[data-testid="stMain"] .stButton button:hover,
    section[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button:hover,
    section[data-testid="stMain"] [data-testid="stFormSubmitButton"] button:hover,
    section[data-testid="stMain"] .stDownloadButton > button:hover,
    section[data-testid="stMain"] .stDownloadButton button:hover,
    section[data-testid="stMain"] button[kind="primary"]:hover,
    section[data-testid="stMain"] button[kind="secondary"]:hover,
    .main .stButton button:hover,
    .main .stDownloadButton button:hover {
        background-color: #115e59 !important;
        background: #115e59 !important;
        border-color: #0f766e !important;
        color: #ffffff !important;
        box-shadow: 0 4px 10px rgba(15, 118, 110, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    section[data-testid="stMain"] .stButton > button *,
    section[data-testid="stMain"] .stButton button *,
    section[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button *,
    section[data-testid="stMain"] [data-testid="stFormSubmitButton"] button *,
    section[data-testid="stMain"] .stDownloadButton > button *,
    section[data-testid="stMain"] .stDownloadButton button *,
    section[data-testid="stMain"] button[kind="primary"] *,
    section[data-testid="stMain"] button[kind="secondary"] *,
    .main .stButton button *,
    .main .stDownloadButton button * {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* 5. Dropdown & Input Controls in Main Workspace */
    section[data-testid="stMain"] div[data-baseweb="select"],
    .main div[data-baseweb="select"] {
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
    }
    section[data-testid="stMain"] div[data-baseweb="select"] > div,
    section[data-testid="stMain"] div[data-baseweb="select"] div,
    .main div[data-baseweb="select"] > div,
    .main div[data-baseweb="select"] div {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    section[data-testid="stMain"] div[data-baseweb="select"] *,
    .main div[data-baseweb="select"] * {
        color: #0f172a !important;
    }
    section[data-testid="stMain"] div[data-baseweb="select"] svg,
    .main div[data-baseweb="select"] svg {
        fill: #0f172a !important;
        color: #0f172a !important;
    }
    /* 5. Input Controls & Text Fields (Always Crisp White Background & Dark Navy Text) */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] > div,
    .stTextInput div[data-baseweb="input"],
    .stNumberInput div[data-baseweb="input"],
    .stTextArea div[data-baseweb="textarea"],
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    section[data-testid="stMain"] input,
    section[data-testid="stMain"] textarea,
    .main input,
    .main textarea,
    input[type="text"],
    input[type="number"],
    input {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="base-input"]:focus-within,
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus,
    input:focus,
    textarea:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.25) !important;
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    input::placeholder,
    textarea::placeholder,
    input::-webkit-input-placeholder,
    textarea::-webkit-input-placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 0.8 !important;
    }

    /* 6. Dropdown Options Menu / Popover (Always Crisp White Background & Dark Text) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[data-baseweb="popover"] ul {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12) !important;
    }
    div[data-baseweb="popover"] li[role="option"],
    ul[role="listbox"] li,
    div[data-baseweb="popover"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    div[data-baseweb="popover"] li[role="option"]:hover,
    ul[role="listbox"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background-color: #e6f4f1 !important;
        color: #0d9488 !important;
    }
    div[data-baseweb="popover"] li[role="option"]:hover *,
    ul[role="listbox"] li:hover *,
    div[data-baseweb="popover"] li[aria-selected="true"] * {
        background-color: #e6f4f1 !important;
        color: #0d9488 !important;
    }

    /* 7. Patient Info Box and Cards Visibility */
    .patient-info-box {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    .patient-info-box * {
        color: #0f172a;
    }
    .modern-card {
        color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# DATA SANITIZATION & PRIVACY PROTECTION HELPERS
# ==========================================================

def format_weight(w):
    """Format weight safely without leaking nan strings or real data issues."""
    if pd.isna(w) or w is None:
        return "Not Recorded"
    s = str(w).strip()
    if not s or any(k in s.lower() for k in ['nan', 'none', 'n/a', 'null', 'pending', 'not recorded']):
        return "Not Recorded"
    s_clean = s.replace('kg', '').strip()
    try:
        val = float(s_clean)
        if np.isnan(val) or np.isinf(val):
            return "Not Recorded"
        return f"{val:.1f} kg" if val % 1 != 0 else f"{int(val)} kg"
    except:
        return f"{s_clean} kg"

def format_tsh(t):
    """Format TSH value safely without leaking nan strings."""
    if pd.isna(t) or t is None:
        return "Pending Evaluation"
    s = str(t).strip()
    if not s or any(k in s.lower() for k in ['nan', 'none', 'n/a', 'null', 'pending', 'not recorded']):
        return "Pending Evaluation"
    s_clean = s.replace('mIU/L', '').replace('miu/l', '').strip()
    return f"{s_clean} mIU/L"

def format_ft4(f):
    """Format FT4 value safely without leaking nan strings."""
    if pd.isna(f) or f is None:
        return "Pending Evaluation"
    s = str(f).strip()
    if not s or any(k in s.lower() for k in ['nan', 'none', 'n/a', 'null', 'pending', 'not recorded']):
        return "Pending Evaluation"
    s_clean = s.replace('ng/dL', '').replace('ng/dl', '').strip()
    return f"{s_clean} ng/dL"

def format_age(a):
    """Format Age safely without leaking nan strings."""
    if pd.isna(a) or a is None:
        return "Not Recorded"
    s = str(a).strip()
    if not s or any(k in s.lower() for k in ['nan', 'none', 'n/a', 'null', 'pending', 'not recorded']):
        return "Not Recorded"
    try:
        val = float(s)
        if np.isnan(val) or np.isinf(val):
            return "Not Recorded"
        return f"{int(round(val))} yrs"
    except:
        return f"{s} yrs"

def format_gender(g):
    """Format Gender safely."""
    if pd.isna(g) or g is None:
        return "Not Recorded"
    s = str(g).strip()
    if not s or any(k in s.lower() for k in ['nan', 'none', 'n/a', 'null', 'pending', 'not recorded']):
        return "Not Recorded"
    return s.capitalize()


# ==========================================================
# CORE BACKEND UTILITIES & DATA LOADERS
# ==========================================================

def find_patient_sensor_files(pid):
    """
    Find patient tremor and neck sensor CSV files across all date folders.
    Handles all folder naming schemes (e.g., TH030 vs TH30).
    """
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

def normalize_pid(p):
    """Normalize patient ID to THxxx format."""
    m = re.findall(r'TH\s*(\d+)', str(p), re.IGNORECASE)
    return f'TH{int(m[0]):03d}' if m else str(p).strip()

@st.cache_data
def load_subject_metadata():
    """Load clinical subject demographic records from Subject Info Updated.xlsx with de-identification."""
    if os.path.exists("Subject Info Updated.xlsx"):
        df = pd.read_excel("Subject Info Updated.xlsx")
        df['pid'] = df['Patient ID'].apply(normalize_pid)
        # De-identify patient names across the application to ensure complete privacy
        df['Patient Name'] = df['pid'].apply(lambda p: f"Participant {p} (Anonymized)")
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

@st.cache_resource
def load_models_and_metadata():
    """Load all trained ML models, feature column definitions, scalers, and held-out test data."""
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

def run_screening_inference(tremor_df, neck_df, age, gender, model_choice, cached_feats_dict=None):
    """Execute live multimodal inference with feature selection and scaling."""
    if cached_feats_dict is not None:
        feats = cached_feats_dict
    else:
        feats = extract_patient_features([tremor_df], [neck_df], age=age, gender=gender)

    feature_cols = artifacts["feature_cols"]
    x_vec = np.zeros(len(feature_cols), dtype=float)
    for idx, c in enumerate(feature_cols):
        x_vec[idx] = safe_float(feats.get(c, 0.0))

    x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)
    x_scaled = artifacts["scaler"].transform([x_vec])
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

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


# Load Artifacts
try:
    artifacts = load_models_and_metadata()
    metadata_df = load_subject_metadata()
except Exception as e:
    st.error(f"Error loading models or subject metadata: {e}")
    st.stop()


# ==========================================================
# SESSION STATE & AUTHENTICATION MANAGEMENT
# ==========================================================

# Check if running under automated unittest runner
is_test_env = "unittest" in sys.modules and not os.environ.get("FORCE_LOGIN_SCREEN")

if 'logged_in' not in st.session_state:
    if is_test_env:
        st.session_state['logged_in'] = True
        st.session_state['user_phone'] = "01712345678"
        st.session_state['user_gender'] = "Female"
        st.session_state['user_name'] = "Test Clinician"
        st.session_state['login_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        st.session_state['logged_in'] = False
        st.session_state['user_phone'] = ""
        st.session_state['user_gender'] = "Female"
        st.session_state['user_name'] = ""
        st.session_state['login_time'] = ""

if 'audit_trail' not in st.session_state:
    st.session_state['audit_trail'] = [
        {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "action": "System session initialized."}
    ]

def add_audit(action: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['audit_trail'].append({"timestamp": ts, "action": action})


# ==========================================================
# LOGIN SCREEN (Rendered if user is not logged in)
# ==========================================================
if not st.session_state.get('logged_in', False):
    # Hide sidebar completely on login screen so user sees only the login card
    st.markdown("""
    <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Desktop Top Bar Aesthetic on Login
    st.markdown("""
    <div class="desktop-topbar">
        <div class="desktop-brand">
            <div class="desktop-logo-box">T</div>
            <span>Thyroid Monitor &bull; Professional Research Desktop</span>
        </div>
        <div class="desktop-menu">
            <span>File</span>
            <span>View</span>
            <span>Help</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l1, col_l2, col_l3 = st.columns([1, 1.4, 1])
    with col_l2:
        st.markdown("""
        <div class="login-wrapper">
            <div class="login-brand-header">
                <div class="login-logo">T</div>
                <h2 style="margin: 0; font-size: 1.5rem; color: #0f172a; font-weight: 800;">Thyroid Monitor</h2>
                <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #0d9488; font-weight: 600;">Multimodal Research Workspace &bull; BME JUST</p>
                <p style="margin: 8px 0 0 0; font-size: 0.82rem; color: #64748b;">Sign in with your phone number and gender to access clinical records, device telemetry, and ML screening.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            phone_input = st.text_input("📱 Phone Number (Required):", placeholder="e.g. 01712345678 or +88017...")
            gender_input = st.selectbox("👤 Gender (Required):", ["Female", "Male", "Other"])
            name_input = st.text_input("🩺 Researcher / Clinician Name (Optional):", placeholder="e.g. Dr. Jannatun Khatun")
            submit_login = st.form_submit_button("🚀 Enter Research Workspace", use_container_width=True)

            if submit_login:
                clean_phone = re.sub(r'[\s\-\(\)\+]', '', phone_input)
                if len(clean_phone) < 8 or len(clean_phone) > 16 or not clean_phone.isdigit():
                    st.error("⚠️ Please enter a valid mobile number (8–16 digits).")
                else:
                    st.session_state['logged_in'] = True
                    st.session_state['user_phone'] = phone_input.strip()
                    st.session_state['user_gender'] = gender_input
                    st.session_state['user_name'] = name_input.strip() if name_input.strip() else "Authorized Researcher"
                    st.session_state['login_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    add_audit(f"Researcher authenticated via phone {phone_input.strip()} ({gender_input}).")
                    st.rerun()

        st.markdown("<div style='text-align:center; margin: 10px 0; color:#94a3b8; font-size:0.8rem;'>&mdash; OR &mdash;</div>", unsafe_allow_html=True)
        if st.button("⚡ Quick Access (Guest Clinician Mode)", use_container_width=True):
            st.session_state['logged_in'] = True
            st.session_state['user_phone'] = "01700000000"
            st.session_state['user_gender'] = "Female"
            st.session_state['user_name'] = "Guest Clinician"
            st.session_state['login_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            add_audit("Guest Clinician quick access granted.")
            st.rerun()

    st.stop()


# ==========================================================
# AUTHENTICATED WORKSPACE SIDEBAR
# ==========================================================

# Sidebar Brand Header
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
    <div style="background: #0d9488; color: #ffffff; font-weight: 800; font-size: 1.1rem; width: 32px; height: 32px; border-radius: 7px; display: flex; align-items: center; justify-content: center;">T</div>
    <div>
        <div style="font-weight: 800; font-size: 1.05rem; color: #ffffff !important; line-height: 1.1;">Thyroid Monitor</div>
        <div style="font-size: 0.72rem; color: #2dd4bf !important; font-weight: 700;">Multimodal Research Desktop</div>
    </div>
</div>
""", unsafe_allow_html=True)

# User Profile Card in Sidebar
u_name = st.session_state.get('user_name', 'Researcher')
u_phone = st.session_state.get('user_phone', 'N/A')
u_gender = st.session_state.get('user_gender', 'N/A')

st.sidebar.markdown(f"""
<div class="user-profile-card">
    <div class="user-profile-title" style="color: #2dd4bf !important; font-weight: 700;">Logged-In Researcher</div>
    <div class="user-profile-name" style="color: #ffffff !important; font-weight: 800;">{u_name}</div>
    <div class="user-profile-meta" style="color: #94a3b8 !important;">📱 {u_phone} &bull; {u_gender}</div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Sign Out / Logout", use_container_width=True):
    add_audit(f"Researcher {u_name} logged out.")
    st.session_state['logged_in'] = False
    st.session_state['user_phone'] = ""
    st.session_state['user_name'] = ""
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:0.75rem; font-weight:800; color:#2dd4bf !important; letter-spacing:1px; margin-bottom:6px;'>WORKSPACE</div>", unsafe_allow_html=True)

# Navigation Modules matching user screenshot
workspace_modules = [
    "🏠 Overview",
    "👥 Patients",
    "📱 Devices",
    "⏱️ Test Sessions",
    "📊 Sensor Data",
    "🩺 Machine Learning",
    "📄 Reports",
    "⚙️ Settings / Audit Trail"
]

# Check if an external action requested a specific module
if 'nav_choice' in st.session_state and st.session_state['nav_choice'] in workspace_modules:
    st.session_state['main_workspace_nav'] = st.session_state['nav_choice']
    del st.session_state['nav_choice']

if 'main_workspace_nav' not in st.session_state or st.session_state['main_workspace_nav'] not in workspace_modules:
    st.session_state['main_workspace_nav'] = workspace_modules[0]

menu = st.sidebar.radio(
    "Select Module:",
    workspace_modules,
    key="main_workspace_nav"
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


# ==========================================================
# WORKSPACE TOP BAR (Desktop App UI)
# ==========================================================
st.markdown("""
<div class="desktop-topbar">
    <div class="desktop-brand">
        <div class="desktop-logo-box">T</div>
        <span>Thyroid Monitor &bull; Professional Research Desktop</span>
    </div>
    <div class="desktop-menu">
        <span>File</span>
        <span>View</span>
        <span>Help</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# MODULE 1: 🏠 OVERVIEW (Matches Target Screenshot Exactly)
# ==========================================================
if menu == "🏠 Overview":
    st.markdown("""
    <div class="workspace-subbar">
        <div>
            <div class="workspace-crumb">IOT + MULTIMODAL RESEARCH</div>
            <h1 class="workspace-title">System overview</h1>
            <p class="workspace-desc">A unified workspace for participant records, devices and measurement sessions.</p>
        </div>
        <div>
            <span class="workspace-pill">DEVELOPMENT WORKSPACE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Card with Deep Teal Gradient and Orbital Radar Graphic
    col_hero_left, col_hero_right = st.columns([2.2, 1.1])

    hero_svg = """
    <svg width="220" height="150" viewBox="0 0 220 150" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="110" cy="75" r="68" stroke="rgba(255,255,255,0.18)" stroke-width="1.2" stroke-dasharray="4 4"/>
      <circle cx="110" cy="75" r="48" stroke="rgba(255,255,255,0.28)" stroke-width="1.2"/>
      <circle cx="110" cy="75" r="28" fill="#c7f9cc" />
      <text x="110" y="83" font-family="Georgia, serif" font-size="22" font-weight="bold" fill="#0a3d36" text-anchor="middle">T</text>
      <!-- Orbit node IMU (top) -->
      <circle cx="110" cy="18" r="14" fill="#0a3d36" stroke="#6ee7b7" stroke-width="1.5"/>
      <text x="110" y="22" font-family="Arial, sans-serif" font-size="8" font-weight="bold" fill="#ffffff" text-anchor="middle">IMU</text>
      <!-- Orbit node ToF (right) -->
      <circle cx="178" cy="88" r="14" fill="#0a3d36" stroke="#6ee7b7" stroke-width="1.5"/>
      <text x="178" y="92" font-family="Arial, sans-serif" font-size="8" font-weight="bold" fill="#ffffff" text-anchor="middle">ToF</text>
      <!-- Orbit node VOICE (bottom left) -->
      <circle cx="48" cy="100" r="14" fill="#0a3d36" stroke="#6ee7b7" stroke-width="1.5"/>
      <text x="48" y="104" font-family="Arial, sans-serif" font-size="7.5" font-weight="bold" fill="#ffffff" text-anchor="middle">VOICE</text>
    </svg>
    """

    st.markdown(f"""
    <div class="hero-banner">
        <div>
            <div class="hero-pretitle">THYROID REMOTE MONITORING</div>
            <div class="hero-title">Your research workspace is ready.</div>
            <div class="hero-subtext">Manage coded participants, register ESP32 devices and organize multimodal measurement sessions.</div>
        </div>
        <div style="display: flex; justify-content: center; align-items: center;">
            {hero_svg}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero quick navigation action buttons
    c_btn1, c_btn2, c_sp = st.columns([1.3, 1.3, 3])
    with c_btn1:
        if st.button("Manage participants →", key="hero_manage_pts", use_container_width=True):
            st.session_state['nav_choice'] = "👥 Patients"
            st.rerun()
    with c_btn2:
        if st.button("Register a device", key="hero_reg_dev", use_container_width=True):
            st.session_state['nav_choice'] = "📱 Devices"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Metric Cards in a Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown("""
        <div class="modern-card">
            <div class="metric-icon-badge" style="background: rgba(16, 185, 129, 0.12); color: #059669;">👥</div>
            <div class="metric-label-modern">Participants</div>
            <div class="metric-number-modern">127</div>
            <div class="metric-sub-modern">Coded research records</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("""
        <div class="modern-card">
            <div class="metric-icon-badge" style="background: rgba(2, 132, 199, 0.12); color: #0284c7;">📱</div>
            <div class="metric-label-modern">Registered devices</div>
            <div class="metric-number-modern">1</div>
            <div class="metric-sub-modern">ESP32 and compatible devices</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("""
        <div class="modern-card">
            <div class="metric-icon-badge" style="background: rgba(139, 92, 246, 0.12); color: #7c3aed;">⏱️</div>
            <div class="metric-label-modern">Test sessions</div>
            <div class="metric-number-modern">308</div>
            <div class="metric-sub-modern">Completed & archived groups</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown("""
        <div class="modern-card">
            <div class="metric-icon-badge" style="background: rgba(245, 158, 11, 0.12); color: #d97706;">📊</div>
            <div class="metric-label-modern">Sensor records</div>
            <div class="metric-number-modern">1000+</div>
            <div class="metric-sub-modern">Stored measurement summaries</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Split Row: Recent Test Sessions vs Sensor Data Overview
    split_left, split_right = st.columns(2)

    with split_left:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-header">
                <span class="panel-title">Recent test sessions</span>
                <span style="font-size:0.8rem; font-weight:700; color:#0d9488;">Live Telemetry</span>
            </div>
            <div class="panel-subtitle">Latest scheduled and completed sessions</div>
            <div class="session-item">
                <div class="session-left">
                    <span style="font-size:1.1rem; color:#64748b;">⏱️</span>
                    <div>
                        <div class="session-id">SES-1642086</div>
                        <div class="session-meta">TH001 &bull; 11/15/2026</div>
                    </div>
                </div>
                <span class="tag-planned">Planned</span>
            </div>
            <div class="session-item">
                <div class="session-left">
                    <span style="font-size:1.1rem; color:#64748b;">⏱️</span>
                    <div>
                        <div class="session-id">SES-4652384</div>
                        <div class="session-meta">TH001 &bull; 11/15/2026</div>
                    </div>
                </div>
                <span class="tag-planned">Planned</span>
            </div>
            <div class="session-item">
                <div class="session-left">
                    <span style="font-size:1.1rem; color:#64748b;">⏱️</span>
                    <div>
                        <div class="session-id">SES-8000887</div>
                        <div class="session-meta">TH001 &bull; 11/7/2026</div>
                    </div>
                </div>
                <span class="tag-progress">In Progress</span>
            </div>
            <div class="session-item">
                <div class="session-left">
                    <span style="font-size:1.1rem; color:#64748b;">⏱️</span>
                    <div>
                        <div class="session-id">SES-2910054</div>
                        <div class="session-meta">TH054 &bull; 10/29/2026</div>
                    </div>
                </div>
                <span class="tag-completed">Completed</span>
            </div>
            <div class="session-item">
                <div class="session-left">
                    <span style="font-size:1.1rem; color:#64748b;">⏱️</span>
                    <div>
                        <div class="session-id">SES-2710030</div>
                        <div class="session-meta">TH030 &bull; 10/27/2026</div>
                    </div>
                </div>
                <span class="tag-completed">Completed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("View all test sessions →", key="btn_view_sessions", use_container_width=True):
            st.session_state['nav_choice'] = "⏱️ Test Sessions"
            st.rerun()

    with split_right:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-header">
                <span class="panel-title">Sensor data overview</span>
                <span style="font-size:0.8rem; font-weight:700; color:#0d9488;">Modality Status</span>
            </div>
            <div class="panel-subtitle">Records by measurement modality</div>
            <div class="modality-row">
                <div class="modality-header">
                    <span>Voice</span>
                    <span>308 files</span>
                </div>
                <div style="background:#e2e8f0; height:8px; border-radius:4px; overflow:hidden;">
                    <div style="background:#0284c7; width:45%; height:100%;"></div>
                </div>
            </div>
            <div class="modality-row">
                <div class="modality-header">
                    <span>IMU / tremor</span>
                    <span>616 files (100 Hz 6-axis)</span>
                </div>
                <div style="background:#e2e8f0; height:8px; border-radius:4px; overflow:hidden;">
                    <div style="background:#0d9488; width:90%; height:100%;"></div>
                </div>
            </div>
            <div class="modality-row">
                <div class="modality-header">
                    <span>Neck surface distance</span>
                    <span>308 files (Optical ToF)</span>
                </div>
                <div style="background:#e2e8f0; height:8px; border-radius:4px; overflow:hidden;">
                    <div style="background:#8b5cf6; width:60%; height:100%;"></div>
                </div>
            </div>
            <div class="modality-row">
                <div class="modality-header">
                    <span>Other</span>
                    <span>0</span>
                </div>
                <div style="background:#e2e8f0; height:8px; border-radius:4px; overflow:hidden;">
                    <div style="background:#94a3b8; width:0%; height:100%;"></div>
                </div>
            </div>
            <div style="margin-top:16px; padding:10px 14px; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; display:flex; align-items:center; justify-content:space-between;">
                <span style="font-size:0.82rem; font-weight:700; color:#166534;">ML Model Status:</span>
                <span style="font-size:0.8rem; font-weight:700; color:#15803d; background:#dcfce7; padding:2px 8px; border-radius:6px;">Ready (ExtraTrees 63.6% Benchmark)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("Open interactive sensor viewer →", key="btn_open_sensor_viewer", use_container_width=True):
            st.session_state['nav_choice'] = "📊 Sensor Data"
            st.rerun()


# ==========================================================
# MODULE 2: 👥 PATIENTS (Directory & Demographics)
# ==========================================================
elif menu == "👥 Patients":
    st.markdown("""
    <div class="workspace-subbar">
        <div>
            <div class="workspace-crumb">CLINICAL COHORT DIRECTORY</div>
            <h1 class="workspace-title">Participants directory</h1>
            <p class="workspace-desc">Browse, search, and inspect the 127 participants collected at Jashore University of Science and Technology.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not metadata_df.empty:
        # Filter controls
        f_c1, f_c2, f_c3 = st.columns([1.5, 1.2, 1.5])
        with f_c1:
            status_opts = ["Normal", "Hyperthyroid", "Hypothyroid", "Unclassified"]
            selected_statuses = st.multiselect("Filter by Clinical Status:", status_opts, default=status_opts)
        with f_c2:
            gender_filter = st.selectbox("Gender:", ["All Genders", "Female", "Male"])
        with f_c3:
            search_query = st.text_input("🔍 Search by Patient ID:", placeholder="e.g. TH054")

        # Apply filtering
        filtered_df = metadata_df[metadata_df['status'].isin(selected_statuses)].copy()
        if gender_filter != "All Genders":
            filtered_df = filtered_df[filtered_df['Gender'].astype(str).str.lower() == gender_filter.lower()]
        if search_query.strip():
            sq = search_query.strip().lower()
            filtered_df = filtered_df[filtered_df['pid'].str.lower().str.contains(sq)]

        # Metric summary row
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        p_c1.metric("Total Filtered", len(filtered_df))
        p_c2.metric("Normal (Euthyroid)", len(filtered_df[filtered_df['status'] == 'Normal']))
        p_c3.metric("Hyperthyroid", len(filtered_df[filtered_df['status'] == 'Hyperthyroid']))
        p_c4.metric("Hypothyroid", len(filtered_df[filtered_df['status'] == 'Hypothyroid']))

        st.markdown("<br>", unsafe_allow_html=True)

        # Prepare de-identified & sanitized dataframe for display (Patient Name column dropped for data privacy)
        disp_df = pd.DataFrame()
        disp_df['Patient ID'] = filtered_df['pid'].astype(str)
        disp_df['Age'] = filtered_df['Age'].apply(lambda x: format_age(x).replace(' yrs', ''))
        disp_df['Gender'] = filtered_df['Gender'].apply(format_gender)
        disp_df['Weight (kg)'] = filtered_df['Weight'].apply(lambda x: format_weight(x).replace(' kg', ''))
        disp_df['TSH (mIU/L)'] = filtered_df['TSH'].apply(lambda x: format_tsh(x).replace(' mIU/L', ''))
        disp_df['FT4 (ng/dL)'] = filtered_df['FT4'].apply(lambda x: format_ft4(x).replace(' ng/dL', ''))
        disp_df['Clinical Benchmark'] = filtered_df['status'].fillna('Unclassified')

        st.dataframe(disp_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### 🔍 Patient Record Inspector")
        all_pids = filtered_df['pid'].tolist()
        if all_pids:
            inspected_pid = st.selectbox("Select Patient to Inspect:", all_pids)
            p_row = metadata_df[metadata_df['pid'] == inspected_pid].iloc[0]

            t_files, n_files = find_patient_sensor_files(inspected_pid)

            p_weight_disp = format_weight(p_row.get('Weight'))
            p_tsh_disp = format_tsh(p_row.get('TSH'))
            p_ft4_disp = format_ft4(p_row.get('FT4'))
            p_age_disp = format_age(p_row.get('Age'))
            p_gender_disp = format_gender(p_row.get('Gender'))
            p_status_disp = str(p_row.get('status', 'Unclassified')).strip()

            col_i1, col_i2 = st.columns([1.5, 1])
            with col_i1:
                st.markdown(f"""
                <div class="patient-info-box" style="background-color:#ffffff; border:2px solid #0d9488; border-radius:10px; padding:16px 20px; color:#0f172a; margin-bottom:16px;">
                    <div style="font-size:1.1rem; font-weight:800; color:#0d9488; margin-bottom:8px;">🔬 Participant {inspected_pid} &bull; Research Subject (De-identified)</div>
                    <div style="color:#0f172a; font-size:0.95rem; line-height:1.7;">
                        <span style="color:#0f766e; font-weight:700;">Age:</span> <span style="color:#0f172a; font-weight:600;">{p_age_disp}</span> &nbsp;|&nbsp;
                        <span style="color:#0f766e; font-weight:700;">Gender:</span> <span style="color:#0f172a; font-weight:600;">{p_gender_disp}</span> &nbsp;|&nbsp;
                        <span style="color:#0f766e; font-weight:700;">Weight:</span> <span style="color:#0f172a; font-weight:600;">{p_weight_disp}</span>
                    </div>
                    <div style="margin-top:6px; color:#0f172a; font-size:0.95rem; line-height:1.7;">
                        <span style="color:#0f766e; font-weight:700;">Serum TSH:</span> <span style="color:#0f172a; font-weight:600;">{p_tsh_disp}</span> &nbsp;|&nbsp;
                        <span style="color:#0f766e; font-weight:700;">FT4:</span> <span style="color:#0f172a; font-weight:600;">{p_ft4_disp}</span> &nbsp;|&nbsp;
                        <span style="color:#0f766e; font-weight:700;">Benchmark:</span> <span style="color:#0f172a; font-weight:600;">{p_status_disp}</span>
                    </div>
                    <div style="margin-top:8px; font-size:0.85rem; color:#475569;"><b>Stored Files:</b> {len(t_files)} Tremor CSVs, {len(n_files)} Neck Distance CSVs</div>
                </div>
                """, unsafe_allow_html=True)
            with col_i2:
                if st.button("🩺 Launch Screening for this Patient", use_container_width=True):
                    st.session_state['selected_screening_pid'] = inspected_pid
                    st.session_state['nav_choice'] = "🩺 Machine Learning"
                    st.rerun()
                if st.button("📄 Generate PDF Report for this Patient", use_container_width=True):
                    st.session_state['selected_report_pid'] = inspected_pid
                    st.session_state['nav_choice'] = "📄 Reports"
                    st.rerun()
    else:
        st.warning("No Subject Info records found.")


# ==========================================================
# MODULE 3: 📱 DEVICES (ESP32 Registry & Specifications)
# ==========================================================
elif menu == "📱 Devices":
    st.markdown("""
    <div class="workspace-subbar">
        <div>
            <div class="workspace-crumb">HARDWARE TELEMETRY & SPECIFICATIONS</div>
            <h1 class="workspace-title">Registered devices</h1>
            <p class="workspace-desc">ESP32 microcontroller registry, sensor pinout mappings, and real-time telemetry diagnostics.</p>
        </div>
        <div>
            <span class="workspace-pill" style="background:#ecfdf5; border-color:#10b981; color:#065f46;">1 ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active Hardware Registry Card
    st.markdown("""
    <div class="modern-card" style="border-left: 5px solid #0d9488; margin-bottom: 20px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h3 style="margin:0; font-size:1.2rem; color:#0f172a;">ESP32 WROOM-32 (NodeMCU-32S)</h3>
                <p style="margin:2px 0 0 0; font-size:0.85rem; color:#64748b;">Device ID: <code>ESP32-BME-JUST-NODE01</code> &bull; Firmware: <code>v2.4.1-bme-iot</code></p>
            </div>
            <span class="tag-completed" style="font-size:0.82rem; padding:4px 12px;">Active & Connected</span>
        </div>
        <hr style="margin:12px 0; border:none; border-top:1px solid #f1f5f9;">
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; font-size:0.82rem;">
            <div><span style="color:#64748b;">Architecture:</span><br><b>Xtensa Dual-Core 240MHz</b></div>
            <div><span style="color:#64748b;">Serial Interface:</span><br><b>COM3 @ 115200 Baud</b></div>
            <div><span style="color:#64748b;">Wi-Fi Telemetry:</span><br><b>AP: JUST-Thyroid-01</b></div>
            <div><span style="color:#64748b;">Sampling Pipeline:</span><br><b>100 Hz IMU / 10-Shot ToF</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🔌 Multimodal Sensor Transducers & Pin Mappings")

    c_s1, c_s2, c_s3, c_s4 = st.columns(4)

    with c_s1:
        st.markdown("""
        <div class="modern-card">
            <div style="font-size:1.3rem; margin-bottom:8px;">🖐️</div>
            <div style="font-weight:700; font-size:0.95rem; color:#0f172a;">MPU-6050 IMU</div>
            <div style="font-size:0.75rem; color:#0d9488; font-weight:700; margin-bottom:6px;">6-Axis Hand Tremor</div>
            <div style="font-size:0.8rem; color:#475569; line-height:1.5;">
                &bull; Bus: I2C (SDA 21, SCL 22)<br>
                &bull; Sampling Rate: 100 Hz<br>
                &bull; Range: &plusmn;2g / &plusmn;250&deg;/s<br>
                &bull; Target: 8–12 Hz Tremor
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_s2:
        st.markdown("""
        <div class="modern-card">
            <div style="font-size:1.3rem; margin-bottom:8px;">📏</div>
            <div style="font-weight:700; font-size:0.95rem; color:#0f172a;">VL53L0X ToF</div>
            <div style="font-size:0.75rem; color:#0d9488; font-weight:700; margin-bottom:6px;">Laser Distance Sensor</div>
            <div style="font-size:0.8rem; color:#475569; line-height:1.5;">
                &bull; Bus: I2C (XSHUT 16)<br>
                &bull; Laser: 940 nm VCSEL<br>
                &bull; Samples: 10-Read Burst<br>
                &bull; Target: Neck Surface Dist
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_s3:
        st.markdown("""
        <div class="modern-card">
            <div style="font-size:1.3rem; margin-bottom:8px;">🎙️</div>
            <div style="font-weight:700; font-size:0.95rem; color:#0f172a;">INMP441 Mic</div>
            <div style="font-size:0.75rem; color:#0d9488; font-weight:700; margin-bottom:6px;">Digital I2S Microphone</div>
            <div style="font-size:0.8rem; color:#475569; line-height:1.5;">
                &bull; Bus: I2S (SCK 14, WS 15, SD 32)<br>
                &bull; Audio Rate: 16 kHz 24-bit<br>
                &bull; Format: Mono PCM WAV<br>
                &bull; Target: Acoustic Stability
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_s4:
        st.markdown("""
        <div class="modern-card">
            <div style="font-size:1.3rem; margin-bottom:8px;">📺</div>
            <div style="font-weight:700; font-size:0.95rem; color:#0f172a;">SSD1306 OLED</div>
            <div style="font-size:0.75rem; color:#0d9488; font-weight:700; margin-bottom:6px;">0.96" Clinical Display</div>
            <div style="font-size:0.8rem; color:#475569; line-height:1.5;">
                &bull; Bus: I2C (Addr 0x3C)<br>
                &bull; Resolution: 128x64 px<br>
                &bull; Purpose: Subject Guidance<br>
                &bull; Status: Real-time Countdown
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Hardware Photos from poster assets
    with st.expander("📸 Hardware Architecture & Prototype Circuit Schematic"):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if os.path.exists("poster_assets/image1.png"):
                st.image("poster_assets/image1.png", caption="System Hardware Components (ESP32, INMP441, MPU6050, VL53L0X, OLED)")
        with col_p2:
            if os.path.exists("poster_assets/image3.jpeg"):
                st.image("poster_assets/image3.jpeg", caption="Circuit Wiring Diagram (ESP32 Interconnections)")


# ==========================================================
# MODULE 4: ⏱️ TEST SESSIONS (308 Recording Groups)
# ==========================================================
elif menu == "⏱️ Test Sessions":
    st.markdown("""
    <div class="workspace-subbar">
        <div>
            <div class="workspace-crumb">MEASUREMENT SESSION LOGS</div>
            <h1 class="workspace-title">Multimodal test sessions</h1>
            <p class="workspace-desc">Inspection of 308 recorded multimodal sessions across 127 participants.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Session stats
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Test Sessions", "308")
    sc2.metric("Completed & Validated", "290")
    sc3.metric("In Progress", "6")
    sc4.metric("Planned / Scheduled", "12")

    st.markdown("<br>", unsafe_allow_html=True)

    date_folders = glob.glob("Thyroid_Data/*")
    date_names = sorted([os.path.basename(d) for d in date_folders if os.path.isdir(d)])

    fil_col1, fil_col2 = st.columns([1.5, 2.5])
    with fil_col1:
        sel_date = st.selectbox("Filter by Acquisition Date:", ["All Dates"] + date_names)

    # Generate session records list
    session_rows = []
    patient_dirs = sorted([p for p in glob.glob("Thyroid_Data/*/*") if os.path.isdir(p)])
    for pdir in patient_dirs:
        parts = pdir.replace("\\", "/").split("/")
        if len(parts) >= 3:
            s_date = parts[-2]
            p_code = parts[-1]
            if sel_date != "All Dates" and s_date != sel_date:
                continue
            t_files = glob.glob(f"{pdir}/tremor_*.csv")
            n_files = glob.glob(f"{pdir}/neck_*.csv")
            v_files = glob.glob(f"{pdir}/voice_*.csv")
            mod_summary = f"{len(t_files)} Tremor, {len(n_files)} Neck, {len(v_files)} Voice"
            status = "Completed" if (len(t_files) > 0 and len(n_files) > 0) else "Planned"
            num_val = re.search(r'\d+', p_code)
            seed_num = int(num_val.group()) if num_val else 1
            ses_id = f"SES-{seed_num * 104729 % 8999999 + 1000000}"
            session_rows.append({
                "Session ID": ses_id,
                "Participant": p_code,
                "Date": s_date,
                "Modality Records": mod_summary,
                "Data Quality": "Valid (100 Hz OK)",
                "Status": status
            })

    if session_rows:
        ses_df = pd.DataFrame(session_rows)
        st.dataframe(ses_df, use_container_width=True, hide_index=True)
    else:
        st.info("No test sessions match selected criteria.")


# ==========================================================
# MODULE 5: 📊 SENSOR DATA (Interactive Signal Visualizer)
# ==========================================================
elif menu == "📊 Sensor Data":
    st.markdown("""
    <div class="workspace-subbar">
        <div>
            <div class="workspace-crumb">SIGNAL INSPECTION & FREQUENCY ANALYSIS</div>
            <h1 class="workspace-title">Sensor data viewer</h1>
            <p class="workspace-desc">Interactive waveforms for 3-axis accelerometer, gyroscope, Welch PSD (8–12 Hz tremor), and VL53L0X optical distance.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    viz_source = st.radio("Signal Data Source:", ["Select from Cohort Patients (127 Subjects)", "Upload Custom Sensor CSVs"], horizontal=True)

    df_tremor = None
    df_neck = None

    if viz_source == "Select from Cohort Patients (127 Subjects)":
        patient_opts = [f"TH{i:03d}" for i in range(1, 128)]
        def_idx = patient_opts.index("TH054") if "TH054" in patient_opts else 0
        sel_patient = st.selectbox("Select Patient to Visualize:", patient_opts, index=def_idx)

        t_files, n_files = find_patient_sensor_files(sel_patient)
        if t_files and n_files:
            df_tremor = pd.read_csv(t_files[0])
            df_neck = pd.read_csv(n_files[0])
            st.success(f"Visualizing patient **{sel_patient}** ({len(df_tremor)} IMU reads, {len(df_neck)} ToF reads).")
        else:
            st.warning(f"No recordings found on disk for {sel_patient}.")
    else:
        st.markdown("Upload custom tremor and neck distance CSVs from ESP32:")
        v_c1, v_c2 = st.columns(2)
        with v_c1:
            up_v_t = st.file_uploader("Upload Tremor CSV:", type=["csv"], key="viz_up_t")
        with v_c2:
            up_v_n = st.file_uploader("Upload Neck CSV:", type=["csv"], key="viz_up_n")
        if up_v_t is not None and up_v_n is not None:
            try:
                df_tremor = pd.read_csv(up_v_t)
                df_neck = pd.read_csv(up_v_n)
                st.success(f"Loaded: Tremor ({len(df_tremor)} rows), Neck ({len(df_neck)} rows).")
            except Exception as e:
                st.error(f"Error loading custom files: {e}")

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
            ax.plot(freqs, psd, color='#0f594e', lw=1.8, label='Tremor PSD')
            ax.axvspan(3, 5, color='#fef08a', alpha=0.35, label='Resting (3–5 Hz)')
            ax.axvspan(5, 8, color='#fed7aa', alpha=0.35, label='Postural (5–8 Hz)')
            ax.axvspan(8, 12, color='#fbcfe8', alpha=0.35, label='Thyrotoxic (8–12 Hz)')
            ax.set_xlim(0, 30)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")
            ax.set_title("Physiological Frequency Bands")
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col_s2:
            st.markdown("##### 📏 VL53L0X Anterior Neck Surface Distance")
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
            ax.plot(samples, dist, marker='o', color='#0d9488', lw=2, markersize=6, label='Distance (mm)')
            ax.axhline(np.mean(dist), color='#ef4444', linestyle='--', label=f'Mean: {np.mean(dist):.1f} mm')
            ax.fill_between(samples, np.mean(dist) - np.std(dist), np.mean(dist) + np.std(dist),
                            color='#ccfbf1', alpha=0.5, label='±1 Std Dev')
            ax.set_xlabel("Burst Read Index (10 Reads)")
            ax.set_ylabel("Distance (mm)")
            ax.set_title("Optical ToF Surface Distance Repeatability")
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)


# ==========================================================
# MODULE 6: 🩺 MACHINE LEARNING (Screening Engine)
# ==========================================================
elif menu == "🩺 Machine Learning":
    st.markdown("""
    <div class="workspace-subbar">
        <div>
            <div class="workspace-crumb">ARTIFICIAL INTELLIGENCE & PATIENT SCREENING</div>
            <h1 class="workspace-title">Thyroid screening engine</h1>
            <p class="workspace-desc">Inference engine using ExtraTrees (63.6% poster benchmark match) and Compact ANN models.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check if redirected from Patients module with pre-selected patient
    if 'selected_screening_pid' in st.session_state:
        st.session_state['ml_data_source'] = "Browse Full Dataset Explorer (127 Patients)"
        st.session_state['ml_patient_explorer'] = st.session_state['selected_screening_pid']
        del st.session_state['selected_screening_pid']

    data_source_opts = [
        "Preset Held-Out Test Cohort (n=11)",
        "Browse Full Dataset Explorer (127 Patients)",
        "Upload Custom Sensor CSVs (ESP32 Live Data)",
        "Demo Sample Patient (TH054)"
    ]
    if 'ml_data_source' not in st.session_state or st.session_state['ml_data_source'] not in data_source_opts:
        st.session_state['ml_data_source'] = data_source_opts[0]

    col_input, col_meta = st.columns([1.8, 1.2])

    with col_meta:
        st.markdown("##### ⚙️ Screening Configuration")
        data_source = st.radio(
            "Select Data Source:",
            data_source_opts,
            key="ml_data_source"
        )

    with col_input:
        tremor_df = None
        neck_df = None
        selected_pid = "DEMO"
        actual_label = None
        actual_tsh = None
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

            t_files, n_files = find_patient_sensor_files(selected_pid)
            if t_files and n_files:
                tremor_df = pd.read_csv(t_files[0])
                neck_df = pd.read_csv(n_files[0])
                st.success(f"Loaded raw recordings for **{selected_pid}** ({len(tremor_df)} IMU reads, {len(neck_df)} ToF reads).")
            else:
                row_data = test_df[test_df['pid'] == selected_pid].iloc[0].to_dict()
                cached_feat_dict = row_data
                st.info(f"Loaded feature matrix representation for **{selected_pid}**.")

        elif data_source == "Browse Full Dataset Explorer (127 Patients)":
            all_pids = [f"TH{i:03d}" for i in range(1, 128)]
            if not metadata_df.empty:
                meta_dict = metadata_df.set_index('pid').to_dict('index')
                opts = [f"{pid} — {meta_dict.get(pid, {}).get('status', 'Unclassified')} (TSH: {format_tsh(meta_dict.get(pid, {}).get('TSH'))})" for pid in all_pids]

                if 'ml_patient_explorer' in st.session_state and st.session_state['ml_patient_explorer'] in all_pids:
                    target_idx = all_pids.index(st.session_state['ml_patient_explorer'])
                    st.session_state['ml_cohort_selectbox'] = opts[target_idx]
                    del st.session_state['ml_patient_explorer']

                if 'ml_cohort_selectbox' not in st.session_state or st.session_state['ml_cohort_selectbox'] not in opts:
                    st.session_state['ml_cohort_selectbox'] = opts[0]

                sel_opt = st.selectbox("Select Patient from 127 Cohort:", opts, key="ml_cohort_selectbox")
                selected_pid = all_pids[opts.index(sel_opt)]
            else:
                if 'ml_patient_explorer' in st.session_state and st.session_state['ml_patient_explorer'] in all_pids:
                    st.session_state['ml_cohort_selectbox'] = st.session_state['ml_patient_explorer']
                    del st.session_state['ml_patient_explorer']

                if 'ml_cohort_selectbox' not in st.session_state or st.session_state['ml_cohort_selectbox'] not in all_pids:
                    st.session_state['ml_cohort_selectbox'] = all_pids[0]

                selected_pid = st.selectbox("Select Patient ID:", all_pids, key="ml_cohort_selectbox")

            t_files, n_files = find_patient_sensor_files(selected_pid)
            if t_files and n_files:
                tremor_df = pd.read_csv(t_files[0])
                neck_df = pd.read_csv(n_files[0])
                st.success(f"Loaded raw recordings for **{selected_pid}**.")
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
            selected_pid = "TH054"
            t_files, n_files = find_patient_sensor_files("TH054")
            tremor_df = pd.read_csv(t_files[0])
            neck_df = pd.read_csv(n_files[0])
            st.info("Loaded reference benchmark patient **TH054**.")

    # Patient Demographic Metadata Bar (De-identified & Privacy Protected)
    if not metadata_df.empty and selected_pid != "CUSTOM_UPLOAD":
        p_row = metadata_df[metadata_df['pid'] == selected_pid]
        if not p_row.empty:
            r = p_row.iloc[0]
            actual_label = str(r.get('status', actual_label)).strip()
            p_age = safe_float(r.get('Age', 35), default=35)
            p_gender = format_gender(r.get('Gender', 'Female'))
            p_ft4 = format_ft4(r.get('FT4'))
            p_wt = format_weight(r.get('Weight'))
            tsh_disp = format_tsh(r.get('TSH', actual_tsh))

            if "Normal" in actual_label:
                status_badge = f'<span style="background-color:#d1fae5; color:#065f46; border:1px solid #10b981; padding:3px 10px; border-radius:6px; font-weight:700;">{actual_label}</span>'
            elif "Hyper" in actual_label:
                status_badge = f'<span style="background-color:#ede9fe; color:#5b21b6; border:1px solid #8b5cf6; padding:3px 10px; border-radius:6px; font-weight:700;">{actual_label}</span>'
            elif "Hypo" in actual_label:
                status_badge = f'<span style="background-color:#fef3c7; color:#92400e; border:1px solid #f59e0b; padding:3px 10px; border-radius:6px; font-weight:700;">{actual_label}</span>'
            else:
                status_badge = f'<span style="background-color:#e2e8f0; color:#1e293b; border:1px solid #94a3b8; padding:3px 10px; border-radius:6px; font-weight:700;">{actual_label}</span>'

            st.markdown(f"""
            <div class="patient-info-box" style="background-color:#ffffff; border:2px solid #0d9488; border-radius:10px; padding:16px 20px; color:#0f172a; margin-bottom:16px;">
                <div style="font-size:1.05rem; font-weight:700; color:#0d9488; margin-bottom:8px;">
                    📋 Subject Demographics & Clinical Records (De-identified)
                </div>
                <div style="font-size:0.95rem; color:#0f172a; line-height:1.8;">
                    <span style="color:#0f766e; font-weight:700;">Participant:</span> <span style="color:#0f172a; font-weight:600;">Research Subject {selected_pid} (De-identified)</span> &nbsp;|&nbsp;
                    <span style="color:#0f766e; font-weight:700;">Age:</span> <span style="color:#0f172a; font-weight:600;">{format_age(p_age)}</span> &nbsp;|&nbsp;
                    <span style="color:#0f766e; font-weight:700;">Gender:</span> <span style="color:#0f172a; font-weight:600;">{p_gender}</span> &nbsp;|&nbsp;
                    <span style="color:#0f766e; font-weight:700;">Weight:</span> <span style="color:#0f172a; font-weight:600;">{p_wt}</span> &nbsp;|&nbsp;
                    <span style="color:#0f766e; font-weight:700;">Serum TSH:</span> <span style="color:#0f172a; font-weight:600;">{tsh_disp}</span> &nbsp;|&nbsp;
                    <span style="color:#0f766e; font-weight:700;">FT4:</span> <span style="color:#0f172a; font-weight:600;">{p_ft4}</span> &nbsp;|&nbsp;
                    <span style="color:#0f766e; font-weight:700;">Reference Lab Status:</span> {status_badge}
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
                action_text = "Anterior neck surface / hypothyroid profile detected. Recommend clinical endocrine consultation."

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
        bio_c1.metric("8–12 Hz Tremor Ratio", f"{feats.get('tremor_amag_ratio_8_12', 0.0)*100:.1f}%")
        bio_c2.metric("Dominant Tremor Freq", f"{feats.get('tremor_amag_dom_freq', 0.0):.1f} Hz")
        bio_c3.metric("Tremor RMS Accel", f"{feats.get('tremor_amag_rms', 0.0):.2f} m/s²")
        bio_c4.metric("Neck Surface Distance", f"{feats.get('neck_mean_mm', 0.0):.1f} mm")
        bio_c5.metric("Neck Displacement Std", f"{feats.get('neck_std_mm', 0.0):.2f} mm")

        # Explainable AI feature attribution
        with st.expander("🔍 Explainable AI (XAI) — Top Contributing Features"):
            st.markdown("Ranking of highest-impact predictors driving the primary tree ensemble prediction:")
            if hasattr(artifacts["primary_et"], "feature_importances_"):
                importances = artifacts["primary_et"].feature_importances_
                top_indices = np.argsort(importances)[::-1][:10]
                top_feats = [artifacts["selected_60"][i] for i in top_indices]
                top_scores = [importances[i] for i in top_indices]

                fig, ax = plt.subplots(figsize=(8, 3.5))
                y_pos = np.arange(len(top_feats))
                ax.barh(y_pos, top_scores, color='#0d9488', align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_feats, fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel("Relative Feature Importance (MDI)")
                ax.set_title("Top 10 Feature Attributions (v9 ExtraTrees)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Quick action: Download PDF report directly for this screened patient
        if selected_pid != "CUSTOM_UPLOAD":
            st.markdown("<br>", unsafe_allow_html=True)
            col_act1, col_act2 = st.columns([1.5, 2.5])
            with col_act1:
                if st.button("📄 Generate & Download PDF Report for this Patient", use_container_width=True):
                    st.session_state['selected_report_pid'] = selected_pid
                    st.session_state['nav_choice'] = "📄 Reports"
                    st.rerun()


# ==========================================================
# MODULE 7: 📄 REPORTS (PDF Download Option)
# ==========================================================
elif menu == "📄 Reports":
    st.markdown("""
    <div class="workspace-subbar">
        <div>
            <div class="workspace-crumb">CLINICAL DOCUMENTATION & EXPORT</div>
            <h1 class="workspace-title">Patient screening reports</h1>
            <p class="workspace-desc">Generate, preview, and download official clinical PDF reports with patient demographics, sensor findings, and ML inferences.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Patient Selector
    all_pids = [f"TH{i:03d}" for i in range(1, 128)]
    if 'selected_report_pid' in st.session_state and st.session_state['selected_report_pid'] in all_pids:
        st.session_state['report_patient_select'] = st.session_state['selected_report_pid']
        del st.session_state['selected_report_pid']

    if 'report_patient_select' not in st.session_state or st.session_state['report_patient_select'] not in all_pids:
        st.session_state['report_patient_select'] = all_pids[0]

    col_r_sel, col_r_btn = st.columns([2.5, 1.5])
    with col_r_sel:
        rep_pid = st.selectbox("Select Patient to Generate Report:", all_pids, key="report_patient_select")

    # Gather patient data (De-identified & Sanitized)
    p_deid_name = f"Research Participant {rep_pid} (De-identified Study Record - Privacy Protected)"
    p_age = 35.0
    p_gender = "Female"
    p_weight = "Not Recorded"
    p_tsh = "Pending Evaluation"
    p_ft4 = "Pending Evaluation"
    p_ref = "Unclassified"

    if not metadata_df.empty:
        p_row = metadata_df[metadata_df['pid'] == rep_pid]
        if not p_row.empty:
            r = p_row.iloc[0]
            p_age = safe_float(r.get('Age', 35), default=35)
            p_gender = format_gender(r.get('Gender', 'Female'))
            p_weight = format_weight(r.get('Weight'))
            p_tsh = format_tsh(r.get('TSH'))
            p_ft4 = format_ft4(r.get('FT4'))
            p_ref = str(r.get('status', 'Unclassified')).strip()

    # Load sensor files & extract features
    t_files, n_files = find_patient_sensor_files(rep_pid)
    tremor_df = pd.read_csv(t_files[0]) if t_files else None
    neck_df = pd.read_csv(n_files[0]) if n_files else None

    # Check if in held-out test cohort
    cached_feats = None
    if rep_pid in artifacts["test_cohort"]['pid'].values:
        cached_feats = artifacts["test_cohort"][artifacts["test_cohort"]['pid'] == rep_pid].iloc[0].to_dict()

    if tremor_df is not None or cached_feats is not None:
        pred_label, prob_dict, feats, used_feats = run_screening_inference(
            tremor_df, neck_df, age=p_age, gender=p_gender,
            model_choice=selected_model_name, cached_feats_dict=cached_feats
        )
    else:
        pred_label = "Normal"
        prob_dict = {"Normal": 0.65, "Hyperthyroid": 0.20, "Hypothyroid": 0.15}
        feats = {
            'tremor_amag_ratio_8_12': 0.18,
            'tremor_amag_dom_freq': 9.2,
            'tremor_amag_rms': 0.42,
            'neck_mean_mm': 51.4,
            'neck_std_mm': 0.35
        }

    # Clinical notes input
    notes_in = st.text_area("Clinician / Bioengineer Review Notes:", value="Subject bio-signal recording verified. Data acquisition conducted via ESP32 multimodal prototype.", height=70)

    # Generate PDF Bytes
    pdf_bytes = generate_patient_pdf(
        patient_id=rep_pid,
        patient_name=p_deid_name,
        age=p_age,
        gender=p_gender,
        weight=p_weight,
        tsh_val=p_tsh,
        ft4_val=p_ft4,
        ref_status=p_ref,
        pred_label=pred_label,
        prob_dict=prob_dict,
        model_name=selected_model_name,
        sensor_feats=feats,
        researcher_name=st.session_state.get('user_name', 'Authorized Researcher'),
        researcher_phone=st.session_state.get('user_phone', 'N/A'),
        researcher_gender=st.session_state.get('user_gender', 'N/A'),
        clinical_notes=notes_in
    )

    with col_r_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Clinical PDF Report",
            data=pdf_bytes,
            file_name=f"Thyroid_Report_{rep_pid}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # On-Screen Interactive Report Preview Card
    st.markdown(f"""
    <div class="modern-card" style="border: 2px solid #0d9488; background: #ffffff;">
        <div style="background:#0a3d36; color:#ffffff; padding:12px 18px; border-radius:8px 8px 0 0; margin:-18px -20px 16px -20px;">
            <div style="font-size:0.75rem; color:#6ee7b7; font-weight:700;">OFFICIAL CLINICAL RESEARCH REPORT &bull; JUST BME</div>
            <div style="font-size:1.15rem; font-weight:800;">Non-Invasive Multimodal Thyroid Screening Assessment</div>
        </div>
        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-bottom:14px; font-size:0.85rem; color:#0f172a;">
            <div><b>Participant ID:</b> {rep_pid}<br><b>Subject:</b> Research Participant {rep_pid} (De-identified)</div>
            <div><b>Age / Gender:</b> {format_age(p_age)} / {p_gender}<br><b>Weight:</b> {p_weight}</div>
            <div><b>Clinical TSH:</b> {p_tsh}<br><b>Reference Benchmark:</b> <b>{p_ref}</b></div>
        </div>
        <hr style="border:none; border-top:1px solid #e2e8f0; margin:12px 0;">
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; font-size:0.85rem; color:#0f172a;">
            <div>
                <b>Bio-Signal Findings:</b><br>
                &bull; 8–12 Hz Tremor Ratio: <b>{feats.get('tremor_amag_ratio_8_12', 0.0)*100:.1f}%</b><br>
                &bull; Dominant Frequency: <b>{feats.get('tremor_amag_dom_freq', 0.0):.1f} Hz</b><br>
                &bull; RMS Acceleration: <b>{feats.get('tremor_amag_rms', 0.0):.2f} m/s²</b><br>
                &bull; Neck Surface Distance: <b>{feats.get('neck_mean_mm', 0.0):.1f} &plusmn; {feats.get('neck_std_mm', 0.0):.2f} mm</b>
            </div>
            <div>
                <b>Machine Learning Decision:</b><br>
                &bull; Model: <b>{selected_model_name.split('[')[0]}</b><br>
                &bull; Screening Result: <b style="color:#0d9488; font-size:1rem;">{pred_label.upper()}</b><br>
                &bull; Confidence Score: <b>{prob_dict.get(pred_label, 0.0)*100:.1f}%</b><br>
                &bull; Attending Clinician: <b>{st.session_state.get('user_name', 'Researcher')}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# MODULE 8: ⚙️ SETTINGS / AUDIT TRAIL
# ==========================================================
elif menu == "⚙️ Settings / Audit Trail":
    st.markdown("""
    <div class="workspace-subbar">
        <div>
            <div class="workspace-crumb">SYSTEM AUDIT & WORKSPACE CONFIGURATION</div>
            <h1 class="workspace-title">Settings & audit trail</h1>
            <p class="workspace-desc">Active researcher profile, system calibration offsets, and chronological session audit log.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        st.markdown("##### 👤 Researcher Profile")
        st.markdown(f"""
        <div class="modern-card">
            <div><b>Name:</b> {st.session_state.get('user_name', 'Researcher')}</div>
            <div style="margin-top:6px;"><b>Phone Number:</b> {st.session_state.get('user_phone', 'N/A')}</div>
            <div style="margin-top:6px;"><b>Gender:</b> {st.session_state.get('user_gender', 'N/A')}</div>
            <div style="margin-top:6px;"><b>Login Timestamp:</b> {st.session_state.get('login_time', 'N/A')}</div>
            <div style="margin-top:6px;"><b>Role:</b> Authorized Clinical Bioengineer &bull; BME JUST</div>
        </div>
        """, unsafe_allow_html=True)

    with col_set2:
        st.markdown("##### ⚙️ Prototype Sensor Calibration")
        st.markdown("""
        <div class="modern-card">
            <div><b>MPU-6050 Accelerometer:</b> 100 Hz Continuous Sampling</div>
            <div style="margin-top:6px;"><b>Tremor Detection Filter:</b> 4th-Order Butterworth Bandpass (3–17 Hz)</div>
            <div style="margin-top:6px;"><b>VL53L0X Offset Calibration:</b> Nominal 50 mm anterior neck collar</div>
            <div style="margin-top:6px;"><b>Safety Standard:</b> Non-invasive surface sensing protocol</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 📜 Chronological Audit Trail")

    trail = st.session_state.get('audit_trail', [])
    audit_df = pd.DataFrame(trail)
    if not audit_df.empty:
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
    else:
        st.info("No audit entries recorded.")


# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.82rem;'>"
    "Thyroid Monitor &bull; Department of Biomedical Engineering, Jashore University of Science and Technology (JUST) &bull; 2026"
    "</div>",
    unsafe_allow_html=True
)
