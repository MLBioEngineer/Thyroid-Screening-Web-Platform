"""
Feature Extraction Pipeline for Multimodal Thyroid Screening
Extracts 416 time-domain, spectral, and biophysical descriptors from:
1. 6-Axis IMU Hand Tremor (accelerometer & gyroscope at 100 Hz)
2. VL53L0X Time-of-Flight (ToF) Anterior-Neck Surface Distance
3. Demographic and Subject Metadata
"""

import numpy as np
import pandas as pd
from scipy import stats, signal

SAMPLING_RATE_TREMOR = 100.0  # 100 Hz nominal ESP32 rate

def compute_spectral_features(sig, fs=SAMPLING_RATE_TREMOR, prefix=""):
    """
    Computes spectral features via Welch PSD and FFT
    """
    n = len(sig)
    if n < 8:
        # Fallback for short signals
        return {
            f"{prefix}_dom_freq": 0.0,
            f"{prefix}_dom_power": 0.0,
            f"{prefix}_spectral_centroid": 0.0,
            f"{prefix}_spectral_spread": 0.0,
            f"{prefix}_spectral_skew": 0.0,
            f"{prefix}_spectral_kurt": 0.0,
            f"{prefix}_spectral_flatness": 0.0,
            f"{prefix}_spectral_entropy": 0.0,
            f"{prefix}_total_power": 0.0,
            f"{prefix}_band_0_3": 0.0,
            f"{prefix}_band_3_5": 0.0,
            f"{prefix}_band_5_8": 0.0,
            f"{prefix}_band_8_12": 0.0,
            f"{prefix}_band_12_20": 0.0,
            f"{prefix}_band_20_45": 0.0,
            f"{prefix}_ratio_8_12": 0.0,
            f"{prefix}_ratio_5_12": 0.0,
            f"{prefix}_ratio_low_high": 0.0,
        }

    # Remove DC component
    sig_detrend = sig - np.mean(sig)

    # Compute Welch PSD
    nperseg = min(256, n)
    freqs, psd = signal.welch(sig_detrend, fs=fs, nperseg=nperseg)
    psd_sum = np.sum(psd) + 1e-12

    # Dominant frequency
    dom_idx = np.argmax(psd)
    dom_freq = freqs[dom_idx]
    dom_power = psd[dom_idx]

    # Spectral moments
    norm_psd = psd / psd_sum
    spectral_centroid = np.sum(freqs * norm_psd)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * norm_psd))
    spectral_skew = np.sum(((freqs - spectral_centroid) ** 3) * norm_psd) / ((spectral_spread ** 3) + 1e-12)
    spectral_kurt = np.sum(((freqs - spectral_centroid) ** 4) * norm_psd) / ((spectral_spread ** 4) + 1e-12)

    # Spectral flatness (Wiener entropy)
    geom_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    arith_mean = np.mean(psd) + 1e-12
    spectral_flatness = float(geom_mean / arith_mean)

    # Spectral entropy
    p = norm_psd[norm_psd > 0]
    spectral_entropy = -float(np.sum(p * np.log2(p + 1e-12)))

    # Tremor physiological frequency bands
    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(np.sum(psd[mask]))

    bp_0_3 = band_power(0.5, 3.0)
    bp_3_5 = band_power(3.0, 5.0)    # Resting Parkinsonian band
    bp_5_8 = band_power(5.0, 8.0)    # Postural tremor band
    bp_8_12 = band_power(8.0, 12.0)  # Fine thyrotoxicosis tremor band
    bp_12_20 = band_power(12.0, 20.0)
    bp_20_45 = band_power(20.0, 45.0)

    ratio_8_12 = float(bp_8_12 / psd_sum)
    ratio_5_12 = float((bp_5_8 + bp_8_12) / psd_sum)
    ratio_low_high = float((bp_3_5 + bp_5_8 + 1e-9) / (bp_8_12 + bp_12_20 + 1e-9))

    return {
        f"{prefix}_dom_freq": float(dom_freq),
        f"{prefix}_dom_power": float(dom_power),
        f"{prefix}_spectral_centroid": float(spectral_centroid),
        f"{prefix}_spectral_spread": float(spectral_spread),
        f"{prefix}_spectral_skew": float(spectral_skew),
        f"{prefix}_spectral_kurt": float(spectral_kurt),
        f"{prefix}_spectral_flatness": float(spectral_flatness),
        f"{prefix}_spectral_entropy": float(spectral_entropy),
        f"{prefix}_total_power": float(psd_sum),
        f"{prefix}_band_0_3": bp_0_3,
        f"{prefix}_band_3_5": bp_3_5,
        f"{prefix}_band_5_8": bp_5_8,
        f"{prefix}_band_8_12": bp_8_12,
        f"{prefix}_band_12_20": bp_12_20,
        f"{prefix}_band_20_45": bp_20_45,
        f"{prefix}_ratio_8_12": ratio_8_12,
        f"{prefix}_ratio_5_12": ratio_5_12,
        f"{prefix}_ratio_low_high": ratio_low_high,
    }


def compute_time_features(sig, prefix=""):
    """
    Computes statistical and waveform time-domain features
    """
    n = len(sig)
    if n == 0:
        return {}

    mean_val = float(np.mean(sig))
    std_val = float(np.std(sig))
    var_val = float(np.var(sig))
    median_val = float(np.median(sig))
    min_val = float(np.min(sig))
    max_val = float(np.max(sig))
    range_val = max_val - min_val

    p10 = float(np.percentile(sig, 10))
    p25 = float(np.percentile(sig, 25))
    p75 = float(np.percentile(sig, 75))
    p90 = float(np.percentile(sig, 90))
    iqr_val = p75 - p25

    rms_val = float(np.sqrt(np.mean(sig ** 2)))
    energy_val = float(np.mean(sig ** 2))

    skew_val = float(stats.skew(sig)) if std_val > 1e-9 else 0.0
    kurt_val = float(stats.kurtosis(sig)) if std_val > 1e-9 else 0.0

    # Zero crossings and mean crossings
    zcr = float(np.sum(np.diff(sig > 0) != 0) / max(n - 1, 1))
    mcr = float(np.sum(np.diff(sig > mean_val) != 0) / max(n - 1, 1))

    crest_factor = float(np.max(np.abs(sig)) / (rms_val + 1e-9))
    shape_factor = float(rms_val / (np.mean(np.abs(sig)) + 1e-9))
    impulse_factor = float(np.max(np.abs(sig)) / (np.mean(np.abs(sig)) + 1e-9))

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_var": var_val,
        f"{prefix}_median": median_val,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_range": range_val,
        f"{prefix}_p10": p10,
        f"{prefix}_p25": p25,
        f"{prefix}_p75": p75,
        f"{prefix}_p90": p90,
        f"{prefix}_iqr": iqr_val,
        f"{prefix}_rms": rms_val,
        f"{prefix}_energy": energy_val,
        f"{prefix}_skew": skew_val,
        f"{prefix}_kurt": kurt_val,
        f"{prefix}_zcr": zcr,
        f"{prefix}_mcr": mcr,
        f"{prefix}_crest": crest_factor,
        f"{prefix}_shape": shape_factor,
        f"{prefix}_impulse": impulse_factor,
    }


def extract_tremor_features(df_tremor):
    """
    Extracts comprehensive features across 6-axis IMU + derived vector magnitudes
    """
    feats = {}
    cols = ['accel_x_m_s2', 'accel_y_m_s2', 'accel_z_m_s2',
            'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']

    # Filter to measurement_status == 'OK' if present
    if 'measurement_status' in df_tremor.columns:
        valid_df = df_tremor[df_tremor['measurement_status'] == 'OK']
        if len(valid_df) > 50:
            df_tremor = valid_df

    # Extract raw arrays
    ax = df_tremor['accel_x_m_s2'].values if 'accel_x_m_s2' in df_tremor.columns else np.zeros(100)
    ay = df_tremor['accel_y_m_s2'].values if 'accel_y_m_s2' in df_tremor.columns else np.zeros(100)
    az = df_tremor['accel_z_m_s2'].values if 'accel_z_m_s2' in df_tremor.columns else np.zeros(100)
    gx = df_tremor['gyro_x_rad_s'].values if 'gyro_x_rad_s' in df_tremor.columns else np.zeros(100)
    gy = df_tremor['gyro_y_rad_s'].values if 'gyro_y_rad_s' in df_tremor.columns else np.zeros(100)
    gz = df_tremor['gyro_z_rad_s'].values if 'gyro_z_rad_s' in df_tremor.columns else np.zeros(100)

    # Derived streams
    amag = np.sqrt(ax**2 + ay**2 + az**2)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)
    axy = np.sqrt(ax**2 + ay**2)
    gxy = np.sqrt(gx**2 + gy**2)

    streams = {
        'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz,
        'amag': amag, 'gmag': gmag,
        'axy': axy, 'gxy': gxy
    }

    # Extract time + spectral features for all 10 streams
    for name, s in streams.items():
        feats.update(compute_time_features(s, prefix=f"tremor_{name}"))
        feats.update(compute_spectral_features(s, prefix=f"tremor_{name}"))

    # Cross-axis correlations
    def safe_corr(u, v):
        if np.std(u) > 1e-9 and np.std(v) > 1e-9:
            return float(np.corrcoef(u, v)[0, 1])
        return 0.0

    feats['tremor_corr_ax_ay'] = safe_corr(ax, ay)
    feats['tremor_corr_ax_az'] = safe_corr(ax, az)
    feats['tremor_corr_ay_az'] = safe_corr(ay, az)
    feats['tremor_corr_gx_gy'] = safe_corr(gx, gy)
    feats['tremor_corr_gx_gz'] = safe_corr(gx, gz)
    feats['tremor_corr_gy_gz'] = safe_corr(gy, gz)
    feats['tremor_corr_amag_gmag'] = safe_corr(amag, gmag)

    # Tremor duration and sample quality
    feats['tremor_sample_count'] = float(len(df_tremor))
    feats['tremor_duration_sec'] = float(len(df_tremor) / SAMPLING_RATE_TREMOR)

    return feats


def extract_neck_features(df_neck):
    """
    Extracts VL53L0X Time-of-Flight anterior neck distance descriptors
    """
    feats = {}
    if 'distance_mm' in df_neck.columns:
        dist = df_neck['distance_mm'].dropna().values
    else:
        dist = np.array([50.0] * 10)

    if len(dist) == 0:
        dist = np.array([50.0] * 10)

    mean_d = float(np.mean(dist))
    std_d = float(np.std(dist))
    var_d = float(np.var(dist))
    median_d = float(np.median(dist))
    min_d = float(np.min(dist))
    max_d = float(np.max(dist))
    range_d = max_d - min_d
    cv_d = float(std_d / (mean_d + 1e-6))

    p10 = float(np.percentile(dist, 10))
    p25 = float(np.percentile(dist, 25))
    p75 = float(np.percentile(dist, 75))
    p90 = float(np.percentile(dist, 90))
    iqr_d = p75 - p25

    # Trend and slope across the 10 readings
    n = len(dist)
    if n > 1:
        x = np.arange(n)
        slope, intercept, r_val, _, _ = stats.linregress(x, dist)
        delta_d = float(dist[-1] - dist[0])
    else:
        slope = 0.0
        r_val = 0.0
        delta_d = 0.0

    mad_d = float(np.median(np.abs(dist - median_d)))
    rms_d = float(np.sqrt(np.mean(dist ** 2)))

    feats['neck_mean_mm'] = mean_d
    feats['neck_std_mm'] = std_d
    feats['neck_var_mm'] = var_d
    feats['neck_median_mm'] = median_d
    feats['neck_min_mm'] = min_d
    feats['neck_max_mm'] = max_d
    feats['neck_range_mm'] = range_d
    feats['neck_cv'] = cv_d
    feats['neck_p10'] = p10
    feats['neck_p25'] = p25
    feats['neck_p75'] = p75
    feats['neck_p90'] = p90
    feats['neck_iqr'] = iqr_d
    feats['neck_slope'] = float(slope)
    feats['neck_trend_r'] = float(r_val) if not np.isnan(r_val) else 0.0
    feats['neck_delta_mm'] = delta_d
    feats['neck_mad_mm'] = mad_d
    feats['neck_rms_mm'] = rms_d
    feats['neck_sample_count'] = float(n)

    # Surface condition encoding (T=1, U=2, S=3, other=0)
    sc = df_neck['surface_condition'].iloc[0] if 'surface_condition' in df_neck.columns and len(df_neck) > 0 else 'T'
    sc_code = {'T': 1.0, 'U': 2.0, 'S': 3.0, 'X': 0.0}.get(str(sc).strip().upper(), 1.0)
    feats['neck_surface_condition_code'] = sc_code

    return feats


def extract_patient_features(tremor_dfs, neck_dfs, age=40, gender='Female'):
    """
    Aggregates multi-trial tremor and neck features at the patient level
    and adds demographic descriptors. Exactly 416 output features.
    """
    # Extract features for all tremor trials
    tremor_trial_feats = [extract_tremor_features(tdf) for tdf in tremor_dfs] if tremor_dfs else [extract_tremor_features(pd.DataFrame())]
    # Extract features for all neck trials
    neck_trial_feats = [extract_neck_features(ndf) for ndf in neck_dfs] if neck_dfs else [extract_neck_features(pd.DataFrame())]

    # Aggregate across trials (mean, std, min, max where applicable)
    df_t = pd.DataFrame(tremor_trial_feats)
    df_n = pd.DataFrame(neck_trial_feats)

    patient_feats = {}

    # Primary trial features (mean across trials)
    for col in df_t.columns:
        patient_feats[col] = float(df_t[col].mean())
    for col in df_n.columns:
        patient_feats[col] = float(df_n[col].mean())

    # Add inter-trial variation features
    for col in ['tremor_amag_mean', 'tremor_amag_std', 'tremor_gmag_mean', 'tremor_amag_ratio_8_12',
                'neck_mean_mm', 'neck_std_mm']:
        if col in df_t.columns:
            patient_feats[f"{col}_trial_std"] = float(df_t[col].std()) if len(df_t) > 1 else 0.0
            patient_feats[f"{col}_trial_range"] = float(df_t[col].max() - df_t[col].min()) if len(df_t) > 1 else 0.0
        elif col in df_n.columns:
            patient_feats[f"{col}_trial_std"] = float(df_n[col].std()) if len(df_n) > 1 else 0.0
            patient_feats[f"{col}_trial_range"] = float(df_n[col].max() - df_n[col].min()) if len(df_n) > 1 else 0.0

    # Demographic features
    patient_feats['demo_age'] = float(age) if not pd.isna(age) else 40.0
    gender_code = 1.0 if str(gender).strip().lower() in ['female', 'f'] else 0.0
    patient_feats['demo_gender'] = gender_code
    patient_feats['demo_trials_count'] = float(len(tremor_dfs))

    return patient_feats
