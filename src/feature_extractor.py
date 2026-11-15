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

def safe_float(v, default=0.0):
    try:
        val = float(v)
        if np.isnan(val) or np.isinf(val):
            return float(default)
        return val
    except Exception:
        return float(default)

def get_series_by_aliases(df, aliases, default_val=0.0, default_len=100):
    if df is None or len(df) == 0:
        return np.full(default_len, default_val, dtype=float)
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        a_low = alias.strip().lower()
        if a_low in col_map:
            s = pd.to_numeric(df[col_map[a_low]], errors='coerce').values
            s = s[np.isfinite(s)]
            if len(s) > 0:
                return s
    return np.full(default_len, default_val, dtype=float)

def compute_spectral_features(sig, fs=SAMPLING_RATE_TREMOR, prefix=""):
    """
    Computes spectral features via Welch PSD and FFT
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig[np.isfinite(sig)]
    n = len(sig)
    
    if n < 8:
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
    psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
    psd_sum = np.sum(psd) + 1e-12

    # Dominant frequency
    dom_idx = np.argmax(psd)
    dom_freq = freqs[dom_idx]
    dom_power = psd[dom_idx]

    # Spectral moments
    norm_psd = psd / psd_sum
    spectral_centroid = np.sum(freqs * norm_psd)
    spectral_spread = np.sqrt(max(0.0, np.sum(((freqs - spectral_centroid) ** 2) * norm_psd)))
    spectral_skew = np.sum(((freqs - spectral_centroid) ** 3) * norm_psd) / ((spectral_spread ** 3) + 1e-12)
    spectral_kurt = np.sum(((freqs - spectral_centroid) ** 4) * norm_psd) / ((spectral_spread ** 4) + 1e-12)

    # Spectral flatness (Wiener entropy)
    geom_mean = np.exp(np.mean(np.log(np.maximum(psd, 1e-12))))
    arith_mean = np.mean(psd) + 1e-12
    spectral_flatness = float(geom_mean / arith_mean)

    # Spectral entropy
    p = norm_psd[norm_psd > 0]
    spectral_entropy = -float(np.sum(p * np.log2(np.maximum(p, 1e-12))))

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
        f"{prefix}_dom_freq": safe_float(dom_freq),
        f"{prefix}_dom_power": safe_float(dom_power),
        f"{prefix}_spectral_centroid": safe_float(spectral_centroid),
        f"{prefix}_spectral_spread": safe_float(spectral_spread),
        f"{prefix}_spectral_skew": safe_float(spectral_skew),
        f"{prefix}_spectral_kurt": safe_float(spectral_kurt),
        f"{prefix}_spectral_flatness": safe_float(spectral_flatness),
        f"{prefix}_spectral_entropy": safe_float(spectral_entropy),
        f"{prefix}_total_power": safe_float(psd_sum),
        f"{prefix}_band_0_3": safe_float(bp_0_3),
        f"{prefix}_band_3_5": safe_float(bp_3_5),
        f"{prefix}_band_5_8": safe_float(bp_5_8),
        f"{prefix}_band_8_12": safe_float(bp_8_12),
        f"{prefix}_band_12_20": safe_float(bp_12_20),
        f"{prefix}_band_20_45": safe_float(bp_20_45),
        f"{prefix}_ratio_8_12": safe_float(ratio_8_12),
        f"{prefix}_ratio_5_12": safe_float(ratio_5_12),
        f"{prefix}_ratio_low_high": safe_float(ratio_low_high),
    }

def compute_time_features(sig, prefix=""):
    """
    Computes statistical and waveform time-domain features
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig[np.isfinite(sig)]
    n = len(sig)
    if n == 0:
        sig = np.zeros(10)
        n = 10

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
        f"{prefix}_mean": safe_float(mean_val),
        f"{prefix}_std": safe_float(std_val),
        f"{prefix}_var": safe_float(var_val),
        f"{prefix}_median": safe_float(median_val),
        f"{prefix}_min": safe_float(min_val),
        f"{prefix}_max": safe_float(max_val),
        f"{prefix}_range": safe_float(range_val),
        f"{prefix}_p10": safe_float(p10),
        f"{prefix}_p25": safe_float(p25),
        f"{prefix}_p75": safe_float(p75),
        f"{prefix}_p90": safe_float(p90),
        f"{prefix}_iqr": safe_float(iqr_val),
        f"{prefix}_rms": safe_float(rms_val),
        f"{prefix}_energy": safe_float(energy_val),
        f"{prefix}_skew": safe_float(skew_val),
        f"{prefix}_kurt": safe_float(kurt_val),
        f"{prefix}_zcr": safe_float(zcr),
        f"{prefix}_mcr": safe_float(mcr),
        f"{prefix}_crest": safe_float(crest_factor),
        f"{prefix}_shape": safe_float(shape_factor),
        f"{prefix}_impulse": safe_float(impulse_factor),
    }

def extract_tremor_features(df_tremor):
    """
    Extracts comprehensive features across 6-axis IMU + derived vector magnitudes
    """
    feats = {}
    if df_tremor is None:
        df_tremor = pd.DataFrame()

    # Filter to measurement_status == 'OK' if present
    if 'measurement_status' in df_tremor.columns:
        valid_df = df_tremor[df_tremor['measurement_status'] == 'OK']
        if len(valid_df) > 50:
            df_tremor = valid_df

    # Extract raw arrays with alias tolerance
    ax = get_series_by_aliases(df_tremor, ['accel_x_m_s2', 'accel_x', 'acc_x', 'ax', 'accelerometer_x'])
    ay = get_series_by_aliases(df_tremor, ['accel_y_m_s2', 'accel_y', 'acc_y', 'ay', 'accelerometer_y'])
    az = get_series_by_aliases(df_tremor, ['accel_z_m_s2', 'accel_z', 'acc_z', 'az', 'accelerometer_z'], default_val=9.8)
    gx = get_series_by_aliases(df_tremor, ['gyro_x_rad_s', 'gyro_x', 'gyr_x', 'gx', 'gyroscope_x'])
    gy = get_series_by_aliases(df_tremor, ['gyro_y_rad_s', 'gyro_y', 'gyr_y', 'gy', 'gyroscope_y'])
    gz = get_series_by_aliases(df_tremor, ['gyro_z_rad_s', 'gyro_z', 'gyr_z', 'gz', 'gyroscope_z'])

    # Align lengths
    min_len = min(len(ax), len(ay), len(az), len(gx), len(gy), len(gz))
    if min_len < 10:
        min_len = 100
        ax = np.zeros(min_len)
        ay = np.zeros(min_len)
        az = np.full(min_len, 9.8)
        gx = np.zeros(min_len)
        gy = np.zeros(min_len)
        gz = np.zeros(min_len)
    else:
        ax, ay, az = ax[:min_len], ay[:min_len], az[:min_len]
        gx, gy, gz = gx[:min_len], gy[:min_len], gz[:min_len]

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
        if len(u) > 2 and np.std(u) > 1e-9 and np.std(v) > 1e-9:
            r = np.corrcoef(u, v)[0, 1]
            return safe_float(r, default=0.0)
        return 0.0

    feats['tremor_corr_ax_ay'] = safe_corr(ax, ay)
    feats['tremor_corr_ax_az'] = safe_corr(ax, az)
    feats['tremor_corr_ay_az'] = safe_corr(ay, az)
    feats['tremor_corr_gx_gy'] = safe_corr(gx, gy)
    feats['tremor_corr_gx_gz'] = safe_corr(gx, gz)
    feats['tremor_corr_gy_gz'] = safe_corr(gy, gz)
    feats['tremor_corr_amag_gmag'] = safe_corr(amag, gmag)

    # Tremor duration and sample quality
    feats['tremor_sample_count'] = float(min_len)
    feats['tremor_duration_sec'] = float(min_len / SAMPLING_RATE_TREMOR)

    return feats

def extract_neck_features(df_neck):
    """
    Extracts VL53L0X Time-of-Flight anterior neck distance descriptors
    """
    feats = {}
    if df_neck is None:
        df_neck = pd.DataFrame()

    dist = get_series_by_aliases(
        df_neck,
        ['distance_mm', 'distance', 'dist_mm', 'dist', 'neck_distance', 'neck_distance_mm', 'distance (mm)'],
        default_val=50.0,
        default_len=10
    )

    valid_mask = (dist > 0) & np.isfinite(dist)
    valid_dist = dist[valid_mask]
    if len(valid_dist) == 0:
        valid_dist = np.array([50.0] * 10)

    mean_d = float(np.mean(valid_dist))
    std_d = float(np.std(valid_dist))
    var_d = float(np.var(valid_dist))
    median_d = float(np.median(valid_dist))
    min_d = float(np.min(valid_dist))
    max_d = float(np.max(valid_dist))
    range_d = max_d - min_d
    cv_d = float(std_d / (mean_d + 1e-6))

    p10 = float(np.percentile(valid_dist, 10))
    p25 = float(np.percentile(valid_dist, 25))
    p75 = float(np.percentile(valid_dist, 75))
    p90 = float(np.percentile(valid_dist, 90))
    iqr_d = p75 - p25

    # Trend and slope across the consecutive readings
    n = len(valid_dist)
    if n > 1 and std_d > 1e-6:
        x = np.arange(n)
        slope, intercept, r_val, _, _ = stats.linregress(x, valid_dist)
        delta_d = float(valid_dist[-1] - valid_dist[0])
    else:
        slope = 0.0
        r_val = 0.0
        delta_d = 0.0

    mad_d = float(np.median(np.abs(valid_dist - median_d)))
    rms_d = float(np.sqrt(np.mean(valid_dist ** 2)))

    feats['neck_mean_mm'] = safe_float(mean_d)
    feats['neck_std_mm'] = safe_float(std_d)
    feats['neck_var_mm'] = safe_float(var_d)
    feats['neck_median_mm'] = safe_float(median_d)
    feats['neck_min_mm'] = safe_float(min_d)
    feats['neck_max_mm'] = safe_float(max_d)
    feats['neck_range_mm'] = safe_float(range_d)
    feats['neck_cv'] = safe_float(cv_d)
    feats['neck_p10'] = safe_float(p10)
    feats['neck_p25'] = safe_float(p25)
    feats['neck_p75'] = safe_float(p75)
    feats['neck_p90'] = safe_float(p90)
    feats['neck_iqr'] = safe_float(iqr_d)
    feats['neck_slope'] = safe_float(slope)
    feats['neck_trend_r'] = safe_float(r_val)
    feats['neck_delta_mm'] = safe_float(delta_d)
    feats['neck_mad_mm'] = safe_float(mad_d)
    feats['neck_rms_mm'] = safe_float(rms_d)
    feats['neck_sample_count'] = float(n)

    # Surface condition encoding (T=1, U=2, S=3, other=0)
    sc = 'T'
    if 'surface_condition' in df_neck.columns and len(df_neck) > 0:
        sc = df_neck['surface_condition'].iloc[0]
    sc_code = {'T': 1.0, 'U': 2.0, 'S': 3.0, 'X': 0.0}.get(str(sc).strip().upper(), 1.0)
    feats['neck_surface_condition_code'] = float(sc_code)

    return feats

def extract_patient_features(tremor_dfs, neck_dfs, age=40, gender='Female'):
    """
    Aggregates multi-trial tremor and neck features at the patient level
    and adds demographic descriptors.
    """
    # Extract features for all tremor trials
    clean_tdfs = [tdf for tdf in tremor_dfs if tdf is not None and len(tdf) > 0]
    clean_ndfs = [ndf for ndf in neck_dfs if ndf is not None and len(ndf) > 0]

    tremor_trial_feats = [extract_tremor_features(tdf) for tdf in clean_tdfs] if clean_tdfs else [extract_tremor_features(pd.DataFrame())]
    neck_trial_feats = [extract_neck_features(ndf) for ndf in clean_ndfs] if clean_ndfs else [extract_neck_features(pd.DataFrame())]

    # Aggregate across trials
    df_t = pd.DataFrame(tremor_trial_feats)
    df_n = pd.DataFrame(neck_trial_feats)

    patient_feats = {}

    # Primary trial features (mean across trials)
    for col in df_t.columns:
        patient_feats[col] = safe_float(df_t[col].mean())
    for col in df_n.columns:
        patient_feats[col] = safe_float(df_n[col].mean())

    # Add inter-trial variation features
    for col in ['tremor_amag_mean', 'tremor_amag_std', 'tremor_gmag_mean', 'tremor_amag_ratio_8_12',
                'neck_mean_mm', 'neck_std_mm']:
        if col in df_t.columns:
            patient_feats[f"{col}_trial_std"] = safe_float(df_t[col].std()) if len(df_t) > 1 else 0.0
            patient_feats[f"{col}_trial_range"] = safe_float(df_t[col].max() - df_t[col].min()) if len(df_t) > 1 else 0.0
        elif col in df_n.columns:
            patient_feats[f"{col}_trial_std"] = safe_float(df_n[col].std()) if len(df_n) > 1 else 0.0
            patient_feats[f"{col}_trial_range"] = safe_float(df_n[col].max() - df_n[col].min()) if len(df_n) > 1 else 0.0

    # Demographic features
    patient_feats['demo_age'] = safe_float(age, default=40.0)
    gender_code = 1.0 if str(gender).strip().lower() in ['female', 'f'] else 0.0
    patient_feats['demo_gender'] = float(gender_code)
    patient_feats['demo_trials_count'] = float(len(clean_tdfs))

    # Final guarantee: ensure all values are clean finite floats
    sanitized = {}
    for k, v in patient_feats.items():
        sanitized[k] = safe_float(v)

    return sanitized
