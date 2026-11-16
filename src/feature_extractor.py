"""
Feature Extraction Pipeline for Multimodal Thyroid Screening
Extracts 416 time-domain, spectral, and biophysical descriptors from:
1. 6-Axis IMU Hand Tremor (accelerometer & gyroscope at 100 Hz)
2. VL53L0X Time-of-Flight (ToF) Anterior-Neck Surface Distance
3. Demographic and Subject Metadata
"""

import numpy as np
import pandas as pd
from scipy import stats, signal, fftpack
import scipy.io.wavfile as wavfile
import io

SAMPLING_RATE_TREMOR = 100.0  # 100 Hz nominal ESP32 rate
SAMPLING_RATE_VOICE = 16000.0  # 16 kHz nominal ESP32 INMP441 rate

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

    # Filter to measurement_status == 'OK' if present (case and whitespace insensitive)
    if 'measurement_status' in df_tremor.columns:
        valid_mask = df_tremor['measurement_status'].astype(str).str.strip().str.upper() == 'OK'
        valid_df = df_tremor[valid_mask]
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

def compute_mfcc_features(sig, fs=SAMPLING_RATE_VOICE, num_ceps=13, n_filt=26, nfft=512):
    """
    Computes Mel-Frequency Cepstral Coefficients (MFCCs) using standard filterbanks and DCT-II.
    Returns 1D array of length num_ceps (mean MFCC across frames).
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig[np.isfinite(sig)]
    if len(sig) < 160:
        return np.zeros(num_ceps, dtype=float)

    # Pre-emphasis
    emphasized = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])

    # Framing
    frame_len = int(0.025 * fs)  # 400 samples (25 ms)
    hop_len = int(0.010 * fs)    # 160 samples (10 ms)
    num_frames = max(1, 1 + int((len(emphasized) - frame_len) / hop_len))
    pad_len = (num_frames - 1) * hop_len + frame_len
    if len(emphasized) < pad_len:
        emphasized = np.pad(emphasized, (0, pad_len - len(emphasized)))

    indices = (np.tile(np.arange(0, frame_len), (num_frames, 1)) +
               np.tile(np.arange(0, num_frames * hop_len, hop_len), (frame_len, 1)).T)
    frames = emphasized[indices.astype(np.int32, copy=False)] * np.hamming(frame_len)

    # Magnitude and power spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = (1.0 / nfft) * (mag_frames ** 2)

    # Mel scale filterbank
    low_mel = 0.0
    high_mel = 2595.0 * np.log10(1.0 + (fs / 2.0) / 700.0)
    mel_pts = np.linspace(low_mel, high_mel, n_filt + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bin_pts = np.floor((nfft + 1) * hz_pts / fs).astype(int)

    fbank = np.zeros((n_filt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, n_filt + 1):
        for k in range(bin_pts[m - 1], bin_pts[m]):
            fbank[m - 1, k] = (k - bin_pts[m - 1]) / max(bin_pts[m] - bin_pts[m - 1], 1)
        for k in range(bin_pts[m], bin_pts[m + 1]):
            fbank[m - 1, k] = (bin_pts[m + 1] - k) / max(bin_pts[m + 1] - bin_pts[m], 1)

    fb_energy = np.dot(pow_frames, fbank.T)
    fb_energy = np.where(fb_energy <= 0, 1e-12, fb_energy)
    fb_energy = 20.0 * np.log10(fb_energy)
    mfcc_mat = fftpack.dct(fb_energy, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return np.mean(mfcc_mat, axis=0)

def extract_voice_features(source, fs=SAMPLING_RATE_VOICE):
    """
    Extracts clinical acoustic phonation descriptors from an audio file, bytes,
    array, or dataframe (sampled nominally at 16,000 Hz recorded via ESP32 INMP441).
    Descriptors extracted:
    - Temporal: RMS energy, Zero Crossing Rate (ZCR), amplitude envelope.
    - Pitch & Perturbation: F0 (via normalized autocorrelation/peak picking),
      Jitter (pitch perturbation), Shimmer (amplitude perturbation), and HNR.
    - Spectral & Cepstral: Spectral Centroid, Spectral Spread, Spectral Rolloff (85%),
      Spectral Flatness, and 13 MFCC coefficients.
    Robustness:
    - Noise gating removes silence and microphone hiss.
    - Zero NaNs or Infs (all features strictly finite floats).
    - Missing/corrupt audio gracefully returns default baseline dictionary.
    """
    # Canonical baseline dictionary with zero/default baseline vectors
    default_feats = {
        'voice_rms': 0.0,
        'voice_zcr': 0.0,
        'voice_amp_env_mean': 0.0,
        'voice_amp_env_std': 0.0,
        'voice_amp_env_max': 0.0,
        'voice_f0_mean': 0.0,
        'voice_f0_std': 0.0,
        'voice_f0_min': 0.0,
        'voice_f0_max': 0.0,
        'voice_jitter_pct': 0.0,
        'voice_shimmer_pct': 0.0,
        'voice_hnr_db': 0.0,
        'voice_stability_index': 0.0,
        'voice_spectral_centroid': 0.0,
        'voice_spectral_spread': 0.0,
        'voice_spectral_rolloff': 0.0,
        'voice_spectral_flatness': 0.0,
        'voice_duration_sec': 0.0,
        'voice_sample_count': 0.0,
    }
    for i in range(1, 14):
        default_feats[f'voice_mfcc_{i}'] = 0.0
    default_feats['voice_mfcc_mean'] = 0.0
    default_feats['voice_mfcc_std'] = 0.0

    if source is None:
        return default_feats

    # Rewind stream if seekable
    if hasattr(source, 'seek'):
        try:
            source.seek(0)
        except Exception:
            pass

    sig = None
    try:
        if isinstance(source, (str, bytes, io.BytesIO)) or hasattr(source, 'read'):
            if isinstance(source, str):
                if source.endswith('.csv'):
                    df_v = pd.read_csv(source)
                    col = next((c for c in ['amplitude_pcm16', 'amplitude', 'val'] if c in df_v.columns), df_v.columns[-1])
                    raw = pd.to_numeric(df_v[col], errors='coerce').fillna(0).values
                    sig = raw.astype(np.float32) / 32768.0
                else:
                    read_fs, data = wavfile.read(source)
                    fs = float(read_fs)
                    if data.dtype == np.int16:
                        sig = data.astype(np.float32) / 32768.0
                    elif data.dtype == np.int32:
                        sig = data.astype(np.float32) / 2147483648.0
                    elif data.dtype == np.uint8:
                        sig = (data.astype(np.float32) - 128.0) / 128.0
                    else:
                        sig = data.astype(np.float32)
            else:
                raw_bytes = source.read() if hasattr(source, 'read') else (source.getvalue() if hasattr(source, 'getvalue') else source)
                # Rewind seekable source after reading so downstream operations can re-read
                if hasattr(source, 'seek'):
                    try:
                        source.seek(0)
                    except Exception:
                        pass
                buf = io.BytesIO(raw_bytes)
                read_fs, data = wavfile.read(buf)
                fs = float(read_fs)
                if data.dtype == np.int16:
                    sig = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    sig = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    sig = (data.astype(np.float32) - 128.0) / 128.0
                else:
                    sig = data.astype(np.float32)
        elif isinstance(source, np.ndarray):
            sig = source.astype(np.float32)
        elif hasattr(source, 'columns'):
            col = next((c for c in ['amplitude_pcm16', 'amplitude', 'val'] if c in source.columns), source.columns[-1])
            raw = pd.to_numeric(source[col], errors='coerce').fillna(0).values
            sig = raw.astype(np.float32) / 32768.0
    except Exception:
        if hasattr(source, 'seek'):
            try:
                source.seek(0)
            except Exception:
                pass
        return default_feats

    if sig is None:
        return default_feats

    # Mono conversion (handles both channels-last and channels-first audio)
    if sig.ndim > 1:
        if sig.shape[0] <= 8 and sig.shape[1] > sig.shape[0]:
            sig = np.mean(sig, axis=0)
        else:
            sig = np.mean(sig, axis=-1)

    sig = sig[np.isfinite(sig)]
    n = len(sig)
    if n < 160:
        return default_feats

    # Standardize sample rate to SAMPLING_RATE_VOICE (16 kHz nominal INMP441 rate)
    duration_sec = float(n / fs) if fs > 0 else 0.0
    if fs > 0 and abs(fs - SAMPLING_RATE_VOICE) > 1.0:
        target_len = int(round(n * SAMPLING_RATE_VOICE / fs))
        if target_len >= 160:
            sig = signal.resample(sig, target_len)
            fs = SAMPLING_RATE_VOICE
            n = len(sig)

    # 1. Temporal Features (RMS, ZCR, Envelope)
    rms_val = float(np.sqrt(np.mean(sig ** 2)))
    zcr_val = float(np.sum(np.diff(sig > 0) != 0) / max(n - 1, 1))

    # Amplitude Envelope via Hilbert transform
    try:
        analytic = signal.hilbert(sig)
        amp_env = np.abs(analytic)
        amp_mean = float(np.mean(amp_env))
        amp_std = float(np.std(amp_env))
        amp_max = float(np.max(amp_env))
    except Exception:
        amp_env = np.abs(sig)
        amp_mean = float(np.mean(amp_env))
        amp_std = float(np.std(amp_env))
        amp_max = float(np.max(amp_env))

    # Noise Gating (Threshold to isolate active phonation frames from silence/hiss)
    frame_len = int(0.025 * fs)  # 400 samples (25 ms)
    hop_len = int(0.010 * fs)    # 160 samples (10 ms)
    num_frames = (n - frame_len) // hop_len
    if num_frames > 0:
        frame_energies = [np.sqrt(np.mean(sig[i * hop_len: i * hop_len + frame_len] ** 2)) for i in range(num_frames)]
        max_e = max(frame_energies) if frame_energies else 0.0
        gate_thresh = max(0.002, 0.08 * max_e) if max_e > 0.002 else 0.0
        active_indices = [i for i, e in enumerate(frame_energies) if e >= gate_thresh]
    else:
        active_indices = []

    # 2. Pitch ($F_0$), Jitter, Shimmer, HNR via normalized autocorrelation / peak picking
    # Vocal range: 70 Hz (lag 228 at 16kHz) to 500 Hz (lag 32 at 16kHz)
    min_lag = max(10, int(fs / 500.0))
    max_lag = min(len(sig) - 1, int(fs / 70.0))
    f0_list = []
    peak_r_list = []
    peak_amp_list = []

    for idx in active_indices:
        f_start = idx * hop_len
        f_slice = sig[f_start: f_start + frame_len]
        f_detrend = f_slice - np.mean(f_slice)
        denom = np.sum(f_detrend ** 2)
        if denom < 1e-8:
            continue
        corr = np.correlate(f_detrend, f_detrend, mode='full')[len(f_detrend) - 1:]
        corr_norm = corr / (denom + 1e-12)
        if len(corr_norm) > max_lag:
            search_region = corr_norm[min_lag:max_lag]
            best_idx = int(np.argmax(search_region))
            best_lag = float(min_lag + best_idx)
            best_r = float(search_region[best_idx])
            if best_r > 0.25:
                # 3-point parabolic peak interpolation for sub-sample precision
                if 0 < best_idx < len(search_region) - 1:
                    alpha = float(search_region[best_idx - 1])
                    beta = float(search_region[best_idx])
                    gamma = float(search_region[best_idx + 1])
                    denom_p = alpha - 2.0 * beta + gamma
                    if abs(denom_p) > 1e-9:
                        delta = 0.5 * (alpha - gamma) / denom_p
                        if abs(delta) < 1.0:
                            best_lag += delta
                if best_lag > 0:
                    f0_list.append(float(fs / best_lag))
                    overlap_ratio = max(0.1, (frame_len - best_lag) / float(frame_len))
                    norm_r = float(np.clip(best_r / overlap_ratio, 0.0, 1.0))
                    peak_r_list.append(norm_r)
                    peak_amp_list.append(float(np.max(np.abs(f_slice))))

    if len(f0_list) > 0:
        f0_arr = np.array(f0_list)
        f0_mean = float(np.mean(f0_arr))
        f0_std = float(np.std(f0_arr))
        f0_min = float(np.min(f0_arr))
        f0_max = float(np.max(f0_arr))

        if len(f0_arr) > 1:
            periods = 1.0 / f0_arr
            jitter_pct = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9) * 100.0)
            amps = np.array(peak_amp_list)
            shimmer_pct = float(np.mean(np.abs(np.diff(amps))) / (np.mean(amps) + 1e-9) * 100.0)
        else:
            jitter_pct, shimmer_pct = 0.0, 0.0

        mean_r = np.clip(np.mean(peak_r_list), 0.01, 0.9999)
        hnr_db = float(10.0 * np.log10(mean_r / (1.0 - mean_r + 1e-9)))
        hnr_db = max(0.0, min(40.0, hnr_db))
        stability_index = max(0.0, min(100.0, 100.0 - 2.5 * jitter_pct - 1.5 * shimmer_pct + 0.5 * hnr_db))
    else:
        f0_mean, f0_std, f0_min, f0_max = 0.0, 0.0, 0.0, 0.0
        jitter_pct, shimmer_pct, hnr_db, stability_index = 0.0, 0.0, 0.0, 0.0

    # 3. Spectral Descriptors (Centroid, Spread, Rolloff 85%, Flatness)
    nperseg = min(1024, n)
    freqs, psd = signal.welch(sig - np.mean(sig), fs=fs, nperseg=nperseg)
    psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
    psd_sum = np.sum(psd) + 1e-12
    norm_psd = psd / psd_sum

    spec_centroid = float(np.sum(freqs * norm_psd))
    spec_spread = float(np.sqrt(max(0.0, np.sum(((freqs - spec_centroid) ** 2) * norm_psd))))

    # Spectral Rolloff 85%
    cum_psd = np.cumsum(norm_psd)
    rolloff_idx = np.searchsorted(cum_psd, 0.85)
    spec_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Spectral Flatness (Wiener entropy)
    geom_mean = np.exp(np.mean(np.log(np.maximum(psd, 1e-12))))
    arith_mean = np.mean(psd) + 1e-12
    spec_flatness = float(geom_mean / arith_mean)

    # 4. 13 MFCC Coefficients
    mfcc_means = compute_mfcc_features(sig, fs=fs, num_ceps=13)

    feats = {
        'voice_rms': safe_float(rms_val),
        'voice_zcr': safe_float(zcr_val),
        'voice_amp_env_mean': safe_float(amp_mean),
        'voice_amp_env_std': safe_float(amp_std),
        'voice_amp_env_max': safe_float(amp_max),
        'voice_f0_mean': safe_float(f0_mean),
        'voice_f0_std': safe_float(f0_std),
        'voice_f0_min': safe_float(f0_min),
        'voice_f0_max': safe_float(f0_max),
        'voice_jitter_pct': safe_float(jitter_pct),
        'voice_shimmer_pct': safe_float(shimmer_pct),
        'voice_hnr_db': safe_float(hnr_db),
        'voice_stability_index': safe_float(stability_index),
        'voice_spectral_centroid': safe_float(spec_centroid),
        'voice_spectral_spread': safe_float(spec_spread),
        'voice_spectral_rolloff': safe_float(spec_rolloff),
        'voice_spectral_flatness': safe_float(spec_flatness),
        'voice_duration_sec': safe_float(duration_sec),
        'voice_sample_count': safe_float(n),
    }
    for i in range(13):
        feats[f'voice_mfcc_{i+1}'] = safe_float(mfcc_means[i])
    feats['voice_mfcc_mean'] = safe_float(np.mean(mfcc_means))
    feats['voice_mfcc_std'] = safe_float(np.std(mfcc_means))

    return feats

def extract_patient_features(tremor_dfs, neck_dfs, age=40, gender='Female', voice_files=None):
    """
    Aggregates multi-trial tremor, neck, and voice features at the patient level
    and adds demographic descriptors.
    """
    # Normalize tremor_dfs and neck_dfs if passed as single DataFrame or non-iterable
    if tremor_dfs is not None:
        if isinstance(tremor_dfs, pd.DataFrame):
            tremor_dfs = [tremor_dfs]
        elif not isinstance(tremor_dfs, (list, tuple)):
            tremor_dfs = [tremor_dfs]
    else:
        tremor_dfs = []

    if neck_dfs is not None:
        if isinstance(neck_dfs, pd.DataFrame):
            neck_dfs = [neck_dfs]
        elif not isinstance(neck_dfs, (list, tuple)):
            neck_dfs = [neck_dfs]
    else:
        neck_dfs = []

    # Extract features for all tremor trials
    clean_tdfs = [tdf for tdf in tremor_dfs if tdf is not None and len(tdf) > 0]
    clean_ndfs = [ndf for ndf in neck_dfs if ndf is not None and len(ndf) > 0]

    tremor_trial_feats = [extract_tremor_features(tdf) for tdf in clean_tdfs] if clean_tdfs else [extract_tremor_features(pd.DataFrame())]
    neck_trial_feats = [extract_neck_features(ndf) for ndf in clean_ndfs] if clean_ndfs else [extract_neck_features(pd.DataFrame())]

    # Voice trials: normalize if passed as single string, BytesIO, UploadedFile, or ndarray
    if voice_files is not None:
        if isinstance(voice_files, (str, bytes, io.BytesIO, np.ndarray)) or hasattr(voice_files, 'read'):
            v_list = [voice_files]
        elif isinstance(voice_files, (list, tuple)):
            v_list = list(voice_files)
        else:
            v_list = [voice_files]
    else:
        v_list = []

    clean_vfiles = [vf for vf in v_list if vf is not None]
    if clean_vfiles:
        voice_trial_feats = [extract_voice_features(vf) for vf in clean_vfiles]
    else:
        voice_trial_feats = [extract_voice_features(None)]

    # Aggregate across trials
    df_t = pd.DataFrame(tremor_trial_feats)
    df_n = pd.DataFrame(neck_trial_feats)
    df_v = pd.DataFrame(voice_trial_feats)

    patient_feats = {}

    # Primary trial features (mean across trials)
    for col in df_t.columns:
        patient_feats[col] = safe_float(df_t[col].mean())
    for col in df_n.columns:
        patient_feats[col] = safe_float(df_n[col].mean())
    for col in df_v.columns:
        patient_feats[col] = safe_float(df_v[col].mean())

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
