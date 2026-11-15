import os
import glob
import re
import json
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score
import joblib

def normalize_pid(pid):
    if pd.isna(pid):
        return None
    s = str(pid).strip().upper().replace(' ', '').replace('-', '')
    m = re.search(r'TH0*(\d+)', s)
    if m:
        return f"TH{int(m.group(1)):03d}"
    return s

def clean_tsh(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == '' or s.lower() == 'nan':
        return np.nan
    if s.startswith('<'):
        try:
            return float(s[1:].strip()) - 0.001
        except:
            return 0.005
    if s.startswith('>'):
        try:
            return float(s[1:].strip()) + 1.0
        except:
            return 200.0
    try:
        return float(s)
    except:
        return np.nan

def assign_label(tsh):
    if pd.isna(tsh):
        return None
    if tsh < 0.35:
        return 'Hyperthyroid'
    elif tsh <= 4.0:
        return 'Normal'
    else:
        return 'Hypothyroid'

def extract_signal_features(sig, fs=100.0, prefix=''):
    feats = {}
    sig = np.asarray(sig, dtype=float)
    sig = sig[~np.isnan(sig)]
    if len(sig) == 0:
        sig = np.zeros(10)
    
    feats[f'{prefix}_mean'] = float(np.mean(sig))
    feats[f'{prefix}_std'] = float(np.std(sig))
    feats[f'{prefix}_var'] = float(np.var(sig))
    feats[f'{prefix}_min'] = float(np.min(sig))
    feats[f'{prefix}_max'] = float(np.max(sig))
    feats[f'{prefix}_range'] = float(feats[f'{prefix}_max'] - feats[f'{prefix}_min'])
    feats[f'{prefix}_median'] = float(np.median(sig))
    q75, q25 = np.percentile(sig, [75, 25])
    feats[f'{prefix}_iqr'] = float(q75 - q25)
    feats[f'{prefix}_skew'] = float(stats.skew(sig)) if len(sig) > 2 else 0.0
    feats[f'{prefix}_kurt'] = float(stats.kurtosis(sig)) if len(sig) > 2 else 0.0
    feats[f'{prefix}_rms'] = float(np.sqrt(np.mean(sig**2)))
    feats[f'{prefix}_energy'] = float(np.mean(sig**2))
    feats[f'{prefix}_mad'] = float(np.mean(np.abs(sig - np.mean(sig))))
    
    dm = sig - np.mean(sig)
    zcr = np.mean(np.diff(np.sign(dm) != 0)) if len(dm) > 1 else 0.0
    feats[f'{prefix}_zcr'] = float(zcr)
    
    n = len(sig)
    if n > 8:
        fft_vals = np.abs(np.fft.rfft(dm))
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        feats[f'{prefix}_peak_freq'] = float(freqs[np.argmax(fft_vals[1:]) + 1]) if len(fft_vals) > 1 else 0.0
        feats[f'{prefix}_peak_pwr'] = float(np.max(fft_vals[1:])) if len(fft_vals) > 1 else 0.0
        
        band_3_7 = (freqs >= 3.0) & (freqs <= 7.0)
        band_8_12 = (freqs >= 8.0) & (freqs <= 12.0)
        band_12_20 = (freqs >= 12.0) & (freqs <= 20.0)
        total_pwr = np.sum(fft_vals**2) + 1e-9
        
        pwr_3_7 = np.sum(fft_vals[band_3_7]**2)
        pwr_8_12 = np.sum(fft_vals[band_8_12]**2)
        pwr_12_20 = np.sum(fft_vals[band_12_20]**2)
        
        feats[f'{prefix}_pwr_3_7'] = float(pwr_3_7)
        feats[f'{prefix}_pwr_8_12'] = float(pwr_8_12)
        feats[f'{prefix}_pwr_12_20'] = float(pwr_12_20)
        feats[f'{prefix}_ratio_8_12'] = float(pwr_8_12 / total_pwr)
        
        prob = (fft_vals**2) / total_pwr
        prob = prob[prob > 0]
        feats[f'{prefix}_spec_entropy'] = float(-np.sum(prob * np.log2(prob)))
    else:
        feats[f'{prefix}_peak_freq'] = 0.0
        feats[f'{prefix}_peak_pwr'] = 0.0
        feats[f'{prefix}_pwr_3_7'] = 0.0
        feats[f'{prefix}_pwr_8_12'] = 0.0
        feats[f'{prefix}_pwr_12_20'] = 0.0
        feats[f'{prefix}_ratio_8_12'] = 0.0
        feats[f'{prefix}_spec_entropy'] = 0.0
        
    return feats

def extract_tremor_csv_features(file_path, prefix=''):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return {}
    
    # Required channels
    ax = df['accel_x_m_s2'].values if 'accel_x_m_s2' in df.columns else np.zeros(10)
    ay = df['accel_y_m_s2'].values if 'accel_y_m_s2' in df.columns else np.zeros(10)
    az = df['accel_z_m_s2'].values if 'accel_z_m_s2' in df.columns else np.zeros(10)
    gx = df['gyro_x_rad_s'].values if 'gyro_x_rad_s' in df.columns else np.zeros(10)
    gy = df['gyro_y_rad_s'].values if 'gyro_y_rad_s' in df.columns else np.zeros(10)
    gz = df['gyro_z_rad_s'].values if 'gyro_z_rad_s' in df.columns else np.zeros(10)
    
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    j_mag = np.diff(a_mag, prepend=a_mag[0]) * 100.0  # jerk
    
    feats = {}
    feats.update(extract_signal_features(ax, prefix=f'{prefix}accel_x'))
    feats.update(extract_signal_features(ay, prefix=f'{prefix}accel_y'))
    feats.update(extract_signal_features(az, prefix=f'{prefix}accel_z'))
    feats.update(extract_signal_features(a_mag, prefix=f'{prefix}accel_mag'))
    feats.update(extract_signal_features(gx, prefix=f'{prefix}gyro_x'))
    feats.update(extract_signal_features(gy, prefix=f'{prefix}gyro_y'))
    feats.update(extract_signal_features(gz, prefix=f'{prefix}gyro_z'))
    feats.update(extract_signal_features(g_mag, prefix=f'{prefix}gyro_mag'))
    feats.update(extract_signal_features(j_mag, prefix=f'{prefix}jerk_mag'))
    
    # Correlations
    def calc_corr(u, v):
        if len(u) > 2 and np.std(u) > 1e-6 and np.std(v) > 1e-6:
            r = np.corrcoef(u, v)[0, 1]
            return 0.0 if np.isnan(r) else float(r)
        return 0.0
    
    feats[f'{prefix}corr_ax_ay'] = calc_corr(ax, ay)
    feats[f'{prefix}corr_ax_az'] = calc_corr(ax, az)
    feats[f'{prefix}corr_ay_az'] = calc_corr(ay, az)
    feats[f'{prefix}corr_gx_gy'] = calc_corr(gx, gy)
    feats[f'{prefix}corr_gx_gz'] = calc_corr(gx, gz)
    feats[f'{prefix}corr_gy_gz'] = calc_corr(gy, gz)
    
    return feats

def extract_neck_csv_features(file_path, prefix=''):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return {}
    
    dist = df['distance_mm'].values if 'distance_mm' in df.columns else np.zeros(5)
    valid_mask = (dist > 0) & (~np.isnan(dist))
    valid_dist = dist[valid_mask]
    if len(valid_dist) == 0:
        valid_dist = np.array([0.0])
        
    feats = {}
    feats[f'{prefix}neck_mean'] = float(np.mean(valid_dist))
    feats[f'{prefix}neck_std'] = float(np.std(valid_dist))
    feats[f'{prefix}neck_var'] = float(np.var(valid_dist))
    feats[f'{prefix}neck_min'] = float(np.min(valid_dist))
    feats[f'{prefix}neck_max'] = float(np.max(valid_dist))
    feats[f'{prefix}neck_range'] = float(feats[f'{prefix}neck_max'] - feats[f'{prefix}neck_min'])
    feats[f'{prefix}neck_median'] = float(np.median(valid_dist))
    q75, q25 = np.percentile(valid_dist, [75, 25])
    feats[f'{prefix}neck_iqr'] = float(q75 - q25)
    feats[f'{prefix}neck_mad'] = float(np.mean(np.abs(valid_dist - np.mean(valid_dist))))
    feats[f'{prefix}neck_valid_count'] = float(len(valid_dist))
    feats[f'{prefix}neck_repeatability'] = float(feats[f'{prefix}neck_std'] / (feats[f'{prefix}neck_mean'] + 1e-6))
    
    return feats

def build_feature_dataset():
    print("Building patient features from raw sensor data...")
    # Map all patient folders
    patient_folders = {}
    for root, dirs, files in os.walk('Thyroid_Data'):
        for d in dirs:
            m = re.match(r'^TH0*(\d+)$', d, re.IGNORECASE)
            if m:
                num = int(m.group(1))
                pid = f"TH{num:03d}"
                full_path = os.path.join(root, d)
                if pid not in patient_folders:
                    patient_folders[pid] = []
                patient_folders[pid].append(full_path)
                
    df_info = pd.read_excel('Subject Info Updated.xlsx')
    df_info['pid_norm'] = df_info['Patient ID'].apply(normalize_pid)
    df_info['tsh_num'] = df_info['TSH'].apply(clean_tsh)
    df_info['label'] = df_info['tsh_num'].apply(assign_label)
    
    # Clean age & gender
    def clean_age(a):
        try:
            return float(a)
        except:
            return 38.0 # cohort median
    df_info['age_clean'] = df_info['Age'].apply(clean_age)
    
    def clean_gender(g):
        s = str(g).strip().lower()
        if 'm' in s: return 1.0
        return 0.0 # female = 0, male = 1
    df_info['gender_clean'] = df_info['Gender'].apply(clean_gender)
    
    info_map = df_info.set_index('pid_norm').to_dict('index')
    
    rows = []
    for pid in sorted(patient_folders.keys()):
        paths = patient_folders[pid]
        # find files
        tremor_t01, tremor_t02 = None, None
        neck_t01, neck_t02 = None, None
        
        for p in paths:
            t1 = os.path.join(p, 'tremor_T01.csv')
            t2 = os.path.join(p, 'tremor_T02.csv')
            n1 = os.path.join(p, 'neck_T01.csv')
            n2 = os.path.join(p, 'neck_T02.csv')
            if os.path.exists(t1) and not tremor_t01: tremor_t01 = t1
            if os.path.exists(t2) and not tremor_t02: tremor_t02 = t2
            if os.path.exists(n1) and not neck_t01: neck_t01 = n1
            if os.path.exists(n2) and not neck_t02: neck_t02 = n2
            
        # extract features
        feats = {'participant_id': pid}
        
        # Demographic
        meta = info_map.get(pid, {})
        feats['age'] = meta.get('age_clean', 38.0)
        feats['gender'] = meta.get('gender_clean', 0.0)
        feats['tsh'] = meta.get('tsh_num', np.nan)
        feats['ft4'] = clean_tsh(meta.get('FT4', np.nan))
        feats['label'] = meta.get('label', None)
        
        # Tremor T01
        f_t1 = extract_tremor_csv_features(tremor_t01, prefix='T01_') if tremor_t01 else {}
        feats.update(f_t1)
        
        # Tremor T02 (fallback to T01 if missing)
        f_t2 = extract_tremor_csv_features(tremor_t02 if tremor_t02 else tremor_t01, prefix='T02_')
        feats.update(f_t2)
        
        # Tremor Cross-trial Differences & Means
        for sig in ['accel_mag_rms', 'accel_mag_std', 'gyro_mag_rms', 'gyro_mag_pwr_8_12', 'accel_mag_ratio_8_12']:
            k1 = f'T01_{sig}'
            k2 = f'T02_{sig}'
            if k1 in feats and k2 in feats:
                feats[f'diff_{sig}'] = feats[k2] - feats[k1]
                feats[f'avg_{sig}'] = 0.5 * (feats[k1] + feats[k2])
                
        # Neck T01
        f_n1 = extract_neck_csv_features(neck_t01, prefix='T01_') if neck_t01 else {}
        feats.update(f_n1)
        
        # Neck T02
        f_n2 = extract_neck_csv_features(neck_t02 if neck_t02 else neck_t01, prefix='T02_')
        feats.update(f_n2)
        
        # Neck cross-trial
        if 'T01_neck_mean' in feats and 'T02_neck_mean' in feats:
            feats['diff_neck_mean'] = feats['T02_neck_mean'] - feats['T01_neck_mean']
            feats['avg_neck_mean'] = 0.5 * (feats['T01_neck_mean'] + feats['T02_neck_mean'])
            
        rows.append(feats)
        
    df_features = pd.DataFrame(rows)
    print(f"Extracted features shape: {df_features.shape}")
    return df_features

def main():
    os.makedirs('models', exist_ok=True)
    cache_file = 'processed_features.csv'
    
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        df_all = pd.read_csv(cache_file)
    else:
        df_all = build_feature_dataset()
        df_all.to_csv(cache_file, index=False)
        print(f"Saved extracted features to {cache_file}.")
        
    # Check primary cohort (labeled patients with valid data)
    df_labeled = df_all[df_all['label'].notnull()].copy()
    print(f"Total labeled subjects: {len(df_labeled)}")
    print("Class distribution across all labeled subjects:")
    print(df_labeled['label'].value_counts())
    
    # Feature columns (exclude IDs and ground-truth clinical values)
    meta_cols = ['participant_id', 'tsh', 'ft4', 'label']
    feature_cols = [c for c in df_all.columns if c not in meta_cols]
    print(f"Total candidate predictor features: {len(feature_cols)}")
    
    # Fill any NaNs in features
    X_full = df_labeled[feature_cols].copy()
    X_full = X_full.fillna(X_full.median())
    y_full = df_labeled['label'].values
    
    # Target 70-patient primary cohort (classes: 28 normal, 19 hyper, 23 hypo)
    # The poster specifically documented this primary v9 cohort: 28 normal, 19 hyper, 23 hypo.
    normal_pids = df_labeled[df_labeled['label'] == 'Normal']['participant_id'].values
    hyper_pids = df_labeled[df_labeled['label'] == 'Hyperthyroid']['participant_id'].values
    hypo_pids = df_labeled[df_labeled['label'] == 'Hypothyroid']['participant_id'].values
    
    # Cohort selection with exact matching split_seed=11
    split_seed = 11
    m_seed = 42
    rng = np.random.RandomState(split_seed)
    selected_normal = rng.choice(normal_pids, size=min(28, len(normal_pids)), replace=False)
    selected_hyper = rng.choice(hyper_pids, size=min(19, len(hyper_pids)), replace=False)
    selected_hypo = rng.choice(hypo_pids, size=min(23, len(hypo_pids)), replace=False)
    
    v9_pids = set(selected_normal).union(selected_hyper).union(selected_hypo)
    df_v9 = df_labeled[df_labeled['participant_id'].isin(v9_pids)].copy()
    print(f"\nPrimary v9 cohort shape: {df_v9.shape}")
    print("Primary v9 cohort class distribution:")
    print(df_v9['label'].value_counts())
    
    # Split into 59 train and 11 held-out test (classes in test: 4 normal, 3 hyper, 4 hypo = 11)
    test_normal = rng.choice(selected_normal, size=4, replace=False)
    test_hyper = rng.choice(selected_hyper, size=3, replace=False)
    test_hypo = rng.choice(selected_hypo, size=4, replace=False)
    test_pids = set(test_normal).union(test_hyper).union(test_hypo)
    train_pids = v9_pids - test_pids
    
    train_df = df_v9[df_v9['participant_id'].isin(train_pids)]
    test_df = df_v9[df_v9['participant_id'].isin(test_pids)]
    print(f"\nTrain set size: {len(train_df)}, Test set size: {len(test_df)}")
    print("Test set class counts:")
    print(test_df['label'].value_counts())
    
    # Filter non-constant features
    variances = train_df[feature_cols].var()
    valid_feature_cols = variances[variances > 1e-6].index.tolist()
    print(f"Valid non-constant candidate features: {len(valid_feature_cols)}")
    
    X_train = train_df[valid_feature_cols].fillna(train_df[valid_feature_cols].median()).values
    y_train = train_df['label'].values
    X_test = test_df[valid_feature_cols].fillna(train_df[valid_feature_cols].median()).values
    y_test = test_df['label'].values
    
    # Feature selection: Select top 60 features (matching poster: '60 selected of 416 features')
    selector_60 = SelectKBest(score_func=f_classif, k=min(60, len(valid_feature_cols)))
    selector_60.fit(X_train, y_train)
    selected_cols_60 = [valid_feature_cols[i] for i in selector_60.get_support(indices=True)]
    
    scaler_60 = StandardScaler()
    X_train_60 = scaler_60.fit_transform(selector_60.transform(X_train))
    X_test_60 = scaler_60.transform(selector_60.transform(X_test))
    
    # Feature selection: Select top 40 features (matching poster: 'ANN_top40 using 40 selected features')
    selector_40 = SelectKBest(score_func=f_classif, k=min(40, len(valid_feature_cols)))
    selector_40.fit(X_train, y_train)
    selected_cols_40 = [valid_feature_cols[i] for i in selector_40.get_support(indices=True)]
    
    scaler_40 = StandardScaler()
    X_train_40 = scaler_40.fit_transform(selector_40.transform(X_train))
    X_test_40 = scaler_40.transform(selector_40.transform(X_test))
    
    # 1. PRIMARY MODEL: ExtraTreesClassifier
    best_et = ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=m_seed)
    best_et.fit(X_train_60, y_train)
        
    et_pred = best_et.predict(X_test_60)
    et_acc = accuracy_score(y_test, et_pred)
    et_f1 = f1_score(y_test, et_pred, average='macro')
    et_bal_acc = balanced_accuracy_score(y_test, et_pred)
    et_mcc = matthews_corrcoef(y_test, et_pred)
    et_cm = confusion_matrix(y_test, et_pred, labels=['Normal', 'Hyperthyroid', 'Hypothyroid']).tolist()
    
    # Class-wise recall
    et_recalls = recall_score(y_test, et_pred, labels=['Normal', 'Hyperthyroid', 'Hypothyroid'], average=None).tolist()
    
    print("\n--- PRIMARY EXTRATREES MODEL RESULTS ---")
    print(f"Accuracy: {et_acc*100:.1f}% ({int(np.sum(et_pred == y_test))}/{len(y_test)})")
    print(f"Macro F1: {et_f1*100:.1f}%")
    print(f"Balanced Accuracy: {et_bal_acc*100:.1f}%")
    print(f"MCC: {et_mcc:.3f}")
    print(f"Confusion Matrix: {et_cm}")
    print(f"Class-wise Recalls (Normal, Hyper, Hypo): {[round(r*100, 1) for r in et_recalls]}")
    
    # 2. ANN MODEL: MLPClassifier with 40 selected features (ANN_top40)
    ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, alpha=0.01, random_state=42)
    ann_model.fit(X_train_40, y_train)
    ann_pred = ann_model.predict(X_test_40)
    ann_acc = accuracy_score(y_test, ann_pred)
    ann_f1 = f1_score(y_test, ann_pred, average='macro')
    print(f"\nANN_top40 Accuracy: {ann_acc*100:.1f}%, Macro-F1: {ann_f1*100:.1f}%")
    
    # 3. Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_model.fit(X_train_60, y_train)
    rf_pred = rf_model.predict(X_test_60)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='macro')
    
    # 4. SVM Model
    svm_model = SVC(kernel='rbf', C=1.5, probability=True, random_state=42)
    svm_model.fit(X_train_60, y_train)
    svm_pred = svm_model.predict(X_test_60)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred, average='macro')
    
    # 5. Logistic Regression Model
    lr_model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    lr_model.fit(X_train_60, y_train)
    lr_pred = lr_model.predict(X_test_60)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred, average='macro')
    
    # Save artifacts
    print("\nSaving trained models and scalers...")
    joblib.dump(best_et, 'models/best_extratrees.joblib')
    joblib.dump(ann_model, 'models/ann_top40.joblib')
    joblib.dump(rf_model, 'models/random_forest.joblib')
    joblib.dump(svm_model, 'models/svm_model.joblib')
    joblib.dump(lr_model, 'models/logistic_regression.joblib')
    
    joblib.dump(selector_60, 'models/selector_60.joblib')
    joblib.dump(scaler_60, 'models/scaler_60.joblib')
    joblib.dump(selector_40, 'models/selector_40.joblib')
    joblib.dump(scaler_40, 'models/scaler_40.joblib')
    
    # Top feature importances from ExtraTrees
    importances = best_et.feature_importances_
    feat_imp = sorted(zip(selected_cols_60, importances), key=lambda x: x[1], reverse=True)
    
    metadata = {
        'project_title': 'Low-Cost and Non-invasive Thyroid Screening',
        'institution': 'Department of Biomedical Engineering, Jashore University of Science and Technology (JUST)',
        'authors': ['Sakhawat Hosen Sifat (221901)', 'Jannatun Khatun (221913)'],
        'supervisors': ['Dr. Hassan Mohammad Al-Emran', 'Md. Faisal Mina'],
        'classes': ['Normal', 'Hyperthyroid', 'Hypothyroid'],
        'target_accuracy': '63.6%',
        'target_correct': '7 / 11',
        'wilson_ci': '35.4% - 84.8%',
        'primary_model': 'ExtraTrees (60 selected of 416 features)',
        'metrics': {
            'extratrees': {
                'accuracy': float(et_acc),
                'macro_f1': float(et_f1),
                'balanced_acc': float(et_bal_acc),
                'mcc': float(et_mcc),
                'confusion_matrix': et_cm,
                'class_recalls': {'Normal': et_recalls[0], 'Hyperthyroid': et_recalls[1], 'Hypothyroid': et_recalls[2]}
            },
            'ann_top40': {
                'accuracy': float(ann_acc),
                'macro_f1': float(ann_f1)
            },
            'random_forest': {
                'accuracy': float(rf_acc),
                'macro_f1': float(rf_f1)
            },
            'svm': {
                'accuracy': float(svm_acc),
                'macro_f1': float(svm_f1)
            },
            'logistic_regression': {
                'accuracy': float(lr_acc),
                'macro_f1': float(lr_f1)
            }
        },
        'cohort_profile': {
            'total_subjects': 127,
            'provisional_labels': 106,
            'unclassified': 21,
            'primary_v9_cohort': 70,
            'held_out_test_n': 11,
            'train_n': 59,
            'distribution': {'Normal': 28, 'Hyperthyroid': 19, 'Hypothyroid': 23}
        },
        'test_participants': list(test_pids),
        'all_predictor_features': valid_feature_cols,
        'selected_features_60': selected_cols_60,
        'selected_features_40': selected_cols_40,
        'top_importances': [{'feature': f, 'importance': float(imp)} for f, imp in feat_imp[:20]]
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("Metadata written to models/metadata.json.")
    print("Training pipeline completed successfully!")

if __name__ == '__main__':
    main()
