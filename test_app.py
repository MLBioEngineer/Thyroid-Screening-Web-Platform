import os
import unittest
import numpy as np
import pandas as pd
import pickle
import joblib
import json

from src.feature_extractor import (
    extract_tremor_features, extract_neck_features,
    extract_patient_features, safe_float, SAMPLING_RATE_TREMOR
)
from app import find_patient_sensor_files, run_screening_inference, load_models_and_metadata

class TestThyroidPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('models/metadata.json') as f:
            cls.metadata = json.load(f)
        with open('models/benchmark_results.json') as f:
            cls.benchmarks = json.load(f)
            
        cls.primary_et_pkl = pickle.load(open('models/primary_extra_trees.pkl', 'rb'))
        cls.scaler_pkl = pickle.load(open('models/scaler.pkl', 'rb'))
        cls.selector_60_pkl = pickle.load(open('models/selector_60.pkl', 'rb'))
        cls.selector_40_pkl = pickle.load(open('models/selector_40.pkl', 'rb'))
        cls.ann_model = pickle.load(open('models/ann_top40.pkl', 'rb'))
        cls.rf_model = pickle.load(open('models/random_forest.pkl', 'rb'))
        cls.svm_model = pickle.load(open('models/svm.pkl', 'rb'))
        
        with open('models/feature_columns.json') as f:
            cls.feature_cols = json.load(f)
            
        cls.test_cohort = pd.read_csv('models/heldout_test_set.csv')

    def test_artifacts_exist(self):
        required_files = [
            'models/primary_extra_trees.pkl',
            'models/ann_top40.pkl',
            'models/random_forest.pkl',
            'models/svm.pkl',
            'models/scaler.pkl',
            'models/selector_60.pkl',
            'models/selector_40.pkl',
            'models/feature_columns.json',
            'models/selected_features_top60.json',
            'models/selected_features_top40.json',
            'models/benchmark_results.json',
            'models/heldout_test_set.csv',
            'models/metadata.json',
            'models/best_extratrees.joblib',
            'Subject Info Updated.xlsx'
        ]
        for f in required_files:
            self.assertTrue(os.path.exists(f), f"Missing required artifact: {f}")

    def test_poster_metrics_match(self):
        # 1. Check primary ExtraTrees benchmark in benchmark_results.json
        et_bm = next(b for b in self.benchmarks if 'ExtraTrees' in b['model'])
        self.assertAlmostEqual(et_bm['accuracy'], 0.636, delta=0.005)
        self.assertAlmostEqual(et_bm['macro_f1'], 0.635, delta=0.005)
        self.assertAlmostEqual(et_bm['balanced_acc'], 0.639, delta=0.005)
        self.assertEqual(et_bm['test_correct'], '7 / 11')
        self.assertEqual(et_bm['recall_normal'], 0.5)
        self.assertAlmostEqual(et_bm['recall_hyper'], 0.667, delta=0.005)
        self.assertEqual(et_bm['recall_hypo'], 0.75)

        # 2. Check metadata.json
        et_meta = self.metadata['metrics']['extratrees']
        self.assertAlmostEqual(et_meta['accuracy'], 0.636, delta=0.01)
        self.assertAlmostEqual(et_meta['balanced_acc'], 0.639, delta=0.01)

    def test_real_model_evaluation_on_test_cohort(self):
        # Verify primary_extra_trees.pkl achieves exactly 7/11 (63.6%) on heldout_test_set.csv
        X_raw = self.test_cohort[self.feature_cols].fillna(0).values
        y_true = self.test_cohort['label'].values
        
        X_scaled = self.scaler_pkl.transform(X_raw)
        X_60 = self.selector_60_pkl.transform(X_scaled)
        preds = self.primary_et_pkl.predict(X_60)
        
        correct = np.sum(preds == y_true)
        self.assertEqual(correct, 7, f"Expected 7/11 correct, got {correct}/11")
        acc = correct / len(y_true)
        self.assertAlmostEqual(acc, 0.636363, places=3)

    def test_patient_file_resolver_handles_all_naming(self):
        # Verify patient TH030 (stored on disk as TH30)
        t_files_30, n_files_30 = find_patient_sensor_files("TH030")
        self.assertGreater(len(t_files_30), 0, "Failed to resolve files for TH030 (disk folder TH30)")
        self.assertGreater(len(n_files_30), 0, "Failed to resolve neck files for TH030")

        # Verify patient TH012
        t_files_12, n_files_12 = find_patient_sensor_files("TH012")
        self.assertGreater(len(t_files_12), 0, "Failed to resolve files for TH012")

        # Verify all 11 held-out test patients have existing sensor files
        for pid in self.test_cohort['pid']:
            tf, nf = find_patient_sensor_files(pid)
            self.assertGreater(len(tf), 0, f"Missing tremor files for heldout test patient {pid}")
            self.assertGreater(len(nf), 0, f"Missing neck files for heldout test patient {pid}")

    def test_feature_extraction_nan_and_inf_resilience(self):
        # Stress-test feature extraction with NaNs, Infs, and extreme spikes
        n_samples = 400
        t_corrupt = pd.DataFrame({
            'accel_x_m_s2': [np.nan, np.inf, -np.inf, 999.0] + list(np.random.randn(n_samples - 4)),
            'accel_y_m_s2': [np.nan] * 5 + list(np.random.randn(n_samples - 5)),
            'accel_z_m_s2': [np.inf] + list(np.random.randn(n_samples - 1)),
            'gyro_x_rad_s': np.random.randn(n_samples),
            'gyro_y_rad_s': np.random.randn(n_samples),
            'gyro_z_rad_s': np.random.randn(n_samples),
        })
        n_corrupt = pd.DataFrame({
            'distance_mm': [np.nan, np.inf, -50.0] + [52.0] * 7,
            'surface_condition': ['T'] * 10
        })

        feats = extract_patient_features([t_corrupt], [n_corrupt], age=45, gender='Male')
        self.assertIsInstance(feats, dict)
        self.assertGreaterEqual(len(feats), 416)
        
        # Verify that EVERY value is finite (no NaNs, no Infs)
        for k, v in feats.items():
            self.assertTrue(np.isfinite(v), f"Feature {k} contains non-finite value: {v}")

    def test_feature_extraction_column_aliases(self):
        # Test alternative column names (ax, ay, az, gx, gy, gz, dist)
        n_samples = 200
        t_alias = pd.DataFrame({
            'ax': np.random.randn(n_samples),
            'ay': np.random.randn(n_samples),
            'az': np.random.randn(n_samples) + 9.8,
            'gx': np.random.randn(n_samples) * 0.05,
            'gy': np.random.randn(n_samples) * 0.05,
            'gz': np.random.randn(n_samples) * 0.05,
        })
        n_alias = pd.DataFrame({
            'dist': [48.0, 49.0, 50.0, 51.0, 50.0, 49.0, 50.0, 51.0, 50.0, 49.0]
        })

        feats = extract_patient_features([t_alias], [n_alias], age=28, gender='Female')
        self.assertIsInstance(feats, dict)
        self.assertGreater(feats['tremor_sample_count'], 50)
        self.assertAlmostEqual(feats['neck_mean_mm'], 49.7, places=1)

    def test_end_to_end_screening_inference_all_models(self):
        # Load sample patient TH054 files
        t_files, n_files = find_patient_sensor_files("TH054")
        t_df = pd.read_csv(t_files[0])
        n_df = pd.read_csv(n_files[0])

        models_to_test = [
            "ExtraTrees (Primary v9, 60 features) [Poster Match: 63.6%]",
            "Compact ANN (ANN_top40, 40 features)",
            "Multimodal Random Forest (60 features)",
            "Support Vector Machine (SVM RBF, 40 features)"
        ]

        for model_name in models_to_test:
            pred_label, prob_dict, feats, used_feats = run_screening_inference(
                t_df, n_df, age=35, gender='Female', model_choice=model_name
            )
            self.assertIn(pred_label, ['Normal', 'Hyperthyroid', 'Hypothyroid'])
            self.assertEqual(len(prob_dict), 3)
            prob_sum = sum(prob_dict.values())
            self.assertAlmostEqual(prob_sum, 1.0, places=4, msg=f"Probabilities do not sum to 1 in {model_name}")

if __name__ == '__main__':
    unittest.main()
