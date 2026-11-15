"""
Comprehensive Test Suite for Multimodal Thyroid Screening System
Validates:
1. Feature extraction pipelines (tremor, neck distance, biometrics)
2. Edge cases (empty data, NaNs, zero division, constant signals)
3. Model inferences and probability distributions (ExtraTrees, ANN_top40, RF, SVM)
4. Exact match of held-out poster benchmark statistics (63.6% accuracy, 7/11, recalls)
5. Artifact integrity (metadata, models, scalers, feature column mappings)
"""

import os
import sys
import json
import pickle
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_extractor import (extract_tremor_features, extract_neck_features,
                                   extract_patient_features, compute_spectral_features,
                                   compute_time_features)

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        # Synthetic standard 100 Hz tremor signal (10 seconds = 1000 samples)
        t = np.linspace(0, 10, 1000)
        self.synth_tremor = pd.DataFrame({
            'accel_x_m_s2': np.sin(2 * np.pi * 9 * t) + 0.1 * np.random.randn(1000),
            'accel_y_m_s2': np.cos(2 * np.pi * 9 * t) + 0.1 * np.random.randn(1000),
            'accel_z_m_s2': 9.8 + 0.2 * np.sin(2 * np.pi * 9 * t),
            'gyro_x_rad_s': 0.05 * np.sin(2 * np.pi * 9 * t),
            'gyro_y_rad_s': 0.05 * np.cos(2 * np.pi * 9 * t),
            'gyro_z_rad_s': 0.02 * np.random.randn(1000),
            'measurement_status': ['OK'] * 1000
        })

        # Synthetic neck distance signal (10 samples)
        self.synth_neck = pd.DataFrame({
            'distance_mm': [52.1, 52.3, 51.9, 52.0, 52.2, 52.1, 52.4, 51.8, 52.0, 52.2],
            'surface_condition': ['T'] * 10,
            'measurement_status': ['OK'] * 10
        })

    def test_tremor_feature_extraction_types_and_ranges(self):
        feats = extract_tremor_features(self.synth_tremor)
        self.assertIsInstance(feats, dict)
        self.assertGreater(len(feats), 350)
        # Check specific physiological tremor band presence
        self.assertIn("tremor_ax_ratio_8_12", feats)
        self.assertIn("tremor_amag_dom_freq", feats)
        # Dominant frequency should be near 9 Hz for synthetic signal
        self.assertAlmostEqual(feats["tremor_ax_dom_freq"], 9.0, delta=1.5)
        # Ratio in 8-12 Hz should be high
        self.assertGreater(feats["tremor_ax_ratio_8_12"], 0.20)

    def test_neck_feature_extraction(self):
        feats = extract_neck_features(self.synth_neck)
        self.assertIsInstance(feats, dict)
        self.assertAlmostEqual(feats["neck_mean_mm"], 52.1, delta=0.5)
        self.assertAlmostEqual(feats["neck_sample_count"], 10.0)
        self.assertEqual(feats["neck_surface_condition_code"], 1.0)

    def test_patient_feature_aggregation(self):
        feats = extract_patient_features([self.synth_tremor], [self.synth_neck], age=45, gender="Male")
        self.assertIn("demo_age", feats)
        self.assertEqual(feats["demo_age"], 45.0)
        self.assertEqual(feats["demo_gender"], 0.0) # Male is 0
        self.assertIn("tremor_amag_mean", feats)
        self.assertIn("neck_mean_mm", feats)

    def test_edge_case_empty_dataframes(self):
        empty_tremor = pd.DataFrame()
        empty_neck = pd.DataFrame()
        feats = extract_patient_features([empty_tremor], [empty_neck], age=30, gender="Female")
        self.assertIsInstance(feats, dict)
        # All values should be finite numbers, no NaNs
        for k, v in feats.items():
            self.assertFalse(np.isnan(v), f"Feature {k} returned NaN on empty input!")
            self.assertFalse(np.isinf(v), f"Feature {k} returned Inf on empty input!")

    def test_edge_case_constant_signal(self):
        const_sig = np.ones(500) * 5.0
        time_feats = compute_time_features(const_sig, prefix="const")
        spec_feats = compute_spectral_features(const_sig, prefix="const")
        self.assertEqual(time_feats["const_std"], 0.0)
        self.assertEqual(time_feats["const_skew"], 0.0)
        self.assertEqual(time_feats["const_kurt"], 0.0)
        self.assertFalse(np.isnan(spec_feats["const_spectral_entropy"]))


class TestModelPerformanceAndPosterMatch(unittest.TestCase):
    def setUp(self):
        self.models_dir = "models"
        with open(os.path.join(self.models_dir, "primary_extra_trees.pkl"), "rb") as f:
            self.primary_et = pickle.load(f)
        with open(os.path.join(self.models_dir, "ann_top40.pkl"), "rb") as f:
            self.ann_model = pickle.load(f)
        with open(os.path.join(self.models_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.models_dir, "selector_60.pkl"), "rb") as f:
            self.selector_60 = pickle.load(f)
        with open(os.path.join(self.models_dir, "selector_40.pkl"), "rb") as f:
            self.selector_40 = pickle.load(f)
        with open(os.path.join(self.models_dir, "feature_columns.json"), "r") as f:
            self.feature_cols = json.load(f)
        with open(os.path.join(self.models_dir, "benchmark_results.json"), "r") as f:
            self.benchmarks = json.load(f)
        self.test_df = pd.read_csv(os.path.join(self.models_dir, "heldout_test_set.csv"))

    def test_heldout_test_size_and_classes(self):
        self.assertEqual(len(self.test_df), 11)
        counts = self.test_df['label'].value_counts()
        self.assertEqual(counts['Normal'], 4)
        self.assertEqual(counts['Hyperthyroid'], 3)
        self.assertEqual(counts['Hypothyroid'], 4)

    def test_primary_extratrees_matches_poster_exactly(self):
        X_test_raw = self.test_df[self.feature_cols].fillna(0).values
        y_test = self.test_df['label'].values

        X_test_scaled = self.scaler.transform(X_test_raw)
        X_test_60 = self.selector_60.transform(X_test_scaled)

        preds = self.primary_et.predict(X_test_60)
        correct_count = int(np.sum(preds == y_test))
        acc = float(correct_count / len(y_test))

        # Check exact 7/11 match (63.6%)
        self.assertEqual(correct_count, 7, f"Expected 7/11 correct, got {correct_count}/11")
        self.assertAlmostEqual(acc, 0.636, delta=0.005)

        # Check class recalls
        norm_mask = (y_test == 'Normal')
        hyper_mask = (y_test == 'Hyperthyroid')
        hypo_mask = (y_test == 'Hypothyroid')

        rec_norm = float(np.mean(preds[norm_mask] == 'Normal'))
        rec_hyper = float(np.mean(preds[hyper_mask] == 'Hyperthyroid'))
        rec_hypo = float(np.mean(preds[hypo_mask] == 'Hypothyroid'))

        self.assertAlmostEqual(rec_norm, 0.50, delta=0.01) # 2/4 = 50.0%
        self.assertAlmostEqual(rec_hyper, 0.667, delta=0.01) # 2/3 = 66.7%
        self.assertAlmostEqual(rec_hypo, 0.75, delta=0.01) # 3/4 = 75.0%

        # Balanced accuracy = (0.50 + 0.667 + 0.75) / 3 = 63.9%
        bal_acc = (rec_norm + rec_hyper + rec_hypo) / 3
        self.assertAlmostEqual(bal_acc, 0.639, delta=0.005)

    def test_probabilities_sum_to_one(self):
        X_test_raw = self.test_df[self.feature_cols].fillna(0).values
        X_test_scaled = self.scaler.transform(X_test_raw)
        X_test_60 = self.selector_60.transform(X_test_scaled)
        probs = self.primary_et.predict_proba(X_test_60)
        sums = np.sum(probs, axis=1)
        for s in sums:
            self.assertAlmostEqual(s, 1.0, places=5)

    def test_compact_ann_execution(self):
        X_test_raw = self.test_df[self.feature_cols].fillna(0).values
        X_test_scaled = self.scaler.transform(X_test_raw)
        X_test_40 = self.selector_40.transform(X_test_scaled)
        preds = self.ann_model.predict(X_test_40)
        self.assertEqual(len(preds), 11)
        self.assertTrue(set(preds).issubset({'Normal', 'Hyperthyroid', 'Hypothyroid'}))

    def test_benchmarks_table_integrity(self):
        b_names = [b["model"] for b in self.benchmarks]
        self.assertIn("ExtraTrees (Primary v9)", b_names)
        self.assertIn("Compact ANN (ANN_top40)", b_names)
        self.assertIn("Tremor-Only RF", b_names)
        self.assertIn("QML Simulator", b_names)

        et_entry = [b for b in self.benchmarks if b["model"] == "ExtraTrees (Primary v9)"][0]
        self.assertEqual(et_entry["accuracy"], 0.636)
        self.assertEqual(et_entry["macro_f1"], 0.635)
        self.assertEqual(et_entry["balanced_acc"], 0.639)
        self.assertEqual(et_entry["macro_auc"], 0.700)
        self.assertEqual(et_entry["mcc"], 0.475)


class TestInteractiveWebUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import find_patient_sensor_files, normalize_pid
        cls.find_patient_sensor_files = staticmethod(find_patient_sensor_files)
        cls.normalize_pid = staticmethod(normalize_pid)
        cls.app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

    def test_patient_folder_resolver_handles_naming_variations(self):
        # TH030 is named TH30 in Thyroid_Data/27.10.26/TH30/
        t_files, n_files = self.find_patient_sensor_files("TH030")
        self.assertGreater(len(t_files), 0, "Failed to resolve TH030 to TH30 folder!")
        self.assertGreater(len(n_files), 0, "Failed to resolve neck files for TH030!")

        # TH054 standard
        t54, n54 = self.find_patient_sensor_files("TH054")
        self.assertGreater(len(t54), 0)
        self.assertGreater(len(n54), 0)

    def test_streamlit_app_initialization_and_rendering(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(self.app_path)
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0, f"AppTest threw exceptions on load: {[e.value for e in at.exception]}")

    def test_streamlit_all_pages_render_without_pyarrow_errors(self):
        from streamlit.testing.v1 import AppTest
        menu_items = [
            "🏠 Overview",
            "👥 Patients",
            "📱 Devices",
            "⏱️ Test Sessions",
            "📊 Sensor Data",
            "🩺 Machine Learning",
            "📄 Reports",
            "⚙️ Settings / Audit Trail"
        ]
        for m in menu_items:
            at = AppTest.from_file(self.app_path)
            at.run(timeout=15)
            r = [r for r in at.radio if "Select Module" in r.proto.label][0]
            r.set_value(m).run(timeout=15)
            self.assertEqual(len(at.exception), 0, f"Page [{m}] raised exception: {[e.value for e in at.exception]}")

    def test_live_screening_heldout_patients_interactive(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(self.app_path)
        at.run(timeout=15)

        # Navigate to Machine Learning module
        r = [r for r in at.radio if "Select Module" in r.proto.label][0]
        r.set_value("🩺 Machine Learning").run(timeout=15)

        p_select = [sb for sb in at.selectbox if "Held-Out" in sb.proto.label][0]
        # Test TH030 specifically
        th030_opts = [o for o in p_select.options if "TH030" in o]
        self.assertTrue(len(th030_opts) > 0)
        p_select.set_value(th030_opts[0]).run(timeout=10)
        self.assertEqual(len(at.exception), 0)
        # Verify screening result badges were generated
        badges = [m.value for m in at.markdown if "badge-" in m.value]
        self.assertGreater(len(badges), 0, "No screening badges rendered for TH030!")

    def test_custom_upload_ui_selection(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(self.app_path)
        at.run(timeout=15)

        # Navigate to Machine Learning module
        r = [r for r in at.radio if "Select Module" in r.proto.label][0]
        r.set_value("🩺 Machine Learning").run(timeout=15)

        ds_radio = [r for r in at.radio if "Select Data Source" in r.proto.label][0]
        ds_radio.set_value("Upload Custom Sensor CSVs (ESP32 Live Data)").run(timeout=10)
        self.assertEqual(len(at.exception), 0)
        # Demographic controls (age slider and gender selectbox) should be present
        sliders = [s for s in at.slider if "Age" in s.proto.label]
        self.assertEqual(len(sliders), 1, "Age slider should be rendered for custom upload")

    def test_pdf_generation_validity(self):
        from src.pdf_generator import generate_patient_pdf
        pdf_bytes = generate_patient_pdf(
            patient_id="TH054",
            patient_name="Mahbuba Khatun",
            age=35.0,
            gender="Female",
            weight="58 kg",
            tsh_val="2.1",
            ft4_val="1.2 ng/dL",
            ref_status="Normal",
            pred_label="Normal",
            prob_dict={"Normal": 0.75, "Hyperthyroid": 0.15, "Hypothyroid": 0.10},
            model_name="ExtraTrees (Primary v9, 60 features) [Poster Match: 63.6%]",
            sensor_feats={
                "tremor_amag_ratio_8_12": 0.18,
                "tremor_amag_dom_freq": 9.2,
                "tremor_amag_rms": 0.42,
                "neck_mean_mm": 51.4,
                "neck_std_mm": 0.35
            },
            researcher_name="Dr. Sifat",
            researcher_phone="01712345678",
            researcher_gender="Male",
            clinical_notes="Verified multimodal profile."
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))
        self.assertGreater(len(pdf_bytes), 2000)

    def test_custom_sensor_upload_aliased_columns_and_inference(self):
        from app import run_screening_inference
        # DataFrame using shorthand column aliases
        custom_tremor = pd.DataFrame({
            'ax': np.random.randn(200) * 0.5,
            'ay': np.random.randn(200) * 0.5,
            'az': 9.8 + np.random.randn(200) * 0.2,
            'gx': np.random.randn(200) * 0.05,
            'gy': np.random.randn(200) * 0.05,
            'gz': np.random.randn(200) * 0.05,
            'measurement_status': [' ok '] * 200
        })
        custom_neck = pd.DataFrame({
            'distance': [48.5, 49.0, 48.8, 49.2, 48.6] * 2,
            'surface_condition': ['T'] * 10
        })
        pred, probs, feats, used = run_screening_inference(
            custom_tremor, custom_neck, age=42, gender="Male",
            model_choice="ExtraTrees (Primary v9, 60 features) [Poster Match: 63.6%]"
        )
        self.assertIn(pred, ['Normal', 'Hyperthyroid', 'Hypothyroid'])
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
        self.assertEqual(feats['demo_age'], 42.0)
        self.assertEqual(feats['demo_gender'], 0.0)

    def test_unauthenticated_login_flow(self):
        from streamlit.testing.v1 import AppTest
        # Force login screen via environment variable
        os.environ["FORCE_LOGIN_SCREEN"] = "1"
        try:
            at = AppTest.from_file(self.app_path)
            at.run(timeout=15)
            self.assertEqual(len(at.exception), 0)
            # Login card inputs must be present
            phones = [ti for ti in at.text_input if "Phone Number" in ti.proto.label]
            self.assertEqual(len(phones), 1, "Phone number input should be present on login screen")
            genders = [sb for sb in at.selectbox if "Gender" in sb.proto.label]
            self.assertEqual(len(genders), 1, "Gender selectbox should be present on login screen")
        finally:
            os.environ.pop("FORCE_LOGIN_SCREEN", None)


if __name__ == '__main__':
    unittest.main()
