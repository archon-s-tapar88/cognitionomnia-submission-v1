#!/usr/bin/env python

import os
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from helper_code import *

# =========================================================================
# CONFIGURATION & GLOBAL UTILITIES
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# Optimization Hyperparameters
GLOBAL_DECISION_THRESHOLD = 0.46
DEMO_WEIGHT = 0.35
CAISR_WEIGHT = 0.45
WAVE_WEIGHT = 0.20

# =========================================================================
# FEATURE EXTRACTION ENGINE (ISOLATED MODALITIES)
# =========================================================================
def parse_demographics_modality(data):
    """Extracts raw demographics for specialized non-linear parsing."""
    age = load_age(data)
    bmi = load_bmi(data)
    
    sex = load_sex(data, standardize=True)
    sex_val = 0 if sex == 'Female' else (1 if sex == 'Male' else 2)
    
    race = load_race(data, standardize=True)
    race_map = {'Asian': 0, 'Black': 1, 'Others': 2, 'Unavailable': 3, 'White': 4}
    race_val = race_map.get(race, 2)
    
    return np.array([age, bmi, sex_val, race_val], dtype=np.float32)

def parse_waveform_modality(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    """Extracts signal dynamics from the raw waveforms."""
    if not physiological_data:
        return np.full(14, np.nan, dtype=np.float32)
        
    original_labels = list(physiological_data.keys())
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    processed_channels = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data

    # Focus on the two highest-yield signal sources for cognitive tracking
    leads_to_check = [
        ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'], # EEG
        ['spo2', 'sao2']                       # Oximetry
    ]
    
    wave_feats = []
    for candidates in leads_to_check:
        sig = None
        for candidate in candidates:
            if candidate in processed_channels and processed_channels[candidate] is not None:
                sig = processed_channels[candidate]
                break
        
        if sig is not None and len(sig) > 5:
            # Native statistical features
            activity = float(np.var(sig))
            mav = float(np.mean(np.abs(sig)))
            rms = float(np.sqrt(np.mean(sig**2)))
            zcr = float(np.mean(np.diff(np.sign(sig)) != 0))
            
            # Hjorth Mobility parameter
            diff_sig = np.diff(sig)
            mobility = float(np.sqrt(np.var(diff_sig) / activity)) if activity > 0 else 0.0
            
            # Hjorth Complexity parameter
            diff2_sig = np.diff(diff_sig)
            var_d2 = np.var(diff2_sig)
            var_d1 = np.var(diff_sig)
            complexity = float((np.sqrt(var_d2 / var_d1) / mobility)) if (var_d1 > 0 and mobility > 0) else 0.0
            
            wave_feats.extend([activity, mav, rms, zcr, mobility, complexity, np.std(sig)])
        else:
            wave_feats.extend([np.nan] * 7)
            
    return np.array(wave_feats, dtype=np.float32)

def parse_caisr_modality(algo_data):
    """Extracts event densities and architecture from CAISR annotations."""
    if not algo_data:
        return np.full(12, np.nan, dtype=np.float32)

    features = []
    total_hours = len(algo_data.get('resp_caisr', [])) / 3600.0
    
    def count_discrete_events(key):
        if key not in algo_data or total_hours <= 0:
            return np.nan
        sig = (algo_data[key].astype(float) > 0).astype(int)
        diff = np.diff(sig, prepend=0)
        return float(np.count_nonzero(diff == 1) / total_hours)
    
    features.extend([
        count_discrete_events('resp_caisr'),
        count_discrete_events('arousal_caisr'),
        count_discrete_events('limb_caisr')
    ])

    stages = algo_data.get('stage_caisr', np.array([]))
    valid_stages = stages[stages < 9.0]
    
    if len(valid_stages) > 0:
        features.extend([
            float(np.mean(valid_stages == 5)), # Wake
            float(np.mean(valid_stages == 3)), # N1
            float(np.mean(valid_stages == 2)), # N2
            float(np.mean(valid_stages == 1)), # N3
            float(np.mean(valid_stages == 4)), # REM
            float(np.mean((valid_stages >= 1) & (valid_stages <= 4))) # Efficiency
        ])
    else:
        features.extend([np.nan] * 6)

    features.extend([
        float(np.mean(algo_data.get('caisr_prob_w', [np.nan]))),
        float(np.mean(algo_data.get('caisr_prob_n3', [np.nan]))),
        float(np.mean(algo_data.get('caisr_prob_arous', [np.nan])))
    ])
    
    return np.array(features, dtype=np.float32)

# =========================================================================
# REQUIRED TRAIN FUNCTION
# =========================================================================
def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    if verbose:
        print('Compiling Challenge data registries...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No diagnostic records found.')

    X_demo, X_wave, X_caisr, y = [], [], [], []

    for i in range(num_records):
        try:
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id    = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            # Extract Label
            diagnosis_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            label = load_diagnoses(diagnosis_file, patient_id)
            if label not in [0, 1]:
                continue

            # Process Modality 1: Demographics
            p_data = load_demographics(patient_data_file, patient_id, session_id)
            demo_vector = parse_demographics_modality(p_data)

            # Process Modality 2: Waveforms
            phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
            if os.path.exists(phys_file):
                phys_data, phys_fs = load_signal_data(phys_file)
                wave_vector = parse_waveform_modality(phys_data, phys_fs, csv_path=csv_path)
            else:
                wave_vector = np.full(14, np.nan, dtype=np.float32)

            # Process Modality 3: CAISR annotations
            algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
            if os.path.exists(algo_file):
                algo_data, _ = load_signal_data(algo_file)
                caisr_vector = parse_caisr_modality(algo_data)
            else:
                caisr_vector = np.full(12, np.nan, dtype=np.float32)

            X_demo.append(demo_vector)
            X_wave.append(wave_vector)
            X_caisr.append(caisr_vector)
            y.append(label)

        except Exception:
            continue

    # Convert to standard arrays
    X_demo = np.array(X_demo, dtype=np.float32)
    X_wave = np.array(X_wave, dtype=np.float32)
    X_caisr = np.array(X_caisr, dtype=np.float32)
    y = np.array(y, dtype=bool)

    if verbose:
        print('Fitting specialized sub-models...')

    # Model 1 Pipeline: Demographic Non-linear Engine
    demo_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(max_iter=120, max_depth=4, class_weight='balanced', random_state=42))
    ])
    demo_pipeline.fit(X_demo, y)

    # Model 2 Pipeline: Structural Waveform Random Forest
    wave_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42))
    ])
    wave_pipeline.fit(X_wave, y)

    # Model 3 Pipeline: CAISR Architecture Tracker
    caisr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(max_iter=100, max_depth=4, class_weight='balanced', random_state=42))
    ])
    caisr_pipeline.fit(X_caisr, y)

    # Save Bundle Dictionary
    os.makedirs(model_folder, exist_ok=True)
    bundle = {
        'demo_model': demo_pipeline,
        'wave_model': wave_pipeline,
        'caisr_model': caisr_pipeline
    }
    joblib.dump(bundle, os.path.join(model_folder, 'model.sav'), protocol=0)
    
    if verbose:
        print('Multi-model blueprint successfully compiled.')

# =========================================================================
# REQUIRED INFERENCE FUNCTIONS
# =========================================================================
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(model_filename)

def run_model(model, record, data_folder, verbose):
    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # Ingest Modality 1: Demographics
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(patient_data_file, patient_id, session_id)
    demo_vector = parse_demographics_modality(patient_data).reshape(1, -1)

    # Ingest Modality 2: Waveforms
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(phys_file):
        phys_data, phys_fs = load_signal_data(phys_file)
        wave_vector = parse_waveform_modality(phys_data, phys_fs).reshape(1, -1)
    else:
        wave_vector = np.full((1, 14), np.nan, dtype=np.float32)

    # Ingest Modality 3: CAISR
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        caisr_vector = parse_caisr_modality(algo_data).reshape(1, -1)
    else:
        caisr_vector = np.full((1, 12), np.nan, dtype=np.float32)

    # Generate probabilities from individual models
    p_demo = model['demo_model'].predict_proba(demo_vector)[0][1]
    p_wave = model['wave_model'].predict_proba(wave_vector)[0][1]
    p_caisr = model['caisr_model'].predict_proba(caisr_vector)[0][1]

    # Weighted Ensemble Fusion
    probability_output = (DEMO_WEIGHT * p_demo) + (WAVE_WEIGHT * p_wave) + (CAISR_WEIGHT * p_caisr)
    binary_output = bool(probability_output >= GLOBAL_DECISION_THRESHOLD)

    return binary_output, float(probability_output)
