#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

"""
PhysioNet Challenge 2026- V2 with SleepFM Transfer Learning
Targets: Age-Conditioned AUROC + Prevalence Reward

Integrates SleepFM foundation model embeddings with existing hand-crafted features.
SleepFM: A multimodal sleep foundation model trained on 585,000+ hours of PSG data
from 65,000+ participants (Thapa et al., Nature Medicine 2026).
Pretraining data: Stanford Sleep Clinic, BioSerenity, MESA, MrOS.
License: CC BY-NC 4.0
"""

import os
import sys
import warnings
import json
import tempfile
import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm
import joblib

# Core ML
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

# LightGBM (best tree model from LOSO)
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# PyTorch for SleepFM
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
from helper_code import *


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model selection strategy:
# "auto"   = internal LOSO-style CV selects best model per-site
# "logistic" = force Logistic Regression (C=0.5, best on I0002)
# "lightgbm" = force LightGBM (best tree on I0006)
# "stack"    = force Linear Stack (best overall s_C: 0.7511)
# "mega"     = average ensemble of logistic + lightgbm + stack
FINAL_MODEL = "auto"

# number of CV folds for internal model selection
N_CV_FOLDS = 5

# window names for temporal feature extraction
WINDOWS = ['early', 'mid', 'late']

# EEG spectral metrics
EEG_METRICS = ['delta','theta','alpha','sigma','beta','alpha_theta',
               'theta_beta','slowing','delta_sigma','entropy','sef50','sef90']

# SleepFM configuration
SLEEPFM_BASE_PATH = os.path.join(os.path.dirname(__file__), 'sleepfm')
SLEEPFM_MODEL_PATH = os.path.join(SLEEPFM_BASE_PATH, 'checkpoints', 'model_base')
SLEEPFM_CONFIG_PATH = os.path.join(SLEEPFM_MODEL_PATH, 'config.json')
SLEEPFM_CHANNEL_GROUPS_PATH = os.path.join(SLEEPFM_BASE_PATH, 'configs', 'channel_groups_challenge.json')

# =============================================================================
# SLEEPFM INTEGRATION
# =============================================================================

class SleepFMFeatureExtractor:
    """
    Wrapper around SleepFM for extracting embeddings from challenge EDF files.
    Handles EDF -> HDF5 conversion, embedding generation, and temporal aggregation.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_config = None
        self.channel_groups = None
        self._load_model()

    def _load_model(self):
        """Load the frozen SleepFM base model."""
        try:
            if not os.path.exists(SLEEPFM_CONFIG_PATH):
                if self.verbose:
                    print(f"SleepFM config not found at {SLEEPFM_CONFIG_PATH}")
                return

            with open(SLEEPFM_CONFIG_PATH, 'r') as f:
                self.model_config = json.load(f)

            if not os.path.exists(SLEEPFM_CHANNEL_GROUPS_PATH):
                if self.verbose:
                    print(f"Channel groups not found at {SLEEPFM_CHANNEL_GROUPS_PATH}")
                return

            # Import SleepFM model architecture
            repo_root = os.path.dirname(__file__)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)

            try:
                from sleepfm.models.models import SetTransformer
                model_class = SetTransformer
            except ImportError as e:
                if self.verbose:
                    print(f"Could not import SleepFM model: {e}")
                self.model = None
                return

            # Load config parameters
            in_channels = self.model_config.get('in_channels', 1)
            patch_size = self.model_config.get('patch_size', 640)
            embed_dim = self.model_config.get('embed_dim', 128)
            num_heads = self.model_config.get('num_heads', 8)
            num_layers = self.model_config.get('num_layers', 6)
            pooling_head = self.model_config.get('pooling_head', 8)
            dropout = self.model_config.get('dropout', 0.0)

            self.model = model_class(
                in_channels=in_channels,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                pooling_head=pooling_head,
                dropout=dropout,
                max_seq_length=128
            )

            # Load weights
            weights_path = os.path.join(SLEEPFM_MODEL_PATH, 'best.pt')
            if not os.path.exists(weights_path):
                if self.verbose:
                    print(f"SleepFM weights not found at {weights_path}")
                self.model = None
                return

            checkpoint = torch.load(weights_path, map_location=self.device)
            state_dict = checkpoint.get('state_dict', checkpoint)

            # Handle DataParallel prefix
            if len(state_dict) > 0 and next(iter(state_dict)).startswith('module.'):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()

            if self.verbose:
                print(f"SleepFM model loaded on {self.device}")

        except Exception as e:
            if self.verbose:
                print(f"WARNING: Could not load SleepFM model: {e}")
            self.model = None

    def _edf_to_hdf5(self, edf_path, temp_dir):
        """Convert EDF to SleepFM-compatible HDF5 format."""
        try:
            import mne
            import h5py

            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # Only resample if necessary to avoid long waits
            if abs(raw.info['sfreq'] - 128.0) > 0.5:
                raw.resample(128.0)

            subject_id = os.path.splitext(os.path.basename(edf_path))[0]
            hdf5_path = os.path.join(temp_dir, f"{subject_id}_psg.hdf5")

            with h5py.File(hdf5_path, 'w') as f:
                # Store signals by modality based on channel_groups
                for mod, channel_list in self.channel_groups.items():
                    mod_group = f.create_group(mod)
                    matched_channels = []
                    matched_data = []

                    for ch_name in raw.ch_names:
                        ch_lower = ch_name.lower().replace(' ', '_').replace('-', '_')
                        for target in channel_list:
                            target_lower = target.lower().replace('-', '_')
                            if target_lower in ch_lower or ch_lower in target_lower:
                                data = raw.get_data(picks=ch_name)[0]
                                matched_channels.append(ch_name)
                                matched_data.append(data)
                                break

                    if len(matched_data) > 0:
                        # Stack channels: (C, T)
                        data_array = np.stack(matched_data, axis=0)
                        mod_group.create_dataset('data', data=data_array)
                        mod_group.create_dataset('channels', data=np.array(matched_channels, dtype='S'))
                        mod_group.attrs['fs'] = 128.0

            return hdf5_path

        except Exception as e:
            if self.verbose:
                print(f"EDF to HDF5 conversion failed: {e}")
            return None

    def _generate_embeddings(self, hdf5_path):
        """Generate SleepFM embeddings from HDF5 file."""
        if self.model is None:
            return None

        try:
            import h5py

            with h5py.File(hdf5_path, 'r') as f:
                all_embeddings = {}

                for mod in self.channel_groups.keys():
                    if mod not in f:
                        all_embeddings[mod] = None
                        continue

                    mod_group = f[mod]
                    if 'data' not in mod_group:
                        all_embeddings[mod] = None
                        continue

                    data = mod_group['data'][:]  # (C, T)
                    C, T = data.shape

                    patch_size = 640
                    n_patches = T // patch_size
                    if n_patches == 0:
                        all_embeddings[mod] = None
                        continue

                    # Truncate to multiple of patch_size
                    data = data[:, :n_patches * patch_size]

                    # Process in chunks to respect max_seq_length and avoid OOM
                    max_patches_per_chunk = 128
                    chunk_embeddings = []

                    for chunk_start in range(0, n_patches, max_patches_per_chunk):
                        chunk_end = min(chunk_start + max_patches_per_chunk, n_patches)
                        chunk_len = (chunk_end - chunk_start) * patch_size

                        chunk_data = data[:, chunk_start*patch_size:chunk_end*patch_size]
                        x = torch.from_numpy(chunk_data).float().to(self.device)
                        x = x.unsqueeze(0)  # (1, C, chunk_len)
                        mask = torch.ones(1, C, dtype=torch.bool).to(self.device)

                        with torch.no_grad():
                            pooled, tokens = self.model(x, mask)

                        # tokens: (1, num_patches_in_chunk, embed_dim)
                        chunk_embeddings.append(tokens.cpu().numpy())

                    # Concatenate along sequence dimension -> (1, n_patches, embed_dim)
                    if chunk_embeddings:
                        all_embeddings[mod] = np.concatenate(chunk_embeddings, axis=1)
                    else:
                        all_embeddings[mod] = None

                return all_embeddings

        except Exception as e:
            if self.verbose:
                print(f"Embedding generation failed: {e}")
            return None

    def extract_features(self, edf_path):
        """
        Main entry point: EDF file -> SleepFM feature vector.
        Returns dict of features or None if failed.
        """
        if self.model is None:
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            hdf5_path = self._edf_to_hdf5(edf_path, temp_dir)
            if hdf5_path is None:
                return None

            embeddings = self._generate_embeddings(hdf5_path)
            if embeddings is None:
                return None

            features = {}
            for mod in self.channel_groups.keys():
                emb = embeddings.get(mod)
                if emb is not None and emb.size > 0:
                    # emb shape: (1, S, E) where E=embed_dim
                    emb_mod = emb[0]  # (S, E)

                    features[f'sleepfm_{mod}_mean'] = float(np.mean(emb_mod))
                    features[f'sleepfm_{mod}_std'] = float(np.std(emb_mod))
                    features[f'sleepfm_{mod}_max'] = float(np.max(emb_mod))
                    features[f'sleepfm_{mod}_min'] = float(np.min(emb_mod))
                    features[f'sleepfm_{mod}_median'] = float(np.median(emb_mod))

                    # Per-dimension statistics (first 8 dims)
                    n_dims = min(8, emb_mod.shape[1])
                    for i in range(n_dims):
                        features[f'sleepfm_{mod}_d{i}_mean'] = float(np.mean(emb_mod[:, i]))
                        features[f'sleepfm_{mod}_d{i}_std'] = float(np.std(emb_mod[:, i]))
                        features[f'sleepfm_{mod}_d{i}_max'] = float(np.max(emb_mod[:, i]))
                        features[f'sleepfm_{mod}_d{i}_min'] = float(np.min(emb_mod[:, i]))

                    # Temporal dynamics
                    if emb_mod.shape[0] > 1:
                        features[f'sleepfm_{mod}_temporal_std'] = float(np.std(np.mean(emb_mod, axis=1)))
                        features[f'sleepfm_{mod}_temporal_range'] = float(
                            np.max(np.mean(emb_mod, axis=1)) - np.min(np.mean(emb_mod, axis=1))
                        )
                    else:
                        features[f'sleepfm_{mod}_temporal_std'] = 0.0
                        features[f'sleepfm_{mod}_temporal_range'] = 0.0
                else:
                    features[f'sleepfm_{mod}_mean'] = np.nan
                    features[f'sleepfm_{mod}_std'] = np.nan
                    features[f'sleepfm_{mod}_max'] = np.nan
                    features[f'sleepfm_{mod}_min'] = np.nan
                    features[f'sleepfm_{mod}_median'] = np.nan
                    for i in range(8):
                        features[f'sleepfm_{mod}_d{i}_mean'] = np.nan
                        features[f'sleepfm_{mod}_d{i}_std'] = np.nan
                        features[f'sleepfm_{mod}_d{i}_max'] = np.nan
                        features[f'sleepfm_{mod}_d{i}_min'] = np.nan
                    features[f'sleepfm_{mod}_temporal_std'] = np.nan
                    features[f'sleepfm_{mod}_temporal_range'] = np.nan

            return features


# =============================================================================
# REQUIRED FUNCTIONS (DO NOT CHANGE SIGNATURES)
# =============================================================================

def train_model(data_folder, model_folder, verbose):
    """
    Train model on official challenge data.
    Uses site-aware LOSO CV for model selection to match hidden test conditions.
    Integrates SleepFM embeddings with hand-crafted features.
    """
    if verbose:
        print('Finding Challenge data...')

    # Initialize SleepFM feature extractor
    sleepfm_extractor = SleepFMFeatureExtractor(verbose=verbose)
    use_sleepfm = (sleepfm_extractor.model is not None)

    if verbose:
        if use_sleepfm:
            print('SleepFM model loaded successfully.')
        else:
            print('SleepFM model NOT loaded. Using hand-crafted features only.')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data provided.')

    if verbose:
        print(f'Found {num_records} records. Extracting features...')

    # PHASE 1: Extract features from all records
    all_features, all_labels, all_ages, all_sites = [], [], [], []

    pbar = tqdm(range(num_records), desc="Extract", unit="rec", disable=not verbose)
    for i in pbar:
        try:
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({"id": patient_id[:20]})

            patient_data = load_demographics(patient_data_file, patient_id, session_id)
            feats = extract_all_features(data_folder, patient_id, site_id, session_id, patient_data)

            # Extract SleepFM features
            if use_sleepfm:
                physio_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                                           site_id, f"{patient_id}_ses-{session_id}.edf")
                if os.path.exists(physio_path):
                    sleepfm_feats = sleepfm_extractor.extract_features(physio_path)
                    if sleepfm_feats:
                        feats.update(sleepfm_feats)

            if feats is not None:
                label = load_diagnoses(patient_data_file, patient_id)
                age = load_age(patient_data)
                if not np.isnan(age):
                    all_features.append(feats)
                    all_labels.append(label)
                    all_ages.append(age)
                    all_sites.append(site_id)

        except Exception as e:
            if verbose:
                tqdm.write(f"  Skip {patient_id}: {e}")
            continue

    pbar.close()

    if len(all_features) == 0:
        raise ValueError("No valid features extracted.")

    feature_names = list(all_features[0].keys())
    df = pd.DataFrame(all_features)
    df['label'] = all_labels
    df['age'] = all_ages
    df['site'] = all_sites

    if verbose:
        print(f"Extracted {len(df)} records, {len(feature_names)} raw features.")
        print(f"Prevalence: {np.mean(all_labels):.3f} | Sites: {df['site'].nunique()}")

    # PHASE 2: Poison filtering (site-directionality tolerance)
    if verbose:
        print('Filtering site-poisonous features...')

    sites = df['site'].unique()
    poison_features = []

    for col in feature_names:
        if col.startswith('inter_'):
            continue
        corrs = []
        for site in sites:
            sub = df[df['site'] == site].dropna(subset=[col, 'label'])
            if len(sub) > 10 and sub['label'].nunique() > 1:
                try:
                    r, _ = scipy.stats.pointbiserialr(sub['label'], sub[col])
                    corrs.append(r)
                except:
                    pass
        valid = [c for c in corrs if not np.isnan(c)]
        if len(valid) >= 2:
            cmax, cmin = max(valid), min(valid)
            if cmax > 0.05 and cmin < -0.05 and (cmax - cmin) > 0.15:
                poison_features.append(col)

    kept = [f for f in feature_names if f not in poison_features]
    if verbose:
        print(f"  Kept {len(kept)} | Dropped {len(poison_features)} poison")

    # PHASE 3: Build interaction features
    interactions = [
        ('resp_caisr_ahi', 'physio_late_spo2_drop', 'inter_AHI_SpO2'),
        ('caisr_prob_w_mean', 'physio_early_eeg_delta', 'inter_WASO_SWA'),
        ('caisr_prob_r_mean', 'physio_late_emg_rms', 'inter_REM_EMG_Atonia'),
        ('physio_late_ecg_hrv', 'physio_late_resp_effort', 'inter_HRV_RespEffort'),
        ('physio_late_eeg_theta_beta', 'physio_late_resp_freq', 'inter_ThetaBeta_RespFreq'),
        ('physio_mid_eeg_delta_sigma', 'physio_mid_spo2_drop', 'inter_mid_DeltaSigma_SpO2'),
        ('physio_mid_eeg_theta_beta', 'physio_mid_resp_effort', 'inter_mid_ThetaBeta_Effort'),
    ]
    for f1, f2, name in interactions:
        if f1 in kept and f2 in kept:
            df[name] = df[f1] * df[f2]
            kept.append(name)

    # PHASE 4: Age residualization
    if verbose:
        print('Age-residualizing features...')

    resid_cols = []
    age_resid_models = {}
    for col in kept:
        sub = df.dropna(subset=[col, 'age'])
        if len(sub) > 10:
            lr = LinearRegression()
            lr.fit(sub[['age']].values, sub[col].values)
            df[f"{col}_resid"] = df[col] - lr.predict(df[['age']].values)
            resid_cols.append(f"{col}_resid")
            age_resid_models[col] = lr

    # PHASE 5: Impute & scale
    for col in resid_cols:
        med = df[col].median()
        if np.isnan(med):
            med = 0.0
        df[col] = df[col].fillna(med)

    X = df[resid_cols].values
    y = df['label'].values
    ages = df['age'].values
    sites_arr = df['site'].values

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xs = scaler.fit_transform(imputer.fit_transform(X))

    # PHASE 6: Train all candidate models
    if verbose:
        print('Training candidate models...')

    candidates = {}

    candidates['logistic'] = LogisticRegression(
        C=0.5, penalty='l2', solver='liblinear',
        class_weight='balanced', max_iter=1000, random_state=42
    )
    candidates['logistic'].fit(Xs, y)

    if lgb is not None:
        candidates['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, n_jobs=1, verbose=-1,
            class_weight='balanced'
        )
        candidates['lightgbm'].fit(Xs, y)

    candidates['stack'] = StackingClassifier(
        estimators=[
            ('lr1', LogisticRegression(C=0.05, class_weight='balanced',
                                       solver='liblinear', max_iter=1000)),
            ('svm', SVC(probability=True, C=0.1, kernel='linear',
                        class_weight='balanced'))
        ],
        final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
        n_jobs=1
    )
    candidates['stack'].fit(Xs, y)

    estimators_for_mega = [('logistic', candidates['logistic'])]
    if lgb is not None:
        estimators_for_mega.append(('lightgbm', candidates['lightgbm']))
    estimators_for_mega.append(('stack', candidates['stack']))

    candidates['mega'] = VotingClassifier(
        estimators=estimators_for_mega,
        voting='soft',
        n_jobs=1
    )
    candidates['mega'].fit(Xs, y)

    # PHASE 7: Model selection via site-aware LOSO CV
    if FINAL_MODEL == "auto":
        if verbose:
            print('Running site-aware LOSO CV for model selection...')

        best_score, best_name = -np.inf, 'logistic'

        for name, model in candidates.items():
            scores = []
            for test_site in sites:
                train_mask = sites_arr != test_site
                test_mask = sites_arr == test_site

                if np.sum(test_mask) < 5 or np.sum(train_mask) < 20:
                    continue
                if len(np.unique(y[test_mask])) < 2:
                    continue

                X_tr, X_val = Xs[train_mask], Xs[test_mask]
                y_tr, y_val = y[train_mask], y[test_mask]
                a_val = ages[test_mask]

                m = _make_fresh_model(name)
                m.fit(X_tr, y_tr)
                p = m.predict_proba(X_val)[:, 1]
                sc = _age_auroc(y_val, p, a_val, delta=2.0)
                if not np.isnan(sc):
                    scores.append(sc)

            avg = np.mean(scores) if scores else 0
            if verbose:
                print(f"  {name:12s}: LOSO age-AUROC = {avg:.4f} (n={len(scores)} folds)")
            if avg > best_score:
                best_score, best_name = avg, name

        selected = candidates[best_name]
        selected_name = best_name
        if verbose:
            print(f"Selected: {best_name} (LOSO age-AUROC = {best_score:.4f})")

    else:
        selected_name = FINAL_MODEL if FINAL_MODEL in candidates else 'logistic'
        selected = candidates[selected_name]
        if verbose:
            print(f"Forced model: {selected_name}")

    # PHASE 8: Save artifact
    os.makedirs(model_folder, exist_ok=True)

    artifact = {
        'model': selected,
        'model_name': selected_name,
        'candidates': candidates,
        'scaler': scaler,
        'imputer': imputer,
        'kept_features': kept,
        'resid_cols': resid_cols,
        'age_resid_models': age_resid_models,
        'interactions': interactions,
        'sites_seen': list(sites),
        'use_sleepfm': use_sleepfm,
        'sleepfm_channel_groups': sleepfm_extractor.channel_groups if use_sleepfm else None,
    }

    joblib.dump(artifact, os.path.join(model_folder, 'model.sav'))
    if verbose:
        print('Training complete. Model saved.')


def load_model(model_folder, verbose):
    """Load trained model artifact."""
    return joblib.load(os.path.join(model_folder, 'model.sav'))


def run_model(model_artifact, record, data_folder, verbose):
    """
    Run trained model on a single record.
    Returns: (binary_prediction, probability)
    """
    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(patient_data_file, patient_id, session_id)

    feats = extract_all_features(data_folder, patient_id, site_id, session_id, patient_data)
    if feats is None:
        return float('nan'), float('nan')

    # Extract SleepFM features if model was trained with them
    if model_artifact.get('use_sleepfm', False):
        sleepfm_extractor = SleepFMFeatureExtractor(verbose=False)
        if sleepfm_extractor.model is not None:
            physio_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                                       site_id, f"{patient_id}_ses-{session_id}.edf")
            if os.path.exists(physio_path):
                sleepfm_feats = sleepfm_extractor.extract_features(physio_path)
                if sleepfm_feats:
                    feats.update(sleepfm_feats)

    df = pd.DataFrame([feats])
    age = load_age(patient_data)

    # Build interaction features
    for f1, f2, name in model_artifact.get('interactions', []):
        if f1 in df.columns and f2 in df.columns:
            df[name] = df[f1] * df[f2]

    # Age residualization
    for col, lr in model_artifact.get('age_resid_models', {}).items():
        if col in df.columns:
            df[f"{col}_resid"] = df[col].values - lr.predict(np.array([[age]]))[0]

    resid_cols = model_artifact['resid_cols']
    for c in resid_cols:
        if c not in df.columns:
            df[c] = float('nan')

    X = df[resid_cols].values
    Xs = model_artifact['scaler'].transform(model_artifact['imputer'].transform(X))

    model = model_artifact['model']
    prob = float(model.predict_proba(Xs)[0, 1])
    binary = int(prob >= 0.5)

    return binary, prob


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_all_features(data_folder, patient_id, site_id, session_id, patient_data):
    """
    Extract all features for a single patient record.
    NOTE: Human annotations are NOT used - they are unavailable in validation/test.
    """
    features = {}

    # 1. Demographic features
    features['age'] = load_age(patient_data)
    sex = load_sex(patient_data, standardize=True)
    features['sex_male'] = 1 if sex == 'Male' else 0
    race = load_race(patient_data, standardize=True)
    features['race_white'] = 1 if race == 'White' else 0
    features['race_black'] = 1 if race == 'Black' else 0
    features['race_asian'] = 1 if race == 'Asian' else 0
    features['race_other'] = 1 if race == 'Others' else 0
    features['bmi'] = load_bmi(patient_data)

    # 2. CAISR algorithmic annotations (available in all sets)
    caisr_path = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                              site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    features.update(_extract_caisr(caisr_path))

    # 3. Physiological signals
    physio_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                               site_id, f"{patient_id}_ses-{session_id}.edf")
    features.update(_extract_physio(physio_path))

    return features


def _extract_caisr(edf_path):
    """Extract features from CAISR algorithmic annotations EDF."""
    out = {
        'stage_caisr_tst': np.nan, 'stage_caisr_se': np.nan,
        'arousal_caisr_rate': np.nan, 'resp_caisr_ahi': np.nan,
        'limb_caisr_rate': np.nan, 'limb_isolated_rate': np.nan,
        'limb_periodic_rate': np.nan,
        'caisr_prob_n3_mean': np.nan, 'caisr_prob_n2_mean': np.nan,
        'caisr_prob_n1_mean': np.nan, 'caisr_prob_r_mean': np.nan,
        'caisr_prob_w_mean': np.nan, 'caisr_prob_arousal_mean': np.nan,
        'stage_transition_rate': np.nan, 'caisr_softmax_entropy': np.nan,
        'high_conf_arousal_rate': np.nan, 'resp_central_ratio': np.nan,
    }
    if not os.path.exists(edf_path):
        return out

    try:
        data_dict, fs_dict = load_signal_data(edf_path)
        if not data_dict:
            return out

        labels = list(data_dict.keys())

        def get(kws):
            for lbl in labels:
                if all(k in lbl for k in kws):
                    return data_dict[lbl]
            return None

        stages = get(['stage'])
        resp = get(['resp'])
        limbs = get(['limb'])
        arousal = next((data_dict[l] for l in labels
                        if 'arousal' in l and 'prob' not in l and 'no-ar' not in l), None)

        pn3 = _sanitize(get(['prob', 'n3']))
        pn2 = _sanitize(get(['prob', 'n2']))
        pn1 = _sanitize(get(['prob', 'n1']))
        pw  = _sanitize(get(['prob', 'w']))
        pr  = _sanitize(next((data_dict[l] for l in labels
                               if 'prob' in l and ('_r' in l or 'prob_r' in l)), None))
        pa  = _sanitize(get(['prob', 'arous']))
        if pa is None:
            pa = _sanitize(get(['prob', 'ar']))

        # Sleep architecture
        if stages is not None and len(stages) > 0:
            tst = (len(stages) * 30) / 3600
            out['stage_caisr_tst'] = tst
            out['stage_caisr_se'] = float(np.sum(np.isin(stages, [1,2,3,4])) / len(stages))
            out['stage_transition_rate'] = float(np.sum(np.diff(stages) != 0) / max(tst, 0.5))

        dh = max(out['stage_caisr_tst'], 0.5)

        # Event rates (per hr)
        if arousal is not None:
            out['arousal_caisr_rate'] = float(_count_events(arousal, [1]) / dh)
        if resp is not None:
            out['resp_caisr_ahi'] = float(_count_events(resp, [1,2,3,4]) / dh)
            tap = _count_events(resp, [2]) + _count_events(resp, [1])
            out['resp_central_ratio'] = float(_count_events(resp, [2]) / tap) if tap > 0 else 0.0
        if limbs is not None:
            out['limb_caisr_rate'] = float(_count_events(limbs, [1,2]) / dh)
            out['limb_isolated_rate'] = float(_count_events(limbs, [1]) / dh)
            out['limb_periodic_rate'] = float(_count_events(limbs, [2]) / dh)

        # Mean probabilities
        if pn3 is not None: out['caisr_prob_n3_mean'] = float(np.mean(pn3))
        if pn2 is not None: out['caisr_prob_n2_mean'] = float(np.mean(pn2))
        if pw  is not None: out['caisr_prob_w_mean']  = float(np.mean(pw))
        if pr  is not None: out['caisr_prob_r_mean']  = float(np.mean(pr))

        # Softmax entropy across sleep stages
        plist = [pn3, pn2, pn1, pr, pw]
        if all(p is not None for p in plist):
            stacked = np.stack(plist, axis=0)
            sv = np.sum(stacked, axis=0, keepdims=True)
            sv[sv == 0] = 1.0
            stacked = stacked / sv
            out['caisr_softmax_entropy'] = float(
                np.mean(-np.sum(stacked * np.log(stacked + 1e-9), axis=0)))

        if pa is not None:
            out['caisr_prob_arousal_mean'] = float(np.mean(pa))
            m = (pa > 0.85).astype(int)
            out['high_conf_arousal_rate'] = float(_count_events(m, [1]) / dh)

    except Exception:
        pass
    return out


def _extract_physio(edf_path):
    """Extract features from physiological signals with early/mid/late windowing."""
    out = {}
    for w in WINDOWS:
        for m in EEG_METRICS:
            out[f'physio_{w}_eeg_{m}'] = np.nan
        out[f'physio_{w}_emg_rms'] = np.nan
        out[f'physio_{w}_ecg_hrv'] = np.nan
        out[f'physio_{w}_resp_freq'] = np.nan
        out[f'physio_{w}_resp_effort'] = np.nan
        out[f'physio_{w}_spo2_drop'] = np.nan

    if not os.path.exists(edf_path):
        return out

    try:
        data_dict, fs_dict = load_signal_data(edf_path)
        if not data_dict:
            return out

        labels = list(data_dict.keys())
        eeg, eeg_fs = _find_sig(labels, data_dict, fs_dict, 'eeg')
        emg, emg_fs = _find_sig(labels, data_dict, fs_dict, 'emg')
        ecg, ecg_fs = _find_sig(labels, data_dict, fs_dict, 'ecg')
        rsp, rsp_fs = _find_sig(labels, data_dict, fs_dict, 'resp_airflow')
        eff, eff_fs = _find_sig(labels, data_dict, fs_dict, 'resp_effort')
        sp2, sp2_fs = _find_sig(labels, data_dict, fs_dict, 'spo2')

        # Estimate recording duration
        dur = 0
        for s, f in [(eeg, eeg_fs), (emg, emg_fs), (ecg, ecg_fs),
                     (rsp, rsp_fs), (eff, eff_fs), (sp2, sp2_fs)]:
            if s is not None:
                dur = max(dur, len(s) / f)
        if dur <= 0:
            return out

        t3 = dur / 3.0
        bounds = {'early': (0, t3), 'mid': (t3, 2*t3), 'late': (2*t3, dur)}

        for stage, (st, en) in bounds.items():
            # EEG spectral features
            if eeg is not None and eeg_fs > 0:
                sl = eeg[int(st*eeg_fs):int(en*eeg_fs)]
                if len(sl) > eeg_fs * 10:
                    sl = (sl - np.nanmean(sl)) / (np.nanstd(sl) + 1e-8)
                    ef = _eeg_spectrum(sl, eeg_fs)
                    for idx, m in enumerate(EEG_METRICS):
                        out[f'physio_{stage}_eeg_{m}'] = ef[idx]

            # EMG RMS
            if emg is not None and emg_fs > 0:
                sl = emg[int(st*emg_fs):int(en*emg_fs)]
                if len(sl) > emg_fs * 10:
                    out[f'physio_{stage}_emg_rms'] = float(
                        np.sqrt(np.mean(np.square(sl - np.mean(sl)))))

            # ECG HRV proxy
            if ecg is not None and ecg_fs > 0:
                sl = ecg[int(st*ecg_fs):int(en*ecg_fs)]
                if len(sl) > ecg_fs * 10:
                    out[f'physio_{stage}_ecg_hrv'] = float(np.var(np.diff(sl)))

            # Respiratory frequency
            if rsp is not None and rsp_fs > 0:
                sl = rsp[int(st*rsp_fs):int(en*rsp_fs)]
                if len(sl) > rsp_fs * 10:
                    out[f'physio_{stage}_resp_freq'] = _resp_spectrum(sl, rsp_fs)[0]

            # Respiratory effort variance
            if eff is not None and eff_fs > 0:
                sl = eff[int(st*eff_fs):int(en*eff_fs)]
                if len(sl) > eff_fs * 10:
                    out[f'physio_{stage}_resp_effort'] = float(np.var(sl))

            # SpO2 drop (95th - 5th percentile)
            if sp2 is not None and sp2_fs > 0:
                sl = sp2[int(st*sp2_fs):int(en*sp2_fs)]
                if len(sl) > sp2_fs * 10:
                    out[f'physio_{stage}_spo2_drop'] = float(
                        np.percentile(sl, 95) - np.percentile(sl, 5))

    except Exception:
        pass
    return out


# =============================================================================
# UTILITIES
# =============================================================================

def _find_sig(labels, data_dict, fs_dict, target):
    """Find signal by type using ordered manifest matching."""
    manifest = {
        'eeg': ['c3-m2','c4-m1','c3','c4','f3-m2','f4-m1','f3','f4',
                'o1-m2','o2-m1','eeg'],
        'emg': ['chin1-chin2','chin','emg.subm','emg','chin1','emg1','chin2','emg2'],
        'ecg': ['ecg','ekg','ecg-la','ecg-v1','ecg i','ecg ii','ecg1'],
        'resp_airflow': ['airflow','flow','thermal','thermistor','nasal_pressure'],
        'resp_effort': ['abd','abdomen','chest','thorax','effort abd','effort tho'],
        'spo2': ['spo2','sao2','osat','o2sat']
    }
    for t in manifest.get(target, []):
        for lbl in labels:
            if t in lbl:
                return data_dict[lbl], fs_dict.get(lbl, 1.0)
    return None, 1.0


def _sanitize(sig):
    """Sanitize probability signal to [0, 1] range."""
    if sig is None or len(sig) == 0:
        return sig
    mn, mx = np.min(sig), np.max(sig)
    if mx > 1.0001 or mn < -0.0001:
        d = mx - mn
        if d > 1e-6:
            sig = (sig - mn) / d
    return np.clip(sig, 0.0, 1.0)


def _count_events(arr, codes):
    """Count contiguous events in a discrete signal."""
    if arr is None or len(arr) == 0:
        return 0
    b = np.isin(arr, codes).astype(int)
    d = np.diff(b)
    return max(np.sum(d == 1) + (1 if b[0] == 1 else 0), 0)


def _eeg_spectrum(signal, fs):
    """Extract 12 EEG spectral features."""
    if signal is None or len(signal) == 0:
        return [np.nan] * 12
    try:
        n = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        ti = (freqs >= 0.5) & (freqs <= 30)
        tp = np.sum(fft_vals[ti])
        if tp == 0:
            return [np.nan] * 12

        delta = np.sum(fft_vals[(freqs >= 0.5) & (freqs < 4)]) / tp
        theta = np.sum(fft_vals[(freqs >= 4) & (freqs < 8)]) / tp
        alpha = np.sum(fft_vals[(freqs >= 8) & (freqs < 12)]) / tp
        sigma = np.sum(fft_vals[(freqs >= 12) & (freqs < 15)]) / tp
        beta  = np.sum(fft_vals[(freqs >= 15) & (freqs <= 30)]) / tp

        at = alpha / (theta + 1e-8)
        tb = theta / (beta + 1e-8)
        sl = (delta + theta) / (alpha + beta + 1e-8)
        ds = delta / (sigma + 1e-8)

        pn = fft_vals[ti] / tp
        ent = scipy.stats.entropy(pn, base=2)
        cp = np.cumsum(fft_vals[ti])
        sef50 = freqs[ti][np.where(cp >= 0.50 * tp)[0][0]]
        sef90 = freqs[ti][np.where(cp >= 0.90 * tp)[0][0]]

        return [delta, theta, alpha, sigma, beta, at, tb, sl, ds, ent, sef50, sef90]
    except Exception:
        return [np.nan] * 12


def _resp_spectrum(signal, fs):
    """Extract respiratory peak frequency and effort variance."""
    if signal is None or len(signal) == 0:
        return [np.nan, np.nan]
    try:
        n = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        ri = (freqs >= 0.1) & (freqs <= 0.5)
        if np.sum(ri) == 0:
            return [np.nan, np.nan]
        pf = freqs[ri][np.argmax(fft_vals[ri])]
        ev = np.var(signal)
        return [float(pf), float(ev)]
    except Exception:
        return [np.nan, np.nan]


def _age_auroc(y_true, y_prob, ages, delta=2.0):
    """
    Compute age-conditioned AUROC (official challenge metric).
    Only compares positive/negative pairs within delta years.
    """
    y_true, y_prob, ages = np.asarray(y_true), np.asarray(y_prob), np.asarray(ages)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    c, d, t = 0, 0, 0
    for i in pos:
        vn = neg[np.abs(ages[neg] - ages[i]) <= delta]
        for j in vn:
            if y_prob[i] > y_prob[j]: c += 1
            elif y_prob[i] < y_prob[j]: d += 1
            else: t += 1
    tot = c + d + t
    return (c + 0.5*t) / tot if tot > 0 else np.nan


def _make_fresh_model(name):
    """Create a fresh unfitted model for CV evaluation."""
    if name == 'logistic':
        return LogisticRegression(C=0.5, penalty='l2', solver='liblinear',
                                   class_weight='balanced', max_iter=1000, random_state=42)
    elif name == 'lightgbm' and lgb is not None:
        return lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   random_state=42, n_jobs=1, verbose=-1,
                                   class_weight='balanced')
    elif name == 'stack':
        return StackingClassifier(
            estimators=[
                ('lr1', LogisticRegression(C=0.05, class_weight='balanced',
                                            solver='liblinear', max_iter=1000)),
                ('svm', SVC(probability=True, C=0.1, kernel='linear',
                           class_weight='balanced'))
            ],
            final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
            n_jobs=1
        )
    elif name == 'mega':
        estimators = [('logistic', LogisticRegression(C=0.5, penalty='l2', solver='liblinear',
                                                       class_weight='balanced', max_iter=1000, random_state=42))]
        if lgb is not None:
            estimators.append(('lightgbm', lgb.LGBMClassifier(n_estimators=200, max_depth=4,
                                                               learning_rate=0.05, random_state=42,
                                                               n_jobs=1, verbose=-1, class_weight='balanced')))
        estimators.append(('stack', StackingClassifier(
            estimators=[
                ('lr1', LogisticRegression(C=0.05, class_weight='balanced', solver='liblinear', max_iter=1000)),
                ('svm', SVC(probability=True, C=0.1, kernel='linear', class_weight='balanced'))
            ],
            final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
            n_jobs=1
        )))
        return VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
    else:
        raise ValueError(f"Unknown model name: {name}")




# #!/usr/bin/env python

# # Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# # change or remove non-required functions, and add your own functions.

# """
# PhysioNet Challenge 2026- V2 with SleepFM Transfer Learning
# Targets: Age-Conditioned AUROC + Prevalence Reward

# Integrates SleepFM foundation model embeddings with existing hand-crafted features.
# SleepFM: A multimodal sleep foundation model trained on 585,000+ hours of PSG data
# from 65,000+ participants (Thapa et al., Nature Medicine 2026).
# Pretraining data: Stanford Sleep Clinic, BioSerenity, MESA, MrOS.
# License: CC BY-NC 4.0
# """

# import os
# import sys
# import warnings
# import json
# import tempfile
# import numpy as np
# import pandas as pd
# import scipy.stats
# from tqdm import tqdm
# import joblib

# # Core ML
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import StackingClassifier, VotingClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold

# # LightGBM (best tree model from LOSO)
# try:
#     import lightgbm as lgb
# except ImportError:
#     lgb = None

# # PyTorch for SleepFM
# import torch
# import torch.nn as nn

# warnings.filterwarnings("ignore")
# from helper_code import *


# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# # Model selection strategy:
# # "auto"   = internal LOSO-style CV selects best model per-site
# # "logistic" = force Logistic Regression (C=0.5, best on I0002)
# # "lightgbm" = force LightGBM (best tree on I0006)
# # "stack"    = force Linear Stack (best overall s_C: 0.7511)
# # "mega"     = average ensemble of logistic + lightgbm + stack
# FINAL_MODEL = "auto"

# # number of CV folds for internal model selection
# N_CV_FOLDS = 5

# # window names for temporal feature extraction
# WINDOWS = ['early', 'mid', 'late']

# # EEG spectral metrics
# EEG_METRICS = ['delta','theta','alpha','sigma','beta','alpha_theta',
#                'theta_beta','slowing','delta_sigma','entropy','sef50','sef90']

# # SleepFM configuration
# SLEEPFM_BASE_PATH = os.path.join(os.path.dirname(__file__), 'sleepfm')
# SLEEPFM_MODEL_PATH = os.path.join(SLEEPFM_BASE_PATH, 'checkpoints', 'model_base')
# SLEEPFM_CONFIG_PATH = os.path.join(SLEEPFM_MODEL_PATH, 'config.json')
# SLEEPFM_CHANNEL_GROUPS_PATH = os.path.join(SLEEPFM_BASE_PATH, 'configs', 'channel_groups_challenge.json')

# # =============================================================================
# # SLEEPFM INTEGRATION
# # =============================================================================

# class SleepFMFeatureExtractor:
#     """
#     Wrapper around SleepFM for extracting embeddings from challenge EDF files.
#     Handles EDF -> model input -> embedding generation -> temporal aggregation.

#     Model input shape: (B, C, S, 640) where:
#       B = batch size (1 for single patient)
#       C = number of channels for this modality
#       S = number of 5-second windows in the recording
#       640 = 5 seconds at 128Hz

#     Mask shape: (B, C) boolean, True for valid channels

#     Forward returns: (pooled_emb, per_timestep_emb) where:
#       pooled_emb = (B, 128) - patient-level embedding per modality
#       per_timestep_emb = (B, S, 128) - per-window embeddings
#     """

#     def __init__(self, verbose=False):
#         self.verbose = verbose
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = None
#         self.model_config = None
#         self.channel_groups = None
#         self.patch_size = 640
#         self.embed_dim = 128
#         self._load_model()

#     def _load_model(self):
#         """Load the frozen SleepFM base model."""
#         try:
#             if not os.path.exists(SLEEPFM_CONFIG_PATH):
#                 if self.verbose:
#                     print(f"SleepFM config not found at {SLEEPFM_CONFIG_PATH}")
#                 return

#             with open(SLEEPFM_CONFIG_PATH, 'r') as f:
#                 self.model_config = json.load(f)
            
            
#             # Import SleepFM model architecture
#             # The model class name is specified in config['model']
#             model_name = self.model_config.get('model', 'SetTransformer')

#             if not os.path.exists(SLEEPFM_CHANNEL_GROUPS_PATH):
#                 if self.verbose:
#                     print(f"Channel groups not found at {SLEEPFM_CHANNEL_GROUPS_PATH}")
#                 return

#             with open(SLEEPFM_CHANNEL_GROUPS_PATH, 'r') as f:
#                 self.channel_groups = json.load(f)

#             # Add repo root to sys.path so 'sleepfm' package is discoverable
#             repo_root = os.path.dirname(__file__)
#             if repo_root not in sys.path:
#                 sys.path.insert(0, repo_root)

#             # Import the real SleepFM model architecture
#             try:
#                 from sleepfm.models.models import SetTransformer
#                 model_class = SetTransformer
#             except ImportError as e:
#                 if self.verbose:
#                     print(f"Could not import SetTransformer: {e}")
#                 model_class = self._build_model_inline()

#             # Model config values
#             in_channels = self.model_config.get('in_channels', 1)
#             patch_size = self.model_config.get('patch_size', 640)
#             embed_dim = self.model_config.get('embed_dim', 128)
#             num_heads = self.model_config.get('num_heads', 8)
#             num_layers = self.model_config.get('num_layers', 6)
#             pooling_head = self.model_config.get('pooling_head', 8)

#             self.patch_size = patch_size
#             self.embed_dim = embed_dim

#             self.model = model_class(
#                 in_channels=in_channels,
#                 patch_size=patch_size,
#                 embed_dim=embed_dim,
#                 num_heads=num_heads,
#                 num_layers=num_layers,
#                 pooling_head=pooling_head,
#                 dropout=0.0
#             )

#             # Load weights
#             weights_path = os.path.join(SLEEPFM_MODEL_PATH, 'best.pt')
#             if not os.path.exists(weights_path):
#                 if self.verbose:
#                     print(f"SleepFM weights not found at {weights_path}")
#                 self.model = None
#                 return

#             checkpoint = torch.load(weights_path, map_location=self.device)
#             state_dict = checkpoint.get('state_dict', checkpoint)

#             # Handle DataParallel prefix
#             if len(state_dict) > 0 and next(iter(state_dict)).startswith('module.'):
#                 state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

#             self.model.load_state_dict(state_dict, strict=False)
#             self.model.to(self.device)
#             self.model.eval()

#             if self.verbose:
#                 print(f"SleepFM model loaded on {self.device}")

#         except Exception as e:
#             if self.verbose:
#                 print(f"WARNING: Could not load SleepFM model: {e}")
#             self.model = None

#     def _build_model_inline(self):
#         """Build minimal SleepFM model inline if imports fail."""
#         class InlineSleepFM(nn.Module):
#             def __init__(self, in_channels, patch_size, embed_dim, num_heads, num_layers, pooling_head, dropout):
#                 super().__init__()
#                 self.in_channels = in_channels
#                 self.patch_size = patch_size
#                 self.embed_dim = embed_dim
#                 self.num_heads = num_heads
#                 self.num_layers = num_layers
#                 self.pooling_head = pooling_head
#                 self.dropout = dropout

#                 # Simple conv tokenizer
#                 self.tokenizer = nn.Sequential(
#                     nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=2),
#                     nn.BatchNorm1d(4),
#                     nn.ELU(),
#                     nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2),
#                     nn.BatchNorm1d(8),
#                     nn.ELU(),
#                     nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
#                     nn.BatchNorm1d(16),
#                     nn.ELU(),
#                     nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
#                     nn.BatchNorm1d(32),
#                     nn.ELU(),
#                     nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
#                     nn.BatchNorm1d(64),
#                     nn.ELU(),
#                     nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
#                     nn.BatchNorm1d(128),
#                     nn.ELU(),
#                     nn.AdaptiveAvgPool1d(1),
#                     nn.Flatten(),
#                     nn.Linear(128, embed_dim)
#                 )

#                 # Temporal transformer
#                 encoder_layer = nn.TransformerEncoderLayer(
#                     d_model=embed_dim, nhead=num_heads,
#                     dim_feedforward=embed_dim*4, dropout=dropout,
#                     batch_first=True
#                 )
#                 self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#             def forward(self, x, mask):
#                 # x: (B, C, S, L)
#                 B, C, S, L = x.shape
#                 # Process each channel independently: (B*C*S, 1, L)
#                 x_flat = x.reshape(B * C * S, 1, L)
#                 tokens = self.tokenizer(x_flat)  # (B*C*S, embed_dim)
#                 # Reshape to (B, C, S, E)
#                 tokens = tokens.reshape(B, C, S, self.embed_dim)
#                 # Average across channels
#                 tokens = tokens.mean(dim=1)  # (B, S, E)
#                 # Temporal transformer
#                 out = self.temporal_transformer(tokens)
#                 # Pool across time
#                 pooled = out.mean(dim=1)  # (B, E)
#                 return pooled, out

#         return InlineSleepFM

#     def _read_edf_channels(self, edf_path):
#         """Read EDF and group channels by SleepFM modality."""
#         try:
#             import mne
#             raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
#             raw.resample(128.0)

#             # Group channels by modality
#             modality_data = {}
#             for mod, channel_list in self.channel_groups.items():
#                 matched_signals = []
#                 for ch_name in raw.ch_names:
#                     ch_lower = ch_name.lower().replace(' ', '_').replace('-', '_')
#                     for target in channel_list:
#                         target_lower = target.lower().replace('-', '_')
#                         if target_lower in ch_lower or ch_lower in target_lower:
#                             data = raw.get_data(picks=ch_name)[0]
#                             matched_signals.append(data)
#                             break

#                 if len(matched_signals) > 0:
#                     # Stack to (C, T)
#                     data_array = np.stack(matched_signals, axis=0)
#                     modality_data[mod] = data_array

#             return modality_data

#         except Exception as e:
#             if self.verbose:
#                 print(f"EDF read failed: {e}")
#             return {}

#     def _segment_into_patches(self, signal, patch_size=640):
#         """Split signal into non-overlapping 5-second patches."""
#         C, T = signal.shape
#         n_patches = T // patch_size
#         if n_patches == 0:
#             return None
#         patches = []
#         for i in range(n_patches):
#             patch = signal[:, i*patch_size:(i+1)*patch_size]
#             patches.append(patch)
#         # Stack: (S, C, L) then transpose to (C, S, L)
#         patches = np.stack(patches, axis=0)  # (S, C, L)
#         patches = patches.transpose(1, 0, 2)  # (C, S, L)
#         return patches

# def _generate_embeddings(self, modality_data):
#     if self.model is None:
#         return None

#     all_embeddings = {}
#     chunk_patches = 60  # 5 minutes = 60 * 640 samples
#     chunk_samples = chunk_patches * self.patch_size

#     for mod in self.channel_groups.keys():
#         if mod not in modality_data:
#             all_embeddings[mod] = None
#             continue

#         signal = modality_data[mod]  # (C, T)
#         C, T = signal.shape

#         # Truncate to multiple of chunk size
#         T_trunc = (T // chunk_samples) * chunk_samples
#         if T_trunc == 0:
#             all_embeddings[mod] = None
#             continue

#         signal = signal[:, :T_trunc]
#         n_chunks = T_trunc // chunk_samples

#         # Process each chunk
#         chunk_embeddings = []
#         for i in range(n_chunks):
#             chunk = signal[:, i*chunk_samples:(i+1)*chunk_samples]
#             x = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)  # (1, C, chunk_samples)
#             mask = torch.ones(1, C, dtype=torch.bool).to(self.device)

#             with torch.no_grad():
#                 pooled, _ = self.model(x, mask)
#             chunk_embeddings.append(pooled.cpu().numpy())

#         # Average across chunks
#         chunk_embeddings = np.stack(chunk_embeddings, axis=0)  # (n_chunks, 1, 128)
#         mean_embedding = chunk_embeddings.mean(axis=0)[0]  # (128,)

#         all_embeddings[mod] = {
#             'pooled': mean_embedding.reshape(1, -1),  # (1, 128)
#             'per_chunk': chunk_embeddings  # (n_chunks, 1, 128)
#         }

#     return all_embeddings

#     def extract_features(self, edf_path):
#         """
#         Main entry point: EDF file -> SleepFM feature vector.
#         Returns dict of features or None if failed.
#         """
#         if self.model is None:
#             return None

#         modality_data = self._read_edf_channels(edf_path)
#         if not modality_data:
#             return None

#         embeddings = self._generate_embeddings(modality_data)
#         if embeddings is None:
#             return None

#         features = {}

#         for mod in self.channel_groups.keys():
#             emb = embeddings.get(mod)
#             if emb is not None:
#                 pooled = emb['pooled'][0]  # (128,)
#                 per_ts = emb['per_timestep'][0]  # (S, 128)

#                 # Pooled embedding stats
#                 features[f'sleepfm_{mod}_pooled_mean'] = float(pooled.mean())
#                 features[f'sleepfm_{mod}_pooled_std'] = float(pooled.std())
#                 features[f'sleepfm_{mod}_pooled_max'] = float(pooled.max())
#                 features[f'sleepfm_{mod}_pooled_min'] = float(pooled.min())

#                 # Per-timestep stats
#                 features[f'sleepfm_{mod}_ts_mean'] = float(per_ts.mean())
#                 features[f'sleepfm_{mod}_ts_std'] = float(per_ts.std())
#                 features[f'sleepfm_{mod}_ts_max'] = float(per_ts.max())
#                 features[f'sleepfm_{mod}_ts_min'] = float(per_ts.min())

#                 # Temporal dynamics
#                 ts_mean_per_window = per_ts.mean(axis=1)  # (S,)
#                 features[f'sleepfm_{mod}_temporal_std'] = float(ts_mean_per_window.std())
#                 features[f'sleepfm_{mod}_temporal_range'] = float(ts_mean_per_window.max() - ts_mean_per_window.min())

#                 # Per-dimension stats (first 8 dims of pooled)
#                 for i in range(min(8, pooled.shape[0])):
#                     features[f'sleepfm_{mod}_d{i}'] = float(pooled[i])
#             else:
#                 # Missing modality - fill with NaN
#                 for prefix in ['pooled_mean', 'pooled_std', 'pooled_max', 'pooled_min',
#                                'ts_mean', 'ts_std', 'ts_max', 'ts_min',
#                                'temporal_std', 'temporal_range']:
#                     features[f'sleepfm_{mod}_{prefix}'] = np.nan
#                 for i in range(8):
#                     features[f'sleepfm_{mod}_d{i}'] = np.nan

#         return features


# # =============================================================================
# # REQUIRED FUNCTIONS (DO NOT CHANGE SIGNATURES)
# # =============================================================================

# def train_model(data_folder, model_folder, verbose):
#     """
#     Train model on official challenge data.
#     Uses site-aware LOSO CV for model selection to match hidden test conditions.
#     Integrates SleepFM embeddings with hand-crafted features.
#     """
#     if verbose:
#         print('Finding Challenge data...')

#     # Initialize SleepFM feature extractor
#     sleepfm_extractor = SleepFMFeatureExtractor(verbose=verbose)
#     use_sleepfm = (sleepfm_extractor.model is not None)

#     if verbose:
#         if use_sleepfm:
#             print('SleepFM model loaded successfully.')
#         else:
#             print('SleepFM model NOT loaded. Using hand-crafted features only.')

#     patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
#     patient_metadata_list = find_patients(patient_data_file)
#     num_records = len(patient_metadata_list)

#     if num_records == 0:
#         raise FileNotFoundError('No data provided.')

#     if verbose:
#         print(f'Found {num_records} records. Extracting features...')

#     # PHASE 1: Extract features from all records
#     all_features, all_labels, all_ages, all_sites = [], [], [], []

#     pbar = tqdm(range(num_records), desc="Extract", unit="rec", disable=not verbose)
#     for i in pbar:
#         try:
#             record = patient_metadata_list[i]
#             patient_id = record[HEADERS['bids_folder']]
#             site_id = record[HEADERS['site_id']]
#             session_id = record[HEADERS['session_id']]

#             if verbose:
#                 pbar.set_postfix({"id": patient_id[:20]})

#             patient_data = load_demographics(patient_data_file, patient_id, session_id)
#             feats = extract_all_features(data_folder, patient_id, site_id, session_id, patient_data)

#             # Extract SleepFM features
#             if use_sleepfm:
#                 physio_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
#                                            site_id, f"{patient_id}_ses-{session_id}.edf")
#                 if os.path.exists(physio_path):
#                     sleepfm_feats = sleepfm_extractor.extract_features(physio_path)
#                     if sleepfm_feats:
#                         feats.update(sleepfm_feats)

#             if feats is not None:
#                 label = load_diagnoses(patient_data_file, patient_id)
#                 age = load_age(patient_data)
#                 if not np.isnan(age):
#                     all_features.append(feats)
#                     all_labels.append(label)
#                     all_ages.append(age)
#                     all_sites.append(site_id)

#         except Exception as e:
#             if verbose:
#                 tqdm.write(f"  Skip {patient_id}: {e}")
#             continue

#     pbar.close()

#     if len(all_features) == 0:
#         raise ValueError("No valid features extracted.")

#     feature_names = list(all_features[0].keys())
#     df = pd.DataFrame(all_features)
#     df['label'] = all_labels
#     df['age'] = all_ages
#     df['site'] = all_sites

#     if verbose:
#         print(f"Extracted {len(df)} records, {len(feature_names)} raw features.")
#         print(f"Prevalence: {np.mean(all_labels):.3f} | Sites: {df['site'].nunique()}")

#     # PHASE 2: Poison filtering (site-directionality tolerance)
#     if verbose:
#         print('Filtering site-poisonous features...')

#     sites = df['site'].unique()
#     poison_features = []

#     for col in feature_names:
#         if col.startswith('inter_'):
#             continue
#         corrs = []
#         for site in sites:
#             sub = df[df['site'] == site].dropna(subset=[col, 'label'])
#             if len(sub) > 10 and sub['label'].nunique() > 1:
#                 try:
#                     r, _ = scipy.stats.pointbiserialr(sub['label'], sub[col])
#                     corrs.append(r)
#                 except:
#                     pass
#         valid = [c for c in corrs if not np.isnan(c)]
#         if len(valid) >= 2:
#             cmax, cmin = max(valid), min(valid)
#             if cmax > 0.05 and cmin < -0.05 and (cmax - cmin) > 0.15:
#                 poison_features.append(col)

#     kept = [f for f in feature_names if f not in poison_features]
#     if verbose:
#         print(f"  Kept {len(kept)} | Dropped {len(poison_features)} poison")

#     # PHASE 3: Build interaction features
#     interactions = [
#         ('resp_caisr_ahi', 'physio_late_spo2_drop', 'inter_AHI_SpO2'),
#         ('caisr_prob_w_mean', 'physio_early_eeg_delta', 'inter_WASO_SWA'),
#         ('caisr_prob_r_mean', 'physio_late_emg_rms', 'inter_REM_EMG_Atonia'),
#         ('physio_late_ecg_hrv', 'physio_late_resp_effort', 'inter_HRV_RespEffort'),
#         ('physio_late_eeg_theta_beta', 'physio_late_resp_freq', 'inter_ThetaBeta_RespFreq'),
#         ('physio_mid_eeg_delta_sigma', 'physio_mid_spo2_drop', 'inter_mid_DeltaSigma_SpO2'),
#         ('physio_mid_eeg_theta_beta', 'physio_mid_resp_effort', 'inter_mid_ThetaBeta_Effort'),
#     ]
#     for f1, f2, name in interactions:
#         if f1 in kept and f2 in kept:
#             df[name] = df[f1] * df[f2]
#             kept.append(name)

#     # PHASE 4: Age residualization
#     if verbose:
#         print('Age-residualizing features...')

#     resid_cols = []
#     age_resid_models = {}
#     for col in kept:
#         sub = df.dropna(subset=[col, 'age'])
#         if len(sub) > 10:
#             lr = LinearRegression()
#             lr.fit(sub[['age']].values, sub[col].values)
#             df[f"{col}_resid"] = df[col] - lr.predict(df[['age']].values)
#             resid_cols.append(f"{col}_resid")
#             age_resid_models[col] = lr

#     # PHASE 5: Impute & scale
#     for col in resid_cols:
#         med = df[col].median()
#         if np.isnan(med):
#             med = 0.0
#         df[col] = df[col].fillna(med)

#     X = df[resid_cols].values
#     y = df['label'].values
#     ages = df['age'].values
#     sites_arr = df['site'].values

#     imputer = SimpleImputer(strategy='median')
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(imputer.fit_transform(X))

#     # PHASE 6: Train all candidate models
#     if verbose:
#         print('Training candidate models...')

#     candidates = {}

#     candidates['logistic'] = LogisticRegression(
#         C=0.5, penalty='l2', solver='liblinear',
#         class_weight='balanced', max_iter=1000, random_state=42
#     )
#     candidates['logistic'].fit(Xs, y)

#     if lgb is not None:
#         candidates['lightgbm'] = lgb.LGBMClassifier(
#             n_estimators=200, max_depth=4, learning_rate=0.05,
#             random_state=42, n_jobs=1, verbose=-1,
#             class_weight='balanced'
#         )
#         candidates['lightgbm'].fit(Xs, y)

#     candidates['stack'] = StackingClassifier(
#         estimators=[
#             ('lr1', LogisticRegression(C=0.05, class_weight='balanced',
#                                        solver='liblinear', max_iter=1000)),
#             ('svm', SVC(probability=True, C=0.1, kernel='linear',
#                         class_weight='balanced'))
#         ],
#         final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
#         n_jobs=1
#     )
#     candidates['stack'].fit(Xs, y)

#     estimators_for_mega = [('logistic', candidates['logistic'])]
#     if lgb is not None:
#         estimators_for_mega.append(('lightgbm', candidates['lightgbm']))
#     estimators_for_mega.append(('stack', candidates['stack']))

#     candidates['mega'] = VotingClassifier(
#         estimators=estimators_for_mega,
#         voting='soft',
#         n_jobs=1
#     )
#     candidates['mega'].fit(Xs, y)

#     # PHASE 7: Model selection via site-aware LOSO CV
#     if FINAL_MODEL == "auto":
#         if verbose:
#             print('Running site-aware LOSO CV for model selection...')

#         best_score, best_name = -np.inf, 'logistic'

#         for name, model in candidates.items():
#             scores = []
#             for test_site in sites:
#                 train_mask = sites_arr != test_site
#                 test_mask = sites_arr == test_site

#                 if np.sum(test_mask) < 5 or np.sum(train_mask) < 20:
#                     continue
#                 if len(np.unique(y[test_mask])) < 2:
#                     continue

#                 X_tr, X_val = Xs[train_mask], Xs[test_mask]
#                 y_tr, y_val = y[train_mask], y[test_mask]
#                 a_val = ages[test_mask]

#                 m = _make_fresh_model(name)
#                 m.fit(X_tr, y_tr)
#                 p = m.predict_proba(X_val)[:, 1]
#                 sc = _age_auroc(y_val, p, a_val, delta=2.0)
#                 if not np.isnan(sc):
#                     scores.append(sc)

#             avg = np.mean(scores) if scores else 0
#             if verbose:
#                 print(f"  {name:12s}: LOSO age-AUROC = {avg:.4f} (n={len(scores)} folds)")
#             if avg > best_score:
#                 best_score, best_name = avg, name

#         selected = candidates[best_name]
#         selected_name = best_name
#         if verbose:
#             print(f"Selected: {best_name} (LOSO age-AUROC = {best_score:.4f})")

#     else:
#         selected_name = FINAL_MODEL if FINAL_MODEL in candidates else 'logistic'
#         selected = candidates[selected_name]
#         if verbose:
#             print(f"Forced model: {selected_name}")

#     # PHASE 8: Save artifact
#     os.makedirs(model_folder, exist_ok=True)

#     artifact = {
#         'model': selected,
#         'model_name': selected_name,
#         'candidates': candidates,
#         'scaler': scaler,
#         'imputer': imputer,
#         'kept_features': kept,
#         'resid_cols': resid_cols,
#         'age_resid_models': age_resid_models,
#         'interactions': interactions,
#         'sites_seen': list(sites),
#         'use_sleepfm': use_sleepfm,
#         'sleepfm_channel_groups': sleepfm_extractor.channel_groups if use_sleepfm else None,
#     }

#     joblib.dump(artifact, os.path.join(model_folder, 'model.sav'))
#     if verbose:
#         print('Training complete. Model saved.')


# def load_model(model_folder, verbose):
#     """Load trained model artifact."""
#     return joblib.load(os.path.join(model_folder, 'model.sav'))


# def run_model(model_artifact, record, data_folder, verbose):
#     """
#     Run trained model on a single record.
#     Returns: (binary_prediction, probability)
#     """
#     patient_id = record[HEADERS['bids_folder']]
#     site_id = record[HEADERS['site_id']]
#     session_id = record[HEADERS['session_id']]

#     patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
#     patient_data = load_demographics(patient_data_file, patient_id, session_id)

#     feats = extract_all_features(data_folder, patient_id, site_id, session_id, patient_data)
#     if feats is None:
#         return float('nan'), float('nan')

#     # Extract SleepFM features if model was trained with them
#     if model_artifact.get('use_sleepfm', False):
#         sleepfm_extractor = SleepFMFeatureExtractor(verbose=False)
#         if sleepfm_extractor.model is not None:
#             physio_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
#                                        site_id, f"{patient_id}_ses-{session_id}.edf")
#             if os.path.exists(physio_path):
#                 sleepfm_feats = sleepfm_extractor.extract_features(physio_path)
#                 if sleepfm_feats:
#                     feats.update(sleepfm_feats)

#     df = pd.DataFrame([feats])
#     age = load_age(patient_data)

#     # Build interaction features
#     for f1, f2, name in model_artifact.get('interactions', []):
#         if f1 in df.columns and f2 in df.columns:
#             df[name] = df[f1] * df[f2]

#     # Age residualization
#     for col, lr in model_artifact.get('age_resid_models', {}).items():
#         if col in df.columns:
#             df[f"{col}_resid"] = df[col].values - lr.predict(np.array([[age]]))[0]

#     resid_cols = model_artifact['resid_cols']
#     for c in resid_cols:
#         if c not in df.columns:
#             df[c] = float('nan')

#     X = df[resid_cols].values
#     Xs = model_artifact['scaler'].transform(model_artifact['imputer'].transform(X))

#     model = model_artifact['model']
#     prob = float(model.predict_proba(Xs)[0, 1])
#     binary = int(prob >= 0.5)

#     return binary, prob


# # =============================================================================
# # FEATURE EXTRACTION
# # =============================================================================

# def extract_all_features(data_folder, patient_id, site_id, session_id, patient_data):
#     """
#     Extract all features for a single patient record.
#     NOTE: Human annotations are NOT used - they are unavailable in validation/test.
#     """
#     features = {}

#     # 1. Demographic features
#     features['age'] = load_age(patient_data)
#     sex = load_sex(patient_data, standardize=True)
#     features['sex_male'] = 1 if sex == 'Male' else 0
#     race = load_race(patient_data, standardize=True)
#     features['race_white'] = 1 if race == 'White' else 0
#     features['race_black'] = 1 if race == 'Black' else 0
#     features['race_asian'] = 1 if race == 'Asian' else 0
#     features['race_other'] = 1 if race == 'Others' else 0
#     features['bmi'] = load_bmi(patient_data)

#     # 2. CAISR algorithmic annotations (available in all sets)
#     caisr_path = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
#                               site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
#     features.update(_extract_caisr(caisr_path))

#     # 3. Physiological signals
#     physio_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
#                                site_id, f"{patient_id}_ses-{session_id}.edf")
#     features.update(_extract_physio(physio_path))

#     return features


# def _extract_caisr(edf_path):
#     """Extract features from CAISR algorithmic annotations EDF."""
#     out = {
#         'stage_caisr_tst': np.nan, 'stage_caisr_se': np.nan,
#         'arousal_caisr_rate': np.nan, 'resp_caisr_ahi': np.nan,
#         'limb_caisr_rate': np.nan, 'limb_isolated_rate': np.nan,
#         'limb_periodic_rate': np.nan,
#         'caisr_prob_n3_mean': np.nan, 'caisr_prob_n2_mean': np.nan,
#         'caisr_prob_n1_mean': np.nan, 'caisr_prob_r_mean': np.nan,
#         'caisr_prob_w_mean': np.nan, 'caisr_prob_arousal_mean': np.nan,
#         'stage_transition_rate': np.nan, 'caisr_softmax_entropy': np.nan,
#         'high_conf_arousal_rate': np.nan, 'resp_central_ratio': np.nan,
#     }
#     if not os.path.exists(edf_path):
#         return out

#     try:
#         data_dict, fs_dict = load_signal_data(edf_path)
#         if not data_dict:
#             return out

#         labels = list(data_dict.keys())

#         def get(kws):
#             for lbl in labels:
#                 if all(k in lbl for k in kws):
#                     return data_dict[lbl]
#             return None

#         stages = get(['stage'])
#         resp = get(['resp'])
#         limbs = get(['limb'])
#         arousal = next((data_dict[l] for l in labels
#                         if 'arousal' in l and 'prob' not in l and 'no-ar' not in l), None)

#         pn3 = _sanitize(get(['prob', 'n3']))
#         pn2 = _sanitize(get(['prob', 'n2']))
#         pn1 = _sanitize(get(['prob', 'n1']))
#         pw  = _sanitize(get(['prob', 'w']))
#         pr  = _sanitize(next((data_dict[l] for l in labels
#                                if 'prob' in l and ('_r' in l or 'prob_r' in l)), None))
#         pa  = _sanitize(get(['prob', 'arous']))
#         if pa is None:
#             pa = _sanitize(get(['prob', 'ar']))

#         # Sleep architecture
#         if stages is not None and len(stages) > 0:
#             tst = (len(stages) * 30) / 3600
#             out['stage_caisr_tst'] = tst
#             out['stage_caisr_se'] = float(np.sum(np.isin(stages, [1,2,3,4])) / len(stages))
#             out['stage_transition_rate'] = float(np.sum(np.diff(stages) != 0) / max(tst, 0.5))

#         dh = max(out['stage_caisr_tst'], 0.5)

#         # Event rates (per hr)
#         if arousal is not None:
#             out['arousal_caisr_rate'] = float(_count_events(arousal, [1]) / dh)
#         if resp is not None:
#             out['resp_caisr_ahi'] = float(_count_events(resp, [1,2,3,4]) / dh)
#             tap = _count_events(resp, [2]) + _count_events(resp, [1])
#             out['resp_central_ratio'] = float(_count_events(resp, [2]) / tap) if tap > 0 else 0.0
#         if limbs is not None:
#             out['limb_caisr_rate'] = float(_count_events(limbs, [1,2]) / dh)
#             out['limb_isolated_rate'] = float(_count_events(limbs, [1]) / dh)
#             out['limb_periodic_rate'] = float(_count_events(limbs, [2]) / dh)

#         # Mean probabilities
#         if pn3 is not None: out['caisr_prob_n3_mean'] = float(np.mean(pn3))
#         if pn2 is not None: out['caisr_prob_n2_mean'] = float(np.mean(pn2))
#         if pw  is not None: out['caisr_prob_w_mean']  = float(np.mean(pw))
#         if pr  is not None: out['caisr_prob_r_mean']  = float(np.mean(pr))

#         # Softmax entropy across sleep stages
#         plist = [pn3, pn2, pn1, pr, pw]
#         if all(p is not None for p in plist):
#             stacked = np.stack(plist, axis=0)
#             sv = np.sum(stacked, axis=0, keepdims=True)
#             sv[sv == 0] = 1.0
#             stacked = stacked / sv
#             out['caisr_softmax_entropy'] = float(
#                 np.mean(-np.sum(stacked * np.log(stacked + 1e-9), axis=0)))

#         if pa is not None:
#             out['caisr_prob_arousal_mean'] = float(np.mean(pa))
#             m = (pa > 0.85).astype(int)
#             out['high_conf_arousal_rate'] = float(_count_events(m, [1]) / dh)

#     except Exception:
#         pass
#     return out


# def _extract_physio(edf_path):
#     """Extract features from physiological signals with early/mid/late windowing."""
#     out = {}
#     for w in WINDOWS:
#         for m in EEG_METRICS:
#             out[f'physio_{w}_eeg_{m}'] = np.nan
#         out[f'physio_{w}_emg_rms'] = np.nan
#         out[f'physio_{w}_ecg_hrv'] = np.nan
#         out[f'physio_{w}_resp_freq'] = np.nan
#         out[f'physio_{w}_resp_effort'] = np.nan
#         out[f'physio_{w}_spo2_drop'] = np.nan

#     if not os.path.exists(edf_path):
#         return out

#     try:
#         data_dict, fs_dict = load_signal_data(edf_path)
#         if not data_dict:
#             return out

#         labels = list(data_dict.keys())
#         eeg, eeg_fs = _find_sig(labels, data_dict, fs_dict, 'eeg')
#         emg, emg_fs = _find_sig(labels, data_dict, fs_dict, 'emg')
#         ecg, ecg_fs = _find_sig(labels, data_dict, fs_dict, 'ecg')
#         rsp, rsp_fs = _find_sig(labels, data_dict, fs_dict, 'resp_airflow')
#         eff, eff_fs = _find_sig(labels, data_dict, fs_dict, 'resp_effort')
#         sp2, sp2_fs = _find_sig(labels, data_dict, fs_dict, 'spo2')

#         # Estimate recording duration
#         dur = 0
#         for s, f in [(eeg, eeg_fs), (emg, emg_fs), (ecg, ecg_fs),
#                      (rsp, rsp_fs), (eff, eff_fs), (sp2, sp2_fs)]:
#             if s is not None:
#                 dur = max(dur, len(s) / f)
#         if dur <= 0:
#             return out

#         t3 = dur / 3.0
#         bounds = {'early': (0, t3), 'mid': (t3, 2*t3), 'late': (2*t3, dur)}

#         for stage, (st, en) in bounds.items():
#             # EEG spectral features
#             if eeg is not None and eeg_fs > 0:
#                 sl = eeg[int(st*eeg_fs):int(en*eeg_fs)]
#                 if len(sl) > eeg_fs * 10:
#                     sl = (sl - np.nanmean(sl)) / (np.nanstd(sl) + 1e-8)
#                     ef = _eeg_spectrum(sl, eeg_fs)
#                     for idx, m in enumerate(EEG_METRICS):
#                         out[f'physio_{stage}_eeg_{m}'] = ef[idx]

#             # EMG RMS
#             if emg is not None and emg_fs > 0:
#                 sl = emg[int(st*emg_fs):int(en*emg_fs)]
#                 if len(sl) > emg_fs * 10:
#                     out[f'physio_{stage}_emg_rms'] = float(
#                         np.sqrt(np.mean(np.square(sl - np.mean(sl)))))

#             # ECG HRV proxy
#             if ecg is not None and ecg_fs > 0:
#                 sl = ecg[int(st*ecg_fs):int(en*ecg_fs)]
#                 if len(sl) > ecg_fs * 10:
#                     out[f'physio_{stage}_ecg_hrv'] = float(np.var(np.diff(sl)))

#             # Respiratory frequency
#             if rsp is not None and rsp_fs > 0:
#                 sl = rsp[int(st*rsp_fs):int(en*rsp_fs)]
#                 if len(sl) > rsp_fs * 10:
#                     out[f'physio_{stage}_resp_freq'] = _resp_spectrum(sl, rsp_fs)[0]

#             # Respiratory effort variance
#             if eff is not None and eff_fs > 0:
#                 sl = eff[int(st*eff_fs):int(en*eff_fs)]
#                 if len(sl) > eff_fs * 10:
#                     out[f'physio_{stage}_resp_effort'] = float(np.var(sl))

#             # SpO2 drop (95th - 5th percentile)
#             if sp2 is not None and sp2_fs > 0:
#                 sl = sp2[int(st*sp2_fs):int(en*sp2_fs)]
#                 if len(sl) > sp2_fs * 10:
#                     out[f'physio_{stage}_spo2_drop'] = float(
#                         np.percentile(sl, 95) - np.percentile(sl, 5))

#     except Exception:
#         pass
#     return out


# # =============================================================================
# # UTILITIES
# # =============================================================================

# def _find_sig(labels, data_dict, fs_dict, target):
#     """Find signal by type using ordered manifest matching."""
#     manifest = {
#         'eeg': ['c3-m2','c4-m1','c3','c4','f3-m2','f4-m1','f3','f4',
#                 'o1-m2','o2-m1','eeg'],
#         'emg': ['chin1-chin2','chin','emg.subm','emg','chin1','emg1','chin2','emg2'],
#         'ecg': ['ecg','ekg','ecg-la','ecg-v1','ecg i','ecg ii','ecg1'],
#         'resp_airflow': ['airflow','flow','thermal','thermistor','nasal_pressure'],
#         'resp_effort': ['abd','abdomen','chest','thorax','effort abd','effort tho'],
#         'spo2': ['spo2','sao2','osat','o2sat']
#     }
#     for t in manifest.get(target, []):
#         for lbl in labels:
#             if t in lbl:
#                 return data_dict[lbl], fs_dict.get(lbl, 1.0)
#     return None, 1.0


# def _sanitize(sig):
#     """Sanitize probability signal to [0, 1] range."""
#     if sig is None or len(sig) == 0:
#         return sig
#     mn, mx = np.min(sig), np.max(sig)
#     if mx > 1.0001 or mn < -0.0001:
#         d = mx - mn
#         if d > 1e-6:
#             sig = (sig - mn) / d
#     return np.clip(sig, 0.0, 1.0)


# def _count_events(arr, codes):
#     """Count contiguous events in a discrete signal."""
#     if arr is None or len(arr) == 0:
#         return 0
#     b = np.isin(arr, codes).astype(int)
#     d = np.diff(b)
#     return max(np.sum(d == 1) + (1 if b[0] == 1 else 0), 0)


# def _eeg_spectrum(signal, fs):
#     """Extract 12 EEG spectral features."""
#     if signal is None or len(signal) == 0:
#         return [np.nan] * 12
#     try:
#         n = len(signal)
#         fft_vals = np.abs(np.fft.rfft(signal)) ** 2
#         freqs = np.fft.rfftfreq(n, d=1.0/fs)
#         ti = (freqs >= 0.5) & (freqs <= 30)
#         tp = np.sum(fft_vals[ti])
#         if tp == 0:
#             return [np.nan] * 12

#         delta = np.sum(fft_vals[(freqs >= 0.5) & (freqs < 4)]) / tp
#         theta = np.sum(fft_vals[(freqs >= 4) & (freqs < 8)]) / tp
#         alpha = np.sum(fft_vals[(freqs >= 8) & (freqs < 12)]) / tp
#         sigma = np.sum(fft_vals[(freqs >= 12) & (freqs < 15)]) / tp
#         beta  = np.sum(fft_vals[(freqs >= 15) & (freqs <= 30)]) / tp

#         at = alpha / (theta + 1e-8)
#         tb = theta / (beta + 1e-8)
#         sl = (delta + theta) / (alpha + beta + 1e-8)
#         ds = delta / (sigma + 1e-8)

#         pn = fft_vals[ti] / tp
#         ent = scipy.stats.entropy(pn, base=2)
#         cp = np.cumsum(fft_vals[ti])
#         sef50 = freqs[ti][np.where(cp >= 0.50 * tp)[0][0]]
#         sef90 = freqs[ti][np.where(cp >= 0.90 * tp)[0][0]]

#         return [delta, theta, alpha, sigma, beta, at, tb, sl, ds, ent, sef50, sef90]
#     except Exception:
#         return [np.nan] * 12


# def _resp_spectrum(signal, fs):
#     """Extract respiratory peak frequency and effort variance."""
#     if signal is None or len(signal) == 0:
#         return [np.nan, np.nan]
#     try:
#         n = len(signal)
#         fft_vals = np.abs(np.fft.rfft(signal)) ** 2
#         freqs = np.fft.rfftfreq(n, d=1.0/fs)
#         ri = (freqs >= 0.1) & (freqs <= 0.5)
#         if np.sum(ri) == 0:
#             return [np.nan, np.nan]
#         pf = freqs[ri][np.argmax(fft_vals[ri])]
#         ev = np.var(signal)
#         return [float(pf), float(ev)]
#     except Exception:
#         return [np.nan, np.nan]


# def _age_auroc(y_true, y_prob, ages, delta=2.0):
#     """
#     Compute age-conditioned AUROC (official challenge metric).
#     Only compares positive/negative pairs within delta years.
#     """
#     y_true, y_prob, ages = np.asarray(y_true), np.asarray(y_prob), np.asarray(ages)
#     pos = np.where(y_true == 1)[0]
#     neg = np.where(y_true == 0)[0]
#     c, d, t = 0, 0, 0
#     for i in pos:
#         vn = neg[np.abs(ages[neg] - ages[i]) <= delta]
#         for j in vn:
#             if y_prob[i] > y_prob[j]: c += 1
#             elif y_prob[i] < y_prob[j]: d += 1
#             else: t += 1
#     tot = c + d + t
#     return (c + 0.5*t) / tot if tot > 0 else np.nan


# def _make_fresh_model(name):
#     """Create a fresh unfitted model for CV evaluation."""
#     if name == 'logistic':
#         return LogisticRegression(C=0.5, penalty='l2', solver='liblinear',
#                                    class_weight='balanced', max_iter=1000, random_state=42)
#     elif name == 'lightgbm' and lgb is not None:
#         return lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
#                                    random_state=42, n_jobs=1, verbose=-1,
#                                    class_weight='balanced')
#     elif name == 'stack':
#         return StackingClassifier(
#             estimators=[
#                 ('lr1', LogisticRegression(C=0.05, class_weight='balanced',
#                                             solver='liblinear', max_iter=1000)),
#                 ('svm', SVC(probability=True, C=0.1, kernel='linear',
#                            class_weight='balanced'))
#             ],
#             final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
#             n_jobs=1
#         )
#     elif name == 'mega':
#         estimators = [('logistic', LogisticRegression(C=0.5, penalty='l2', solver='liblinear',
#                                                        class_weight='balanced', max_iter=1000, random_state=42))]
#         if lgb is not None:
#             estimators.append(('lightgbm', lgb.LGBMClassifier(n_estimators=200, max_depth=4,
#                                                                learning_rate=0.05, random_state=42,
#                                                                n_jobs=1, verbose=-1, class_weight='balanced')))
#         estimators.append(('stack', StackingClassifier(
#             estimators=[
#                 ('lr1', LogisticRegression(C=0.05, class_weight='balanced', solver='liblinear', max_iter=1000)),
#                 ('svm', SVC(probability=True, C=0.1, kernel='linear', class_weight='balanced'))
#             ],
#             final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
#             n_jobs=1
#         )))
#         return VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
#     else:
#         raise ValueError(f"Unknown model name: {name}")


