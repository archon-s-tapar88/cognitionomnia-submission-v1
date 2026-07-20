#!/usr/bin/env python
"""
PhysioNet Challenge 2026 - Submission V7b: Epoch-Level Stage-Segmented Edition
Base: V7 with TRUE epoch-level stage-conditional spectral features.

PHILOSOPHY:
- V7 used PROXIES for stage-conditional features (e.g., late-night delta ≈ N3 SWA).
- V7b computes TRUE stage-conditional spectral features by aligning CAISR stages
  with EEG signal epochs. This is more precise and dementia-biomarker-specific.
- Additionally adds "relative features" (within-subject ratios) to handle the
  extreme hardware heterogeneity in I0004 (3 different amplifier batches).
- Also adds arousal-clustering features inspired by competitor abstract (#175).

KEY DIFFERENCES FROM V7:
1. True epoch-level stage-conditional spectral analysis:
   - Load CAISR stage array (30s epochs) from algorithmic annotations
   - For each stage (N3, N2, REM, N1, Wake), extract corresponding EEG segments
   - Compute Welch PSD per-stage → genuine stage-conditional band powers
2. Relative features (within-subject ratios) — resist inter-site hardware shift:
   - N3 delta / Wake delta (slow-wave activity relative to wake baseline)
   - N2 sigma / N2 theta (spindle-to-theta ratio)
   - REM theta / REM alpha
   - N3 slowing ratio: (delta+theta)/(alpha+beta) in N3 only
3. Arousal clustering features (inspired by Momochi-SleepAI):
   - Arousal clustering ratio: contiguous arousal bouts vs isolated arousals
   - REM arousal ratio: arousals during REM / total arousals
4. Simpler model: only Ridge Logistic + ensemble (drop LightGBM to reduce variance)
5. All processing uses official channel_table.csv for robust channel mapping
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
from tqdm import tqdm
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore")

from helper_code import *

# =============================================================================
# CONFIGURATION
# =============================================================================
FINAL_MODEL = "auto"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# EEG spectral bands
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'sigma': (12, 15),
    'beta': (15, 30),
}

# Stage encoding in CAISR: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unavailable
STAGE_NAMES = {1: 'n3', 2: 'n2', 3: 'n1', 4: 'rem', 5: 'wake'}

# Features to age-residualize
AGE_RESID_FEATURES = [
    'stage_caisr_tst', 'stage_caisr_se',
    'caisr_n3_pct', 'caisr_n2_pct', 'caisr_rem_pct',
    'caisr_n1_sleep_pct', 'caisr_waso_min',
    'n3_bout_mean_dur', 'rem_bout_mean_dur',
    'ecg_mean_hr', 'ecg_hrv_proxy',
]

# =============================================================================
# REQUIRED FUNCTIONS
# =============================================================================

def train_model(data_folder, model_folder, verbose):
    """Train model with site-aware LOSO CV."""
    if verbose:
        print('Finding Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data provided.')

    if verbose:
        print(f'Found {num_records} records. Extracting features...')

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
                tqdm.write(f" Skip {patient_id}: {e}")
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

    # -------------------------------------------------------------------------
    # PHASE 2: Poison filtering
    # -------------------------------------------------------------------------
    if verbose:
        print('Filtering site-poisonous features...')

    sites = df['site'].unique()
    poison_features = []

    for col in feature_names:
        if col.startswith('inter_'):
            continue
        if col == 'age':
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
        print(f" Kept {len(kept)} | Dropped {len(poison_features)} poison")

    # -------------------------------------------------------------------------
    # PHASE 3: Build interaction features
    # -------------------------------------------------------------------------
    interactions = [
        ('resp_caisr_ahi', 'physio_late_spo2_drop', 'inter_AHI_SpO2'),
        ('caisr_prob_w_mean', 'n3_delta_power', 'inter_WASO_N3SWA'),
        ('caisr_prob_r_mean', 'physio_late_emg_rms', 'inter_REM_EMG_Atonia'),
        ('n3_delta_power', 'caisr_n3_pct', 'inter_N3SWA_N3Pct'),
        ('age', 'n3_delta_power', 'inter_age_N3SWA'),
        ('n3_slowing_ratio', 'caisr_n1_sleep_pct', 'inter_Slowing_Frag'),
    ]

    for f1, f2, name in interactions:
        if f1 in kept and f2 in kept:
            df[name] = df[f1] * df[f2]
            if name not in kept:
                kept.append(name)

    # -------------------------------------------------------------------------
    # PHASE 4: Age residualization — SELECTIVE
    # -------------------------------------------------------------------------
    if verbose:
        print('Age-residualizing select features...')

    resid_cols = []
    age_resid_models = {}
    raw_kept = []

    for col in kept:
        if col in AGE_RESID_FEATURES and col != 'age':
            sub = df.dropna(subset=[col, 'age'])
            if len(sub) > 10:
                lr = LinearRegression()
                lr.fit(sub[['age']].values, sub[col].values)
                df[f"{col}_resid"] = df[col] - lr.predict(df[['age']].values)
                resid_cols.append(f"{col}_resid")
                age_resid_models[col] = lr
        else:
            raw_kept.append(col)

    final_features = raw_kept + resid_cols

    for col in final_features:
        med = df[col].median()
        if np.isnan(med):
            med = 0.0
        df[col] = df[col].fillna(med)

    X = df[final_features].values
    y = df['label'].values
    ages = df['age'].values
    sites_arr = df['site'].values

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xs = scaler.fit_transform(imputer.fit_transform(X))

    # -------------------------------------------------------------------------
    # PHASE 5: Train candidate models — ONLY 2 for less selection bias
    # -------------------------------------------------------------------------
    if verbose:
        print('Training candidate models...')

    candidates = {}

    # 1. Ridge Logistic (strong L2 regularization)
    candidates['logistic'] = LogisticRegression(
        C=0.1, penalty='l2', solver='liblinear',
        class_weight='balanced', max_iter=1000, random_state=42
    )
    candidates['logistic'].fit(Xs, y)

    # 2. Soft-voting ensemble: logistic + slightly different logistic
    candidates['ensemble'] = VotingClassifier(
        estimimators=[
            ('lr1', LogisticRegression(C=0.1, penalty='l2', solver='liblinear',
                                       class_weight='balanced', max_iter=1000, random_state=42)),
            ('lr2', LogisticRegression(C=0.05, penalty='l2', solver='liblinear',
                                       class_weight='balanced', max_iter=1000, random_state=43)),
        ],
        voting='soft',
        n_jobs=1
    )
    candidates['ensemble'].fit(Xs, y)

    # -------------------------------------------------------------------------
    # PHASE 6: LOSO CV model selection
    # -------------------------------------------------------------------------
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
                print(f" {name:12s}: LOSO age-AUROC = {avg:.4f} (n={len(scores)} folds)")
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

    # -------------------------------------------------------------------------
    # PHASE 7: Save artifact
    # -------------------------------------------------------------------------
    os.makedirs(model_folder, exist_ok=True)
    artifact = {
        'model': selected,
        'model_name': selected_name,
        'candidates': candidates,
        'scaler': scaler,
        'imputer': imputer,
        'kept_features': kept,
        'final_features': final_features,
        'raw_kept': raw_kept,
        'resid_cols': resid_cols,
        'age_resid_models': age_resid_models,
        'interactions': interactions,
        'sites_seen': list(sites),
    }
    joblib.dump(artifact, os.path.join(model_folder, 'model.sav'))

    if verbose:
        print('Training complete. Model saved.')


def load_model(model_folder, verbose):
    return joblib.load(os.path.join(model_folder, 'model.sav'))


def run_model(model_artifact, record, data_folder, verbose):
    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(patient_data_file, patient_id, session_id)
    feats = extract_all_features(data_folder, patient_id, site_id, session_id, patient_data)

    if feats is None:
        return float('nan'), float('nan')

    df = pd.DataFrame([feats])
    age = load_age(patient_data)

    for f1, f2, name in model_artifact.get('interactions', []):
        if f1 in df.columns and f2 in df.columns:
            df[name] = df[f1] * df[f2]

    for col, lr in model_artifact.get('age_resid_models', {}).items():
        if col in df.columns:
            df[f"{col}_resid"] = df[col].values - lr.predict(np.array([[age]]))[0]

    final_features = model_artifact['final_features']
    for c in final_features:
        if c not in df.columns:
            df[c] = float('nan')

    X = df[final_features].values
    Xs = model_artifact['scaler'].transform(model_artifact['imputer'].transform(X))

    model = model_artifact['model']
    prob = float(model.predict_proba(Xs)[0, 1])
    binary = int(prob >= 0.5)
    return binary, prob


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_all_features(data_folder, patient_id, site_id, session_id, patient_data):
    """Extract all features for a single patient record."""
    features = {}

    # 1. Demographics
    age = load_age(patient_data)
    features['age'] = age
    sex = load_sex(patient_data, standardize=True)
    features['sex_male'] = 1 if sex == 'Male' else 0
    race = load_race(patient_data, standardize=True)
    features['race_white'] = 1 if race == 'White' else 0
    features['race_black'] = 1 if race == 'Black' else 0
    features['race_asian'] = 1 if race == 'Asian' else 0
    features['race_other'] = 1 if race == 'Others' else 0
    features['bmi'] = load_bmi(patient_data)

    # 2. CAISR algorithmic annotations
    caisr_path = os.path.join(
        data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id,
        f"{patient_id}_ses-{session_id}_caisr_annotations.edf"
    )
    caisr_feats = _extract_caisr(caisr_path)
    features.update(caisr_feats)

    # 3. Physiological signals with epoch-level stage segmentation
    physio_path = os.path.join(
        data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id,
        f"{patient_id}_ses-{session_id}.edf"
    )
    physio_feats = _extract_physio(physio_path, caisr_path, caisr_feats)
    features.update(physio_feats)

    return features


def _extract_caisr(edf_path):
    """Extract CAISR features with dementia-relevant architecture markers."""
    out = {
        'stage_caisr_tst': np.nan, 'stage_caisr_se': np.nan,
        'arousal_caisr_rate': np.nan, 'resp_caisr_ahi': np.nan,
        'limb_caisr_rate': np.nan, 'limb_periodic_rate': np.nan,
        'caisr_prob_n3_mean': np.nan, 'caisr_prob_n2_mean': np.nan,
        'caisr_prob_n1_mean': np.nan, 'caisr_prob_r_mean': np.nan,
        'caisr_prob_w_mean': np.nan, 'caisr_prob_arousal_mean': np.nan,
        'stage_transition_rate': np.nan, 'caisr_softmax_entropy': np.nan,
        'high_conf_arousal_rate': np.nan, 'resp_central_ratio': np.nan,
        'caisr_n3_pct': np.nan, 'caisr_n2_pct': np.nan,
        'caisr_n1_pct': np.nan, 'caisr_rem_pct': np.nan,
        'caisr_wake_pct': np.nan, 'caisr_sleep_latency_min': np.nan,
        'caisr_rem_latency_min': np.nan, 'caisr_waso_min': np.nan,
        'caisr_n1_sleep_pct': np.nan, 'caisr_plm_index': np.nan,
        'caisr_rdi': np.nan, 'n3_bout_mean_dur': np.nan,
        'rem_bout_mean_dur': np.nan,
        # Arousal clustering features
        'arousal_cluster_ratio': np.nan,
        'rem_arousal_ratio': np.nan,
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
        pw = _sanitize(get(['prob', 'w']))
        pr = _sanitize(next((data_dict[l] for l in labels
                             if 'prob' in l and ('_r' in l or 'prob_r' in l)), None))
        pa = _sanitize(get(['prob', 'arous']))
        if pa is None:
            pa = _sanitize(get(['prob', 'ar']))

        if stages is not None and len(stages) > 0:
            epoch_dur_min = 0.5
            tst_hours = (len(stages) * 30) / 3600
            out['stage_caisr_tst'] = tst_hours

            valid = stages[stages < 9]
            if len(valid) > 0:
                out['caisr_wake_pct'] = float(np.mean(valid == 5))
                out['caisr_n1_pct'] = float(np.mean(valid == 3))
                out['caisr_n2_pct'] = float(np.mean(valid == 2))
                out['caisr_n3_pct'] = float(np.mean(valid == 1))
                out['caisr_rem_pct'] = float(np.mean(valid == 4))
                out['stage_caisr_se'] = float(np.mean((valid >= 1) & (valid <= 4)))

                sleep_idx = np.where(np.isin(valid, [1, 2, 3, 4]))[0]
                if len(sleep_idx) > 0:
                    out['caisr_sleep_latency_min'] = float(sleep_idx[0] * epoch_dur_min)
                else:
                    out['caisr_sleep_latency_min'] = float(len(valid) * epoch_dur_min)

                rem_idx = np.where(valid == 4)[0]
                if len(rem_idx) > 0:
                    out['caisr_rem_latency_min'] = float(rem_idx[0] * epoch_dur_min)

                sleep_started = False
                wake_epochs = 0
                for s in valid:
                    if not sleep_started and s in [1, 2, 3, 4]:
                        sleep_started = True
                    if sleep_started and s == 5:
                        wake_epochs += 1
                out['caisr_waso_min'] = float(wake_epochs * epoch_dur_min)

                sleep_stages = valid[np.isin(valid, [1, 2, 3, 4])]
                if len(sleep_stages) > 0:
                    out['caisr_n1_sleep_pct'] = float(np.mean(sleep_stages == 3))

                out['n3_bout_mean_dur'] = _mean_bout_duration(valid, target=1, epoch_min=epoch_dur_min)
                out['rem_bout_mean_dur'] = _mean_bout_duration(valid, target=4, epoch_min=epoch_dur_min)

            out['stage_transition_rate'] = float(np.sum(np.diff(stages) != 0) / max(tst_hours, 0.5))
            dh = max(out['stage_caisr_tst'], 0.5)

            if arousal is not None:
                out['arousal_caisr_rate'] = float(_count_events(arousal, [1]) / dh)
                # Arousal clustering: ratio of clustered (bout duration >= 2 epochs = 60s) to total
                out['arousal_cluster_ratio'] = _arousal_cluster_ratio(arousal, min_bout_epochs=2)
                # REM arousal ratio: arousals during REM epochs / total arousals
                if len(valid) == len(arousal):
                    rem_arousals = np.sum((valid == 4) & (arousal == 1))
                    total_arousals = np.sum(arousal == 1)
                    out['rem_arousal_ratio'] = float(rem_arousals / total_arousals) if total_arousals > 0 else 0.0

            if resp is not None:
                out['resp_caisr_ahi'] = float(_count_events(resp, [1, 2, 3, 4]) / dh)
                tap = _count_events(resp, [2]) + _count_events(resp, [1])
                out['resp_central_ratio'] = float(_count_events(resp, [2]) / tap) if tap > 0 else 0.0
                out['caisr_rdi'] = out['resp_caisr_ahi'] + out.get('arousal_caisr_rate', 0)
            if limbs is not None:
                out['limb_caisr_rate'] = float(_count_events(limbs, [1, 2]) / dh)
                out['limb_periodic_rate'] = float(_count_events(limbs, [2]) / dh)
                out['caisr_plm_index'] = out['limb_periodic_rate']

            if pn3 is not None: out['caisr_prob_n3_mean'] = float(np.mean(pn3))
            if pn2 is not None: out['caisr_prob_n2_mean'] = float(np.mean(pn2))
            if pw is not None: out['caisr_prob_w_mean'] = float(np.mean(pw))
            if pr is not None: out['caisr_prob_r_mean'] = float(np.mean(pr))

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


def _extract_physio(physio_path, caisr_path, caisr_feats):
    """
    Extract physio features with TRUE epoch-level stage segmentation.
    KEY: Loads CAISR stages from caisr_path, aligns with EEG signal,
    computes spectral features per-stage.
    """
    out = {}

    # Initialize temporal window features (for fallback + compat)
    for w in ['early', 'mid', 'late']:
        for band in BANDS.keys():
            out[f'physio_{w}_eeg_{band}'] = np.nan
        out[f'physio_{w}_emg_rms'] = np.nan
        out[f'physio_{w}_ecg_hrv'] = np.nan
        out[f'physio_{w}_resp_freq'] = np.nan
        out[f'physio_{w}_resp_effort'] = np.nan
        out[f'physio_{w}_spo2_drop'] = np.nan

    # Stage-conditional spectral features (TRUE, not proxy)
    for stg in ['n3', 'n2', 'rem', 'n1', 'wake']:
        for band in BANDS.keys():
            out[f'{stg}_{band}_power'] = np.nan
        out[f'{stg}_spectral_entropy'] = np.nan

    # Relative features (within-subject ratios)
    out['n3_delta_power'] = np.nan
    out['n3_slowing_ratio'] = np.nan
    out['n2_spindle_theta_ratio'] = np.nan
    out['rem_theta_alpha_ratio'] = np.nan
    out['n3_delta_wake_delta_ratio'] = np.nan
    out['ecg_mean_hr'] = np.nan
    out['ecg_hrv_proxy'] = np.nan

    if not os.path.exists(physio_path):
        return out

    try:
        # Load physiological data
        data_dict, fs_dict = load_signal_data(physio_path)
        if not data_dict:
            return out

        # Channel mapping using official channel_table.csv
        rename_rules = load_rename_rules(DEFAULT_CSV_PATH)
        original_labels = list(data_dict.keys())
        rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

        processed_channels = {}
        processed_fs = {}
        for old_label, data in data_dict.items():
            if old_label in cols_to_drop:
                continue
            new_label = rename_map.get(old_label, old_label.lower())
            processed_channels[new_label] = data
            processed_fs[new_label] = fs_dict.get(old_label, 1.0)

        # Bipolar derivations for EEG
        bipolar_configs = [
            ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
            ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
            ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ]
        for target, pos, neg_list in bipolar_configs:
            if target in processed_channels or pos not in processed_channels:
                continue
            if not all(n in processed_channels for n in neg_list):
                continue
            ref_sig = processed_channels[neg_list[0]] if len(neg_list) == 1 else tuple(processed_channels[n] for n in neg_list)
            derived = derive_bipolar_signal(processed_channels[pos], ref_sig)
            if derived is not None:
                processed_channels[target] = derived
                processed_fs[target] = processed_fs[pos]

        # Find best channels
        eeg, eeg_fs = _find_best_channel(processed_channels, processed_fs,
                                         ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'c3', 'c4'])
        emg, emg_fs = _find_best_channel(processed_channels, processed_fs,
                                         ['chin1-chin2', 'chin', 'chin1'])
        ecg, ecg_fs = _find_best_channel(processed_channels, processed_fs,
                                         ['ecg', 'ekg'])
        rsp, rsp_fs = _find_best_channel(processed_channels, processed_fs,
                                         ['airflow', 'flow', 'thermal', 'nasal_pressure'])
        eff, eff_fs = _find_best_channel(processed_channels, processed_fs,
                                         ['abd', 'chest', 'thorax'])
        sp2, sp2_fs = _find_best_channel(processed_channels, processed_fs,
                                         ['spo2', 'sao2'])

        # Estimate recording duration
        dur = 0
        for s, f in [(eeg, eeg_fs), (emg, emg_fs), (ecg, ecg_fs),
                     (rsp, rsp_fs), (eff, eff_fs), (sp2, sp2_fs)]:
            if s is not None and f > 0:
                dur = max(dur, len(s) / f)

        if dur <= 0:
            return out

        # --- Temporal window features (fallback) ---
        t3 = dur / 3.0
        bounds = {'early': (0, t3), 'mid': (t3, 2*t3), 'late': (2*t3, dur)}
        for stage, (st, en) in bounds.items():
            if eeg is not None and eeg_fs > 0:
                sl = eeg[int(st * eeg_fs):int(en * eeg_fs)]
                if len(sl) > eeg_fs * 10:
                    sl = (sl - np.nanmean(sl)) / (np.nanstd(sl) + 1e-8)
                    ef = _eeg_spectrum(sl, eeg_fs)
                    for idx, band in enumerate(BANDS.keys()):
                        out[f'physio_{stage}_eeg_{band}'] = ef[idx]
            if emg is not None and emg_fs > 0:
                sl = emg[int(st * emg_fs):int(en * emg_fs)]
                if len(sl) > emg_fs * 10:
                    out[f'physio_{stage}_emg_rms'] = float(
                        np.sqrt(np.mean(np.square(sl - np.mean(sl)))))
            if ecg is not None and ecg_fs > 0:
                sl = ecg[int(st * ecg_fs):int(en * ecg_fs)]
                if len(sl) > ecg_fs * 10:
                    out[f'physio_{stage}_ecg_hrv'] = float(np.var(np.diff(sl)))
            if rsp is not None and rsp_fs > 0:
                sl = rsp[int(st * rsp_fs):int(en * rsp_fs)]
                if len(sl) > rsp_fs * 10:
                    out[f'physio_{stage}_resp_freq'] = _resp_spectrum(sl, rsp_fs)[0]
            if eff is not None and eff_fs > 0:
                sl = eff[int(st * eff_fs):int(en * eff_fs)]
                if len(sl) > eff_fs * 10:
                    out[f'physio_{stage}_resp_effort'] = float(np.var(sl))
            if sp2 is not None and sp2_fs > 0:
                sl = sp2[int(st * sp2_fs):int(en * sp2_fs)]
                if len(sl) > sp2_fs * 10:
                    out[f'physio_{stage}_spo2_drop'] = float(
                        np.percentile(sl, 95) - np.percentile(sl, 5))

        # --- TRUE epoch-level stage-conditional spectral features ---
        if eeg is not None and eeg_fs > 0 and os.path.exists(caisr_path):
            stages = _load_caisr_stages(caisr_path)
            if stages is not None and len(stages) > 0:
                # Each stage epoch = 30 seconds
                epoch_len_samples = int(30 * eeg_fs)
                total_epochs = min(len(stages), len(eeg) // epoch_len_samples)

                # Collect per-stage spectral features
                stage_band_powers = {stg: {band: [] for band in BANDS.keys()} for stg in STAGE_NAMES.values()}

                for epoch_idx in range(total_epochs):
                    st = epoch_idx * epoch_len_samples
                    en = st + epoch_len_samples
                    if en > len(eeg):
                        break
                    stage_code = stages[epoch_idx]
                    if stage_code not in STAGE_NAMES:
                        continue
                    stg_name = STAGE_NAMES[stage_code]

                    seg = eeg[st:en]
                    if len(seg) < eeg_fs * 10:
                        continue
                    seg = (seg - np.nanmean(seg)) / (np.nanstd(seg) + 1e-8)
                    powers = _eeg_spectrum(seg, eeg_fs)
                    for idx, band in enumerate(BANDS.keys()):
                        if not np.isnan(powers[idx]):
                            stage_band_powers[stg_name][band].append(powers[idx])

                # Aggregate per-stage features
                for stg_name, band_dict in stage_band_powers.items():
                    for band, vals in band_dict.items():
                        if len(vals) > 0:
                            out[f'{stg_name}_{band}_power'] = float(np.mean(vals))

                    # Spectral entropy per stage
                    if all(len(band_dict[b]) > 0 for b in BANDS.keys()):
                        means = np.array([np.mean(band_dict[b]) for b in BANDS.keys()])
                        means = means / (np.sum(means) + 1e-8)
                        out[f'{stg_name}_spectral_entropy'] = float(
                            scipy.stats.entropy(means, base=2))

                # --- Relative features (within-subject ratios) ---
                # These resist hardware differences because they're ratios within the same recording
                n3_delta = out.get('n3_delta_power', np.nan)
                n3_theta = out.get('n3_theta_power', np.nan)
                n3_alpha = out.get('n3_alpha_power', np.nan)
                n3_beta = out.get('n3_beta_power', np.nan)
                wake_delta = out.get('wake_delta_power', np.nan)
                n2_sigma = out.get('n2_sigma_power', np.nan)
                n2_theta = out.get('n2_theta_power', np.nan)
                rem_theta = out.get('rem_theta_power', np.nan)
                rem_alpha = out.get('rem_alpha_power', np.nan)

                # N3 slow-wave activity (delta in N3)
                out['n3_delta_power'] = n3_delta

                # N3 slowing ratio: (delta+theta)/(alpha+beta) in N3
                if not any(np.isnan(v) for v in [n3_delta, n3_theta, n3_alpha, n3_beta]):
                    out['n3_slowing_ratio'] = (n3_delta + n3_theta) / (n3_alpha + n3_beta + 1e-8)

                # N2 spindle-to-theta ratio (spindle proxy)
                if not np.isnan(n2_sigma) and not np.isnan(n2_theta) and n2_theta > 0:
                    out['n2_spindle_theta_ratio'] = n2_sigma / n2_theta

                # REM theta/alpha ratio
                if not np.isnan(rem_theta) and not np.isnan(rem_alpha) and rem_alpha > 0:
                    out['rem_theta_alpha_ratio'] = rem_theta / rem_alpha

                # N3 delta relative to wake delta (hardware-normalized SWA)
                if not np.isnan(n3_delta) and not np.isnan(wake_delta) and wake_delta > 0:
                    out['n3_delta_wake_delta_ratio'] = n3_delta / wake_delta

        # --- ECG HRV ---
        if ecg is not None and ecg_fs > 0:
            hrv_feats = _simple_ecg_hrv(ecg, ecg_fs)
            out['ecg_mean_hr'] = hrv_feats['mean_hr']
            out['ecg_hrv_proxy'] = hrv_feats['hrv_proxy']

    except Exception:
        pass

    return out


# =============================================================================
# UTILITIES
# =============================================================================

def _load_caisr_stages(caisr_path):
    """Load CAISR sleep stage array from EDF. Returns numpy array of stage codes."""
    if not os.path.exists(caisr_path):
        return None
    try:
        data_dict, fs_dict = load_signal_data(caisr_path)
        if not data_dict:
            return None
        labels = list(data_dict.keys())
        for lbl in labels:
            if 'stage' in lbl and 'prob' not in lbl:
                return data_dict[lbl]
        return None
    except Exception:
        return None


def _find_best_channel(processed_channels, processed_fs, candidates):
    for c in candidates:
        if c in processed_channels and processed_channels[c] is not None:
            return processed_channels[c], processed_fs.get(c, 1.0)
    return None, 1.0


def _sanitize(sig):
    if sig is None or len(sig) == 0:
        return sig
    mn, mx = np.min(sig), np.max(sig)
    if mx > 1.0001 or mn < -0.0001:
        d = mx - mn
        if d > 1e-6:
            sig = (sig - mn) / d
    return np.clip(sig, 0.0, 1.0)


def _count_events(arr, codes):
    if arr is None or len(arr) == 0:
        return 0
    b = np.isin(arr, codes).astype(int)
    d = np.diff(b)
    return max(np.sum(d == 1) + (1 if b[0] == 1 else 0), 0)


def _mean_bout_duration(stages, target, epoch_min=0.5):
    if stages is None or len(stages) == 0:
        return np.nan
    mask = (stages == target).astype(int)
    if np.sum(mask) == 0:
        return 0.0
    d = np.diff(mask, prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if len(starts) == 0:
        return 0.0
    durations = (ends - starts) * epoch_min
    return float(np.mean(durations)) if len(durations) > 0 else 0.0


def _arousal_cluster_ratio(arousal, min_bout_epochs=2):
    """
    Ratio of clustered arousals (in bouts of >= min_bout_epochs contiguous epochs)
    to total arousals. Arousal signal is at 0.5s resolution (2 samples per second),
    so min_bout_epochs corresponds to min_bout_epochs*0.5 seconds of contiguous arousal.
    But CAISR arousals are at 0.5s resolution, so we'll count in the raw signal.
    """
    if arousal is None or len(arousal) == 0:
        return np.nan
    try:
        b = (arousal == 1).astype(int)
        d = np.diff(b, prepend=0, append=0)
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        if len(starts) == 0:
            return 0.0
        bout_lens = ends - starts
        clustered = np.sum(bout_lens >= min_bout_epochs * 2)  # 0.5s resolution, so *2 per epoch
        total = len(starts)
        return float(clustered / total) if total > 0 else 0.0
    except Exception:
        return np.nan


def _eeg_spectrum(signal, fs):
    if signal is None or len(signal) == 0:
        return [np.nan] * len(BANDS)
    try:
        nperseg = min(4 * int(fs), len(signal))
        if nperseg < 2 * int(fs):
            nperseg = len(signal)
        freqs, psd = scipy.signal.welch(signal, fs, nperseg=nperseg, window='hann')
        ti = (freqs >= 0.5) & (freqs <= 30)
        tp = np.sum(psd[ti])
        if tp == 0 or np.isnan(tp):
            return [np.nan] * len(BANDS)
        powers = []
        for low, high in BANDS.values():
            p = np.sum(psd[(freqs >= low) & (freqs < high)]) / tp
            powers.append(p)
        return powers
    except Exception:
        return [np.nan] * len(BANDS)


def _resp_spectrum(signal, fs):
    if signal is None or len(signal) == 0:
        return [np.nan, np.nan]
    try:
        n = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        ri = (freqs >= 0.1) & (freqs <= 0.5)
        if np.sum(ri) == 0:
            return [np.nan, np.nan]
        pf = freqs[ri][np.argmax(fft_vals[ri])]
        ev = np.var(signal)
        return [float(pf), float(ev)]
    except Exception:
        return [np.nan, np.nan]


def _simple_ecg_hrv(signal, fs):
    out = {'mean_hr': np.nan, 'hrv_proxy': np.nan}
    if signal is None or len(signal) < fs * 30:
        return out
    try:
        sos = scipy.signal.butter(2, 0.5, btype='high', fs=fs, output='sos')
        filt = scipy.signal.sosfiltfilt(sos, signal)
        min_dist = int(0.4 * fs)
        prominence = np.std(filt) * 0.5
        peaks, _ = scipy.signal.find_peaks(filt, distance=min_dist, prominence=prominence)
        if len(peaks) < 5:
            return out
        rr = np.diff(peaks) / fs
        rr = rr[(rr > 0.3) & (rr < 2.0)]
        if len(rr) < 4:
            return out
        out['mean_hr'] = float(60.0 / np.mean(rr))
        out['hrv_proxy'] = float(np.std(rr) / np.mean(rr))
    except Exception:
        pass
    return out


def _age_auroc(y_true, y_prob, ages, delta=2.0):
    y_true, y_prob, ages = np.asarray(y_true), np.asarray(y_prob), np.asarray(ages)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    numer, denom = 0, 0
    for i in pos:
        for j in neg:
            if abs(ages[i] - ages[j]) <= delta:
                if y_prob[i] > y_prob[j]:
                    numer += 1
                elif y_prob[i] == y_prob[j]:
                    numer += 0.5
                denom += 1
    return numer / denom if denom > 0 else np.nan


def _make_fresh_model(name):
    if name == 'logistic':
        return LogisticRegression(
            C=0.1, penalty='l2', solver='liblinear',
            class_weight='balanced', max_iter=1000, random_state=42
        )
    elif name == 'ensemble':
        return VotingClassifier(
            estimators=[
                ('lr1', LogisticRegression(C=0.1, penalty='l2', solver='liblinear',
                                           class_weight='balanced', max_iter=1000, random_state=42)),
                ('lr2', LogisticRegression(C=0.05, penalty='l2', solver='liblinear',
                                           class_weight='balanced', max_iter=1000, random_state=43)),
            ],
            voting='soft', n_jobs=1
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

# #!/usr/bin/env python

# # Edit this script to add your team's code. Some functions are *required*, but you can edit most
# # parts of the required functions,
# # change or remove non-required functions, and add your own functions.
# """
# PhysioNet Challenge 2026 - Submission 7 ("v7: De-confound + Complete-the-Data")
# ================================================================================

# LINEAGE
# -------
# This is a revision of v1-v6, not a from-scratch rewrite. Internal LOSO-CV validation
# (site-aware, matching the hidden-site structure of validation/test) plus the actual
# scored results across v1-v6 pointed at two concrete, mechanistic problems, both of
# which this version fixes directly:

#   1. AGE LEAKAGE. Every version that beat v1's Age-Conditioned AUROC (0.665) on the
#      scored validation set fed the model some form of raw age (age, age^2, or an
#      age-by-feature interaction) as a direct input. v2 (raw age + age^2 + 8
#      interactions, no residualization) scored 0.537 - barely above chance - despite
#      having the *best* raw/unconditioned AUROC (0.742) of any version, which is the
#      signature of a model leaning on age as a shortcut rather than learning
#      physiology. v3.1 and v4 partially reintroduced the same leak (residualized
#      everything except age/age^2/age-interactions) and landed in between (0.646,
#      0.613). v1 and v5 never exposed raw age to the model and were the two best
#      scores (0.665, 0.620). v7 removes every raw-age-derived column from the
#      feature matrix; age is used ONLY (a) as the covariate for residualizing other
#      features and (b) for age-conditioned pairing during internal validation and
#      model selection - never as a value the model can read directly.

#   2. OVER-TRUSTING A 3-SITE INTERNAL VALIDATION SIGNAL. Training sites are S0001,
#      I0002, I0006 - only 3 site-folds for Leave-One-Site-Out CV, with wildly
#      different sizes (I0002 can be <100 records). v2's reward-threshold search and
#      v5's softmax-weighted ensemble both tuned something *against* this noisy
#      3-fold signal and both generalized worse than v1's simpler "pick one winning
#      model type via LOSO" strategy. v7 keeps v1's simpler selection rule, but
#      computes it more carefully (see below) and adds a complexity tie-break that
#      prefers the simpler candidate when scores are close, rather than adding more
#      tunable machinery on top of 3 noisy folds.

# v7 also fixes a real (if quieter) methodological issue present even in v1: v1 fits
# its age-residualization on the *entire* training set before running LOSO-CV, so the
# internal validation loop leaks a small amount of held-out-site information into
# preprocessing. v7 refits residualization/imputation/scaling from scratch inside each
# LOSO fold using only that fold's training sites, then fits everything on 100% of the
# data for the artifact that is actually saved (which is correct practice, not
# leakage - the real held-out site is never touched at any point).

# NEW DATA v1-v6 NEVER USED (confirmed against the Challenge website's data
# description and the organizers' own team_code.py example):
#   - EOG. The Challenge site lists EOG as an available signal, useful for
#     identifying REM and brain-state transitions; the official example derives
#     e1-m2/e2-m1 exactly the way this file does. No v1-v6 submission touched it.
#   - Limb EMG (LAT/RAT). Only CAISR's summarized limb-movement counts were used;
#     the raw leg leads were never read.
#   - A second EEG derivation. Only one central channel was ever used, even though
#     frontal slow-wave activity specifically (not just central) has an established
#     literature link to cognitive aging. v7 adds a small, targeted frontal-central
#     contrast rather than a full duplicate feature block, to avoid the
#     dimensionality blowup that hurt v2/v6.
#   - Ethnicity (helper_code.load_ethnicity exists, was never called).
#   - The official channel-standardization/bipolar-derivation utilities in
#     helper_code (load_rename_rules / standardize_channel_names_rename_only /
#     derive_bipolar_signal) that read channel_table.csv. v1-v6 all used a hand-rolled
#     substring-matching manifest that silently mixes referential and bipolar
#     montages across sites - exactly the kind of cross-site inconsistency the
#     Challenge's own FAQ names as the central generalization problem ("Why are the
#     training/validation/test data drawn from slightly different populations?").
#     v7 uses the official utilities, with the old manifest kept only as a fallback
#     if channel_table.csv is missing.
#   - Philosopher's Stone (bdsp-core/philosophers-stone, Ganglberger et al. 2026,
#     NEJM AI). This is a small, cheap, purpose-built brain-health/cognition model
#     (not a general sleep-staging model like SleepFM) trained on 36,000 PSGs
#     across 6 cohorts, posted to the Challenge's own forum by a CAISR co-author on
#     2026-07-07. It was not available to earlier submissions' authors in any
#     actionable way. v7 uses ONLY its 4 scalar clinical outputs (brain_health_score,
#     total_cognition_score, fluid_cognition_score, crystallized_cognition_score),
#     not its raw 1024-d latent, specifically to avoid repeating v6's mistake of
#     dumping hundreds of unvetted embedding dimensions into a model trained on a
#     few thousand records at most. See USE_PHILOSOPHERS_STONE below and README.md
#     for compute/licensing/compliance notes - this block is fully optional and
#     fails soft (never crashes the pipeline) so it can be switched off with one
#     flag if it proves too slow or the checkpoint isn't available in the runner.

#     v6's own SleepFM integration is NOT reused here. Its own log line
#     ("SleepFM model loaded successfully." vs "SleepFM model NOT loaded.") should
#     still be in your v6 training output - check that first, since it directly
#     disambiguates why v6 scored 0.496: either the checkpoint never loaded (in
#     which case v6 was functionally v1 plus ~300 dead zero columns) or it loaded
#     and its high-dimensional embedding leaked age through nonlinear feature
#     combinations that a per-column *linear* residualization can't remove. Either
#     diagnosis argues for what v7 does: fewer, lower-dimensional, better-vetted
#     additions instead of a raw embedding dump. See README.md for the full
#     writeup.

# METRIC IMPLEMENTATION NOTE
# ---------------------------
# The Challenge website's written scoring section states the age-conditioning window
# is delta = 2 years for both metrics. The current physionetchallenges/python-example-2026
# evaluate_model.py instead calls compute_auroc_age(..., gap=1) (and the reward/
# prevalence helpers with gap=1). That is a real, verifiable discrepancy between the
# prose spec and the reference code as of 2026-07-18, not a bug in this file. Because
# ranking is decided by the age-conditioned AUROC, this file's internal validation
# computes and prints BOTH gap=1 and gap=2 for every candidate, selects using gap=2
# (matching the written definition, which is what determines "the winners"), and
# flags the discrepancy in the printed diagnostics so you see both numbers before
# choosing. Consider asking the forum to confirm which one the scoring server
# actually runs - it's exactly the kind of question the organizers say they want.

# Both age-conditioned AUROC and the prevalence-based reward metric were re-derived
# line-by-line against the actual evaluate_model.py in physionetchallenges/python-
# example-2026 (not just the website prose), including the 0.5-credit tie handling.
# Reward is NOT the ranking metric (confirmed on the Challenge forum, 2026-07-07:
# "We will use the age-conditioned AUROC metric to determine the final rankings") and
# is threshold-dependent while AUROC is not, so no effort in this file goes into
# threshold tuning beyond a documented default - that was wasted effort in v2 with
# respect to the metric that actually decides rankings.
# """

# import os
# import warnings

# import numpy as np
# import pandas as pd
# import joblib

# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import StackingClassifier, VotingClassifier
# from sklearn.svm import SVC

# warnings.filterwarnings("ignore")

# try:
#     import lightgbm as lgb
#     _HAVE_LGB = True
# except Exception:
#     _HAVE_LGB = False

# from helper_code import *  # noqa: F401,F403  (official Challenge I/O: find_patients, load_demographics,
#                             # load_diagnoses, load_age, load_sex, load_race, load_bmi, load_ethnicity,
#                             # load_signal_data, HEADERS, DEMOGRAPHICS_FILE, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
#                             # PHYSIOLOGICAL_DATA_SUBFOLDER, load_rename_rules, standardize_channel_names_rename_only,
#                             # derive_bipolar_signal)

# # =====================================================================================
# # CONFIGURATION
# # =====================================================================================

# # --- Philosopher's Stone (optional; see README.md before flipping this on for a real
# #     submission - it needs its ~2.3GB checkpoint baked into the Docker image at BUILD
# #     time [no network access exists during train_model/run_model], and you likely need
# #     to request a GPU on the submission form for it to run in reasonable time on the
# #     full training set). Fails soft: if unavailable, every downstream feature from it
# #     is NaN -> median-imputed, and the rest of the pipeline is unaffected. ---
# USE_PHILOSOPHERS_STONE = True
# PHI_STONE_TARGET_FS = 200  # required input rate for the model

# # --- Age-conditioning window: see METRIC IMPLEMENTATION NOTE above. ---
# AGE_GAP_PRIMARY = 2.0    # matches the Challenge website's written definition
# AGE_GAP_SECONDARY = 1.0  # matches the current python-example-2026 evaluate_model.py call

# # --- Candidate model zoo. Deliberately the same small set v1 used (not the larger
# #     zoo v2/v3.1/v4 added) - LOSO selection is only as reliable as the 3 folds voting
# #     on it, and more candidates means more chances to pick a fold-specific fluke. ---
# CANDIDATE_MODELS = ['logistic', 'lightgbm', 'stack', 'mega'] if _HAVE_LGB else ['logistic', 'stack', 'mega']
# MODEL_COMPLEXITY_RANK = {'logistic': 0, 'lightgbm': 1, 'stack': 2, 'mega': 3}
# LOSO_TIE_EPSILON = 0.01  # if candidates are within this on weighted gap-2 AUROC, prefer the simpler one

# WINDOWS = ['early', 'mid', 'late']

# EEG_METRICS = [
#     'delta_power', 'theta_power', 'alpha_power', 'sigma_power', 'beta_power',
#     'total_power', 'delta_rel', 'theta_rel', 'alpha_rel', 'spindle_density',
#     'spectral_edge95', 'slowing_ratio',
# ]

# # Channel candidate priority lists (used both as fallback manifest AND to pick which
# # standardized-name channel to read after official renaming/derivation).
# EEG_CENTRAL_CANDIDATES = ['c4-m1', 'c3-m2', 'c4', 'c3', 'eeg']
# EEG_FRONTAL_CANDIDATES = ['f4-m1', 'f3-m2', 'f4', 'f3']
# EOG_CANDIDATES = ['e1-m2', 'e2-m1', 'e1', 'e2', 'eog', 'loc', 'roc']
# CHIN_EMG_CANDIDATES = ['chin1-chin2', 'chin', 'emg']
# LEG_EMG_CANDIDATES = ['lat', 'rat', 'lleg', 'rleg', 'leg']
# ECG_CANDIDATES = ['ecg', 'ekg']
# RESP_AIRFLOW_CANDIDATES = ['airflow', 'flow', 'thermal', 'thermistor', 'nasal_pressure', 'ptaf']
# RESP_EFFORT_CANDIDATES = ['abd', 'abdomen', 'chest', 'thorax', 'effort abd', 'effort tho']
# SPO2_CANDIDATES = ['spo2', 'sao2', 'sat', 'o2sat']

# CHANNEL_TABLE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'channel_table.csv')

# # Demographic features used DIRECTLY (not age-residualized - the metric conditions on
# # age, not on these, so there is no shortcut risk in using them raw the way there is for
# # age). 'age' itself is deliberately NOT in this list or in any feature list below: it
# # is tracked as a separate variable for the whole pipeline and never becomes a model
# # input. See docstring "AGE LEAKAGE" above.
# DEMOGRAPHIC_DIRECT_COLS = [
#     'sex_male', 'sex_unknown',
#     'race_white', 'race_black', 'race_asian', 'race_unavailable', 'race_other',
#     'ethnicity_hispanic', 'bmi',
# ]

# # Modality-level missingness flags (direct features, not residualized - they're binary
# # and their informativeness is about which channels a site/protocol provided, which is
# # exactly the cross-site heterogeneity the Challenge FAQ names as the core difficulty).
# MISSINGNESS_COLS = [
#     'is_missing_eeg_central', 'is_missing_eeg_frontal', 'is_missing_eog',
#     'is_missing_chin_emg', 'is_missing_leg_emg', 'is_missing_ecg',
#     'is_missing_resp', 'is_missing_spo2',
# ]

# # Domain-informed interaction terms (unchanged from v1 - proven, clinically motivated,
# # and deliberately NOT expanded with age interactions, unlike v2/v3.1/v4).
# INTERACTION_PAIRS = [
#     ('resp_caisr_ahi', 'spo2_drop_mean', 'inter_ahi_spo2'),
#     ('stage_caisr_waso_min', 'eeg_c_slow_wave_activity', 'inter_waso_swa'),
#     ('caisr_prob_r_mean', 'emg_chin_rms_mean', 'inter_rem_emgatonia'),
#     ('ecg_hrv_proxy_mean', 'resp_effort_rms_mean', 'inter_hrv_respeffort'),
#     ('arousal_caisr_rate', 'stage_caisr_se', 'inter_arousal_se'),
#     ('resp_caisr_ahi', 'stage_caisr_n3_pct_tst', 'inter_ahi_n3'),
#     ('spo2_drop_mean', 'eeg_c_delta_rel', 'inter_spo2_delta'),
# ]


# # =====================================================================================
# # REQUIRED CHALLENGE ENTRY POINTS
# # =====================================================================================

# def train_model(data_folder, model_folder, verbose):
#     """
#     Required Challenge function. Trains on data_folder, saves an artifact to model_folder.
#     """
#     if verbose:
#         print('Finding Challenge data...')

#     # find_patients(patient_data_file) takes the demographics.csv PATH, not the data
#     # folder (confirmed against the real helper_code.py) - patient_data_file has to be
#     # built first.
#     patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
#     records = find_patients(patient_data_file)
#     if len(records) == 0:
#         raise FileNotFoundError('No data were provided.')
#     if verbose:
#         print(f'Found {len(records)} patient/session records.')

#     demo_df = pd.read_csv(patient_data_file)  # loaded ONCE; official load_demographics()/
#                                                # load_diagnoses() re-read this file from disk
#                                                # on every single call, which is fine for one
#                                                # record but is an O(n) x O(file size) cost
#                                                # across a few thousand records. We keep using
#                                                # the official *parsing* logic (load_age, load_sex,
#                                                # load_race, load_bmi, load_ethnicity all operate on
#                                                # an in-memory dict and have no I/O cost), we just
#                                                # stop re-reading the CSV per patient.

#     phi_extractor = _PhilosophersStoneExtractor(verbose=verbose) if USE_PHILOSOPHERS_STONE else None
#     if verbose and USE_PHILOSOPHERS_STONE:
#         status = 'loaded successfully' if (phi_extractor is not None and phi_extractor.available) else 'NOT loaded (falling back to NaN for its features)'
#         print(f"Philosopher's Stone model {status}.")

#     rename_rules = None
#     if os.path.exists(CHANNEL_TABLE_CSV):
#         try:
#             rename_rules = load_rename_rules(CHANNEL_TABLE_CSV)
#             if verbose:
#                 print('Loaded official channel_table.csv renaming rules.')
#         except Exception as e:
#             if verbose:
#                 print(f'Could not load channel_table.csv ({e}); falling back to manifest matching.')
#     elif verbose:
#         print('channel_table.csv not found next to team_code.py; falling back to manifest matching. '
#               'Copy it from the python-example-2026 repo for more robust cross-site channel handling.')

#     rows = []
#     n_ok, n_fail = 0, 0
#     for i, rec in enumerate(records):
#         patient_id = rec[HEADERS['bids_folder']]
#         site_id = rec[HEADERS['site_id']]
#         session_id = rec[HEADERS['session_id']]
#         if verbose and (i % 200 == 0):
#             print(f'  Extracting features: {i}/{len(records)} ({n_ok} ok, {n_fail} failed)...')
#         try:
#             patient_data = _lookup_patient_row(demo_df, patient_id, session_id)
#             if not patient_data:
#                 raise ValueError('Patient row not found in demographics.csv.')
#             label = _lookup_label(demo_df, patient_id)
#             age = load_age(patient_data)
#             if age is None or not np.isfinite(age):
#                 raise ValueError('Missing/invalid age.')

#             feats = extract_all_features(
#                 data_folder, patient_id, site_id, session_id, patient_data,
#                 rename_rules=rename_rules, phi_extractor=phi_extractor, age=age,
#             )
#             feats['label'] = int(label)
#             feats['age'] = float(age)
#             feats['site'] = site_id
#             rows.append(feats)
#             n_ok += 1
#         except Exception as e:
#             n_fail += 1
#             if verbose:
#                 print(f'  Skipping {patient_id}/{session_id}: {e}')
#             continue

#     if verbose:
#         print(f'Feature extraction complete: {n_ok} ok, {n_fail} skipped.')
#     if n_ok < 20:
#         raise RuntimeError(f'Only {n_ok} usable records; cannot train reliably.')

#     df = pd.DataFrame(rows)
#     del rows

#     raw_feature_cols = [c for c in df.columns if c not in ('label', 'age', 'site')]

#     # --- Site-directionality ("poison") filtering: drop any feature whose correlation
#     #     with the label flips sign across training sites. A feature that helps at one
#     #     site and hurts at another is very unlikely to be real physiology and very
#     #     likely to be a site/protocol artifact - exactly the kind of thing that will not
#     #     transfer to the hidden validation/test sites. Unchanged from v1. ---
#     kept_raw_cols = _poison_filter(df, raw_feature_cols, verbose=verbose)

#     resid_candidate_cols = [c for c in kept_raw_cols if c not in DEMOGRAPHIC_DIRECT_COLS + MISSINGNESS_COLS]
#     direct_cols = [c for c in kept_raw_cols if c in DEMOGRAPHIC_DIRECT_COLS + MISSINGNESS_COLS]

#     sites = sorted(df['site'].unique().tolist())
#     if verbose:
#         print(f'Training sites: {sites} (sizes: {df["site"].value_counts().to_dict()})')

#     # --- Honest LOSO-CV model selection: preprocessing (age-residualization, imputer,
#     #     scaler) is refit from scratch on each fold's training sites only, then applied
#     #     to the held-out site. This is stricter than v1 (which fit residualization once
#     #     on all data before the LOSO loop) and mirrors what v3.1 *intended* to fix,
#     #     implemented without also leaking raw age back in the way v3.1 did. ---
#     loso_summary = {}
#     if len(sites) >= 2:
#         loso_summary = _run_loso_evaluation(
#             df, resid_candidate_cols, direct_cols, INTERACTION_PAIRS, sites, verbose=verbose,
#         )
#     best_model_name = _select_best_model(loso_summary, verbose=verbose)

#     # --- Final artifact: preprocessing + model fit on 100% of the provided training
#     #     data. This is standard practice, not leakage - the real held-out site is
#     #     never touched during training regardless of how preprocessing is fit. ---
#     prep, X_full = _fit_preprocessing(df, resid_candidate_cols, direct_cols, INTERACTION_PAIRS)
#     y_full = df['label'].values
#     final_model = _make_fresh_model(best_model_name)
#     final_model.fit(X_full, y_full)

#     os.makedirs(model_folder, exist_ok=True)
#     artifact = {
#         'model': final_model,
#         'model_name': best_model_name,
#         'prep': prep,
#         'raw_feature_cols': raw_feature_cols,
#         'kept_raw_cols': kept_raw_cols,
#         'resid_candidate_cols': resid_candidate_cols,
#         'direct_cols': direct_cols,
#         'rename_rules': rename_rules,
#         'use_philosophers_stone': bool(USE_PHILOSOPHERS_STONE),
#         'training_sites': sites,
#         'loso_summary': loso_summary,
#         'age_gap_primary': AGE_GAP_PRIMARY,
#         'age_gap_secondary': AGE_GAP_SECONDARY,
#     }
#     joblib.dump(artifact, os.path.join(model_folder, 'model.sav'))

#     if verbose:
#         print(f'Selected model: {best_model_name}')
#         s2_train = _age_auroc(y_full, final_model.predict_proba(X_full)[:, 1], df['age'].values, gap=AGE_GAP_PRIMARY)
#         print(f'  In-sample (training data) age-conditioned AUROC (gap=2): {s2_train:.4f} '
#               f'(optimistic - see loso_summary for the honest held-out-site estimate)')
#         print('Training complete.')


# def load_model(model_folder, verbose):
#     """Required Challenge function."""
#     return joblib.load(os.path.join(model_folder, 'model.sav'))


# def run_model(model_artifact, record, data_folder, verbose):
#     """
#     Required Challenge function. `record` is (patient_id, site_id, session_id) or a dict
#     with those keys, depending on how the caller structures it - handled defensively below
#     to match whichever convention run_challenge_models.py in this Challenge year uses.
#     """
#     patient_id, site_id, session_id = _unpack_record(record)

#     patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
#     patient_data = load_demographics(patient_data_file, patient_id, session_id)
#     age = load_age(patient_data)

#     rename_rules = model_artifact.get('rename_rules', None)
#     phi_extractor = None
#     if model_artifact.get('use_philosophers_stone', False):
#         phi_extractor = _get_shared_phi_extractor(verbose=verbose)

#     try:
#         feats = extract_all_features(
#             data_folder, patient_id, site_id, session_id, patient_data,
#             rename_rules=rename_rules, phi_extractor=phi_extractor, age=age,
#         )
#     except Exception as e:
#         if verbose:
#             print(f'Feature extraction failed for {patient_id}/{session_id}: {e}; using an all-missing row.')
#         feats = {}

#     row = {c: feats.get(c, np.nan) for c in model_artifact['raw_feature_cols']}
#     row['age'] = age if (age is not None and np.isfinite(age)) else np.nan
#     df_row = pd.DataFrame([row])

#     X = _apply_preprocessing(df_row, model_artifact['prep'])
#     prob = float(model_artifact['model'].predict_proba(X)[:, 1][0])

#     # Threshold: the ranking metric (age-conditioned AUROC) is computed directly on the
#     # continuous probability and is threshold-invariant (confirmed on the Challenge
#     # forum, 2026-07-07: it "evaluates a model across operating points from the models'
#     # real-valued outputs"). 0.5 is used only to produce the required binary output and
#     # has no bearing on the metric that decides rankings, so no effort goes into tuning
#     # it here (unlike v2, which spent real effort on threshold search for a metric that
#     # doesn't decide the outcome).
#     binary_pred = 1 if prob >= 0.5 else 0
#     return binary_pred, prob


# # =====================================================================================
# # FEATURE EXTRACTION
# # =====================================================================================

# def extract_all_features(data_folder, patient_id, site_id, session_id, patient_data,
#                           rename_rules, phi_extractor, age):
#     """
#     Builds the raw (pre-interaction, pre-residualization) feature dict for one record.
#     `age` is accepted as an argument only so it can be threaded through to Philosopher's
#     Stone (which takes age/sex as its own model inputs) - it is deliberately NOT written
#     into the returned dict, so it can never end up in the model's feature matrix. The
#     caller (train_model/run_model) tracks age separately.
#     """
#     features = {}

#     # --- Demographics (direct, unresidualized; see DEMOGRAPHIC_DIRECT_COLS) ---
#     sex = load_sex(patient_data, standardize=True)
#     features['sex_male'] = 1 if sex == 'Male' else 0
#     features['sex_unknown'] = 1 if sex not in ('Male', 'Female') else 0

#     race = load_race(patient_data, standardize=True)
#     features['race_white'] = 1 if race == 'White' else 0
#     features['race_black'] = 1 if race == 'Black' else 0
#     features['race_asian'] = 1 if race == 'Asian' else 0
#     features['race_unavailable'] = 1 if race == 'Unavailable' else 0
#     features['race_other'] = 1 if race == 'Others' else 0

#     try:
#         ethnicity = load_ethnicity(patient_data, standardize=True)
#     except Exception:
#         ethnicity = 'Unavailable'
#     features['ethnicity_hispanic'] = 1 if ethnicity == 'Hispanic' else 0

#     try:
#         bmi = load_bmi(patient_data)
#         features['bmi'] = float(bmi) if bmi is not None and np.isfinite(bmi) else np.nan
#     except Exception:
#         features['bmi'] = np.nan

#     # --- CAISR algorithmic annotations (sleep stage / arousal / respiratory / limb) ---
#     anno_path = _find_annotation_edf(data_folder, patient_id, site_id, session_id)
#     caisr_feats = _extract_caisr(anno_path)
#     features.update(caisr_feats)

#     # --- Physiological signals (EEG x2 derivations, EOG, chin EMG, leg EMG, ECG, resp
#     #     airflow, resp effort, SpO2) ---
#     sig_path = _find_signal_edf(data_folder, patient_id, site_id, session_id)
#     physio_feats, chans_used = _extract_physio(sig_path, rename_rules)
#     features.update(physio_feats)

#     # --- Missingness-by-modality indicators ---
#     for key, col in [('eeg_central', 'is_missing_eeg_central'), ('eeg_frontal', 'is_missing_eeg_frontal'),
#                       ('eog', 'is_missing_eog'), ('chin_emg', 'is_missing_chin_emg'),
#                       ('leg_emg', 'is_missing_leg_emg'), ('ecg', 'is_missing_ecg'),
#                       ('resp', 'is_missing_resp'), ('spo2', 'is_missing_spo2')]:
#         features[col] = 0 if chans_used.get(key, False) else 1

#     # --- Philosopher's Stone (optional; NaN-safe) ---
#     if phi_extractor is not None:
#         phi_feats = phi_extractor.extract(sig_path, chans_used.get('eeg_central_signal'),
#                                            chans_used.get('eeg_central_fs'), age=age,
#                                            sex_code=(1 if sex == 'Male' else 0))
#         features.update(phi_feats)

#     return _sanitize(features)


# def _extract_caisr(anno_path):
#     """
#     CAISR encoding (per the Challenge data description): sleep stage 1=N3, 2=N2, 3=N1,
#     4=REM, 5=Wake, 9=Unavailable; arousal 0/1 (+ probabilities); respiratory events
#     0=None,1=OA,2=CA,3=MA,4=HY,5=RERA (class only - CAISR's respiratory/limb models are
#     rule-based, no probabilities exist for them); limb 0=None,1=Isolated,2=Periodic.
#     """
#     out = {
#         'stage_caisr_trt_hr': np.nan, 'stage_caisr_se': np.nan,
#         'stage_caisr_waso_min': np.nan, 'stage_caisr_rem_latency_min': np.nan,
#         'stage_caisr_n3_pct_tst': np.nan, 'stage_caisr_transition_rate': np.nan,
#         'stage_caisr_softmax_entropy_mean': np.nan,
#         'caisr_prob_n3_mean': np.nan, 'caisr_prob_n2_mean': np.nan, 'caisr_prob_n1_mean': np.nan,
#         'caisr_prob_r_mean': np.nan, 'caisr_prob_w_mean': np.nan,
#         'arousal_caisr_rate': np.nan, 'caisr_prob_arousal_mean': np.nan, 'high_conf_arousal_rate': np.nan,
#         'resp_caisr_ahi': np.nan, 'resp_central_ratio': np.nan,
#         'limb_caisr_rate': np.nan, 'limb_isolated_rate': np.nan, 'limb_periodic_rate': np.nan,
#     }
#     if anno_path is None or not os.path.exists(anno_path):
#         return out
#     try:
#         sig_dict, fields_dict = load_edf_to_nparrays_safe(anno_path)
#     except Exception:
#         return out

#     stages = _find_epoch_series(sig_dict, ['stage_caisr', 'stage'])
#     if stages is not None:
#         stages = np.asarray(stages, dtype=float)
#         valid = stages[stages != 9]
#         n_epochs = len(valid)
#         if n_epochs > 0:
#             out['stage_caisr_trt_hr'] = float(n_epochs * 30.0 / 3600.0)
#             asleep = np.isin(valid, [1, 2, 3, 4])
#             out['stage_caisr_se'] = float(np.mean(asleep))
#             sleep_idx = np.where(asleep)[0]
#             if len(sleep_idx) > 0:
#                 onset = sleep_idx[0]
#                 post_onset = valid[onset:]
#                 out['stage_caisr_waso_min'] = float(np.sum(post_onset == 5) * 30.0 / 60.0)
#                 rem_idx = np.where(post_onset == 4)[0]
#                 if len(rem_idx) > 0:
#                     out['stage_caisr_rem_latency_min'] = float(rem_idx[0] * 30.0 / 60.0)
#                 n_tst = int(np.sum(asleep))
#                 if n_tst > 0:
#                     out['stage_caisr_n3_pct_tst'] = float(np.sum(valid == 1) / n_tst)
#             if n_epochs > 1:
#                 out['stage_caisr_transition_rate'] = float(np.mean(np.diff(valid) != 0))

#     probs = {}
#     for key in ['n3', 'n2', 'n1', 'r', 'w']:
#         p = _find_epoch_series(sig_dict, [f'caisr_prob_{key}', f'prob_{key}'])
#         if p is not None:
#             probs[key] = np.asarray(p, dtype=float)
#             out[f'caisr_prob_{key}_mean'] = float(np.nanmean(probs[key]))
#     if probs:
#         stacked = np.vstack([probs[k] for k in probs if k in probs and len(probs[k]) == len(next(iter(probs.values())))]) \
#             if len({len(v) for v in probs.values()}) == 1 else None
#         if stacked is not None and stacked.shape[0] > 1:
#             p_clipped = np.clip(stacked, 1e-9, 1.0)
#             ent = -np.sum(p_clipped * np.log(p_clipped), axis=0)
#             out['stage_caisr_softmax_entropy_mean'] = float(np.nanmean(ent))

#     arousal = _find_epoch_series(sig_dict, ['arousal_caisr'], exclude_substr=['prob', 'no-ar'])
#     if arousal is not None:
#         arousal = np.asarray(arousal, dtype=float)
#         hours = max(len(arousal) * 30.0 / 3600.0, 1e-6)
#         out['arousal_caisr_rate'] = float(np.sum(arousal > 0) / hours)
#     prob_arousal = _find_epoch_series(sig_dict, ['caisr_prob_arousal', 'prob_arousal'])
#     if prob_arousal is not None:
#         prob_arousal = np.asarray(prob_arousal, dtype=float)
#         out['caisr_prob_arousal_mean'] = float(np.nanmean(prob_arousal))
#         out['high_conf_arousal_rate'] = float(np.mean(prob_arousal > 0.8))

#     resp = _find_epoch_series(sig_dict, ['resp_caisr', 'resp'])
#     if resp is not None:
#         resp = np.asarray(resp, dtype=float)
#         hours = max(len(resp) * 30.0 / 3600.0, 1e-6)  # CAISR resp/limb are commonly 1s-resolution;
#         # if so this under-counts hours by 30x, but AHI is a *ratio* of event-count to
#         # hours, and both this constant and the 30s-epoch case cancel consistently as
#         # long as the same convention is used across all records, which it is here.
#         out['resp_caisr_ahi'] = float(np.sum(np.isin(resp, [1, 2, 3, 4])) / hours)
#         n_events = np.sum(np.isin(resp, [1, 2, 3, 4]))
#         if n_events > 0:
#             out['resp_central_ratio'] = float(np.sum(resp == 2) / n_events)

#     limb = _find_epoch_series(sig_dict, ['limb_caisr', 'limb'])
#     if limb is not None:
#         limb = np.asarray(limb, dtype=float)
#         hours = max(len(limb) * 30.0 / 3600.0, 1e-6)
#         out['limb_caisr_rate'] = float(np.sum(limb > 0) / hours)
#         out['limb_isolated_rate'] = float(np.sum(limb == 1) / hours)
#         out['limb_periodic_rate'] = float(np.sum(limb == 2) / hours)

#     return out


# # =====================================================================================
# # SIGNAL / CHANNEL UTILITIES
# # =====================================================================================
# # NOTE ON ROBUSTNESS: the official helper_code.py (as of 2026-07) exposes
# # load_rename_rules / standardize_channel_names_rename_only / derive_bipolar_signal for
# # exactly this cross-site channel-naming problem, and _get_channel below tries to use
# # them first. Function names/signatures in helper_code can shift between Challenge
# # refreshes, so every official call here is wrapped and falls back to the plain
# # substring-manifest approach v1-v6 used if anything about the official path doesn't
# # match what's actually in your local helper_code.py. Before you submit, run this file
# # once on a handful of training records with verbose=True and confirm you see
# # "using official channel standardization" rather than "falling back to manifest
# # matching" in the log if you want the more robust path active - if your local
# # helper_code.py's function signatures differ, this degrades to v1's behavior
# # automatically rather than crashing, but you'd lose the cross-site robustness benefit.

# def load_edf_to_nparrays_safe(path):
#     """Wraps whichever official EDF-loading function is available in this Challenge
#     year's helper_code.py. Returns (signal_dict[name->float array], fields_dict[name->fs])."""
#     for fn_name in ('load_signal_data', 'load_edf_to_nparrays'):
#         fn = globals().get(fn_name, None)
#         if fn is not None:
#             try:
#                 return fn(path)
#             except Exception:
#                 continue
#     raise RuntimeError('No usable EDF-loading function found in helper_code.')


# def _standardize_channels_safe(raw_names, rename_rules):
#     """Returns {standardized_name: raw_name} if the official standardizer is available
#     and works; otherwise returns {lowercased_raw_name: raw_name} unchanged.

#     NOTE: the real helper_code.py's standardize_channel_names_rename_only returns a
#     TUPLE (rename_map, cols_to_drop), where rename_map is {Original Raw Name: New
#     Standard Name} - the opposite direction from what this function needs to hand
#     back to _get_channel (which wants {standard_name: raw_name} so it can look a
#     candidate name up directly). This inverts it accordingly. cols_to_drop
#     (duplicate-alias columns the official function flags) is intentionally unused
#     here - this file never mutates or drops raw channels, it only ever *reads* from
#     sig_dict by name, so a channel being on that list has no effect on correctness,
#     just a minor missed tidiness optimization."""
#     fn = globals().get('standardize_channel_names_rename_only', None)
#     if fn is not None and rename_rules is not None:
#         try:
#             result = fn(list(raw_names), rename_rules)
#             rename_map = result[0] if isinstance(result, tuple) else result
#             if isinstance(rename_map, dict) and len(rename_map) > 0:
#                 inverted = {std_name: raw_name for raw_name, std_name in rename_map.items()}
#                 return inverted
#         except Exception:
#             pass
#     return {str(n).strip().lower(): n for n in raw_names}


# def _derive_bipolar_safe(sig_dict, fields_dict, pos_key, neg_keys):
#     """Wraps the official derive_bipolar_signal if present; falls back to manual
#     subtraction (averaging multiple reference channels if more than one is given)."""
#     if pos_key not in sig_dict:
#         return None, None
#     pos_sig = sig_dict[pos_key]
#     pos_fs = fields_dict.get(pos_key, None)
#     present_negs = [k for k in neg_keys if k in sig_dict]
#     if not present_negs:
#         return None, None
#     fn = globals().get('derive_bipolar_signal', None)
#     if fn is not None:
#         try:
#             ref = sig_dict[present_negs[0]] if len(present_negs) == 1 else tuple(sig_dict[k] for k in present_negs)
#             derived = fn(pos_sig, ref)
#             return derived, pos_fs
#         except Exception:
#             pass
#     try:
#         ref_stack = np.mean([np.asarray(sig_dict[k])[:len(pos_sig)] for k in present_negs], axis=0)
#         n = min(len(pos_sig), len(ref_stack))
#         return np.asarray(pos_sig)[:n] - ref_stack[:n], pos_fs
#     except Exception:
#         return None, None


# def _get_channel(sig_dict, fields_dict, candidates, rename_rules=None):
#     """
#     Finds a channel matching a priority list of candidate names/substrings. Tries, in
#     order, for each candidate: (1) a direct hit through the official channel-alias map
#     (handles a channel that's already combined/bipolar but filed under a non-standard
#     name, e.g. raw 'C4-A1' aliased to standard 'c4-m1' in channel_table.csv); (2) a
#     direct substring match against raw channel names (handles EDFs that already ship a
#     combined/bipolar channel under a name that happens to match); (3) only if the
#     candidate itself looks like a bipolar pair (e.g. 'c4-m1') and neither of the above
#     fired, deriving it from separate positive/negative electrodes via official bipolar
#     derivation. This order matters: trying (3) before (1)/(2) can miss a channel that's
#     only reachable through the alias map, since a candidate like 'c4-m1' splitting into
#     pos='c4'/neg='m1' has no reason to find raw electrodes named something else
#     entirely - the alias map has to be checked as a whole-candidate lookup first.
#     Returns (signal_array, sampling_rate) or (None, None).
#     """
#     if sig_dict is None or len(sig_dict) == 0:
#         return None, None
#     raw_names = list(sig_dict.keys())
#     std_map = _standardize_channels_safe(raw_names, rename_rules)  # {standard_name: raw_name}
#     lower_sig = {str(k).strip().lower(): v for k, v in sig_dict.items()}
#     lower_fields = {str(k).strip().lower(): v for k, v in fields_dict.items()} if fields_dict else {}

#     for cand in candidates:
#         cand_l = cand.lower()

#         # (1) direct hit via the official alias map, keyed on the WHOLE candidate.
#         if cand_l in std_map and std_map[cand_l] in sig_dict:
#             raw_name = std_map[cand_l]
#             sig = sig_dict[raw_name]
#             fs = fields_dict.get(raw_name, None) if fields_dict else None
#             if sig is not None and len(sig) > 0:
#                 return np.asarray(sig, dtype=float), fs

#         # (2) direct substring match against raw channel names.
#         for name_l in lower_sig.keys():
#             if cand_l == name_l or cand_l in name_l:
#                 sig = lower_sig[name_l]
#                 fs = lower_fields.get(name_l, None)
#                 if sig is not None and len(sig) > 0:
#                     return np.asarray(sig, dtype=float), fs

#         # (3) bipolar derivation from separate electrodes, e.g. 'c4-m1' -> pos='c4', neg=['m1'].
#         # Electrode components are looked up through BOTH the alias map and raw lowercased
#         # names, since a single electrode (e.g. bare 'm1') could be aliased too.
#         if '-' in cand_l:
#             pos, neg = cand_l.split('-', 1)
#             neg_list = [n.strip() for n in neg.split('+')]
#             pool = {**lower_sig}
#             pool_fields = {**lower_fields}
#             for std_name, raw_name in std_map.items():
#                 if raw_name in sig_dict and std_name not in pool:
#                     pool[std_name] = sig_dict[raw_name]
#                     if fields_dict and raw_name in fields_dict:
#                         pool_fields[std_name] = fields_dict[raw_name]
#             pos_key = pos if pos in pool else None
#             negs_present = [n for n in neg_list if n in pool]
#             if pos_key and negs_present:
#                 derived, fs = _derive_bipolar_safe(pool, pool_fields, pos_key, negs_present)
#                 if derived is not None and len(derived) > 0:
#                     return np.asarray(derived, dtype=float), fs
#     return None, None


# def _find_signal_edf(data_folder, patient_id, site_id, session_id):
#     base = os.path.join(data_folder, patient_id, session_id, PHYSIOLOGICAL_DATA_SUBFOLDER)
#     return _first_edf_in(base)


# def _find_annotation_edf(data_folder, patient_id, site_id, session_id):
#     base = os.path.join(data_folder, patient_id, session_id, ALGORITHMIC_ANNOTATIONS_SUBFOLDER)
#     return _first_edf_in(base)


# def _first_edf_in(folder):
#     if not os.path.isdir(folder):
#         return None
#     for fname in sorted(os.listdir(folder)):
#         if fname.lower().endswith('.edf'):
#             return os.path.join(folder, fname)
#     return None


# def _find_epoch_series(sig_dict, name_candidates, exclude_substr=None):
#     exclude_substr = exclude_substr or []
#     for name, arr in sig_dict.items():
#         name_l = str(name).strip().lower()
#         if any(ex in name_l for ex in exclude_substr):
#             continue
#         if any(cand in name_l for cand in name_candidates):
#             return arr
#     return None


# # =====================================================================================
# # PHILOSOPHER'S STONE (optional transfer-learning feature block)
# # =====================================================================================
# # bdsp-core/philosophers-stone (Ganglberger et al., "Brain Health from Sleep EEG: A
# # Multicohort, Deep Learning Biomarker for Cognition, Disease, and Mortality", NEJM AI
# # 2026). Pretrained on 36,000 PSGs across 6 cohorts; takes ONE overnight EEG channel
# # (preferably C4-M1, which is also this file's primary EEG derivation) plus age/sex,
# # and returns brain_health_score, total_cognition_score, fluid_cognition_score,
# # crystallized_cognition_score, and a 1024-d latent (lhl_1..lhl_1024). Only the 4
# # scalar scores are used as features here - deliberately not the 1024-d latent, to
# # avoid v6's mistake of adding hundreds of unvetted embedding dimensions to a model
# # with, at most, a few thousand training rows split across 3 very unevenly sized
# # sites. The 4 scores get the same age-residualization as every other physiological
# # feature in this file (see _fit_preprocessing) even though the source paper's own
# # age-adjusted Cox analysis suggests the score is already fairly age-robust on its
# # own - this file does not rely on that being fully true for THIS metric's specific
# # age-conditioning and defends in depth anyway.
# #
# # BEFORE YOU SUBMIT WITH THIS ENABLED, READ README.md. Three things need to be true:
# #   1. The ~2.3GB checkpoint must be baked into the Docker image at BUILD time (e.g. a
# #      `pip install` + download RUN command in the Dockerfile, or git-lfs) - there is
# #      no network access during train_model/run_model (confirmed on the Challenge
# #      FAQ), so PHILOSOPHER_MODEL_FILE or the default cache path must already be
# #      populated inside the container before training starts.
# #   2. If you want this to run in reasonable time on the full training set, you
# #      likely want to request a GPU on the submission form (A30 or RTX 6000 Ada are
# #      offered on request per the 2026 FAQ) - CPU-only timing should be benchmarked
# #      locally first (see the timing helper in README.md); with no GPU, consider
# #      training on the SMALL training-set version instead of the large one, or
# #      setting USE_PHILOSOPHERS_STONE = False.
# #   3. The Challenge FAQ says transfer learning "must include...code to retrain
# #      (continue training) on the training data we provide" and that pretraining data
# #      must be documented. This file treats the downstream classifier fit in
# #      train_model (on top of Philosopher's Stone's frozen outputs) as that
# #      continuation of training, and documents the pretraining dataset/citation in
# #      this comment block and in README.md. The model's own CC BY-NC 4.0 license is
# #      a restriction on this specific dependency, not on your entry's own code
# #      license - the organizers explicitly allow non-commercial-restricted entry
# #      licenses (2026 FAQ) - but note the dependency's license in your repo for
# #      transparency, and consider asking the forum to confirm if you want certainty
# #      before relying on it for your one chosen test-set entry.

# _SHARED_PHI_EXTRACTOR = None


# def _get_shared_phi_extractor(verbose=False):
#     global _SHARED_PHI_EXTRACTOR
#     if _SHARED_PHI_EXTRACTOR is None:
#         _SHARED_PHI_EXTRACTOR = _PhilosophersStoneExtractor(verbose=verbose)
#     return _SHARED_PHI_EXTRACTOR


# class _PhilosophersStoneExtractor:
#     PHI_SCORE_KEYS = ['brain_health_score', 'total_cognition_score',
#                        'fluid_cognition_score', 'crystallized_cognition_score']

#     def __init__(self, verbose=False):
#         self.available = False
#         self._infer_fn = None
#         self._Config = None
#         try:
#             from philosophers_stone import Config, infer_brain_health  # preferred import path
#             self._infer_fn = infer_brain_health
#             self._Config = Config
#             self.available = True
#         except Exception:
#             try:
#                 from phi_utils.philosopher_utils import Config, infer_brain_health  # legacy path
#                 self._infer_fn = infer_brain_health
#                 self._Config = Config
#                 self.available = True
#             except Exception as e:
#                 self.available = False
#                 if verbose:
#                     print(f"  (Philosopher's Stone import failed: {e})")

#     def extract(self, sig_path, eeg_central_signal, eeg_central_fs, age, sex_code):
#         out = {f'phi_{k}': np.nan for k in self.PHI_SCORE_KEYS}
#         if not self.available or eeg_central_signal is None or eeg_central_fs is None:
#             return out
#         try:
#             eeg_uv = np.asarray(eeg_central_signal, dtype=float)
#             # NOTE: assumes the EDF physical dimension for EEG is already microvolts,
#             # which is the near-universal PSG convention and matches how v1-v6 treated
#             # these signals; if Philosopher's Stone's outputs look degenerate on your
#             # data, check the EDF header's physical_dimension for this channel and
#             # rescale here (e.g. x1000 if signals are actually in millivolts).
#             result = self._infer_fn(
#                 eeg_uv, fs_hz=float(eeg_central_fs), age=float(age) if age is not None else 65.0,
#                 sex=int(sex_code), file_id='challenge_record', cfg=self._Config(),
#             )
#             for k in self.PHI_SCORE_KEYS:
#                 val = result.get(k, np.nan) if isinstance(result, dict) else getattr(result, k, np.nan)
#                 out[f'phi_{k}'] = float(val) if val is not None and np.isfinite(float(val)) else np.nan
#         except Exception:
#             pass  # fail soft: this block never blocks feature extraction for a record
#         return out


# # =====================================================================================
# # PHYSIOLOGICAL SIGNAL FEATURES
# # =====================================================================================

# def _extract_physio(sig_path, rename_rules):
#     out = {}
#     chans_used = {}
#     if sig_path is None or not os.path.exists(sig_path):
#         return out, chans_used
#     try:
#         sig_dict, fields_dict = load_edf_to_nparrays_safe(sig_path)
#     except Exception:
#         return out, chans_used

#     eeg_c, fs_eeg_c = _get_channel(sig_dict, fields_dict, EEG_CENTRAL_CANDIDATES, rename_rules)
#     eeg_f, fs_eeg_f = _get_channel(sig_dict, fields_dict, EEG_FRONTAL_CANDIDATES, rename_rules)
#     eog, fs_eog = _get_channel(sig_dict, fields_dict, EOG_CANDIDATES, rename_rules)
#     chin, fs_chin = _get_channel(sig_dict, fields_dict, CHIN_EMG_CANDIDATES, rename_rules)
#     leg, fs_leg = _get_channel(sig_dict, fields_dict, LEG_EMG_CANDIDATES, rename_rules)
#     ecg, fs_ecg = _get_channel(sig_dict, fields_dict, ECG_CANDIDATES, rename_rules)
#     airflow, fs_air = _get_channel(sig_dict, fields_dict, RESP_AIRFLOW_CANDIDATES, rename_rules)
#     effort, fs_eff = _get_channel(sig_dict, fields_dict, RESP_EFFORT_CANDIDATES, rename_rules)
#     spo2, fs_spo2 = _get_channel(sig_dict, fields_dict, SPO2_CANDIDATES, rename_rules)

#     chans_used['eeg_central'] = eeg_c is not None
#     chans_used['eeg_frontal'] = eeg_f is not None
#     chans_used['eog'] = eog is not None
#     chans_used['chin_emg'] = chin is not None
#     chans_used['leg_emg'] = leg is not None
#     chans_used['ecg'] = ecg is not None
#     chans_used['resp'] = (airflow is not None) or (effort is not None)
#     chans_used['spo2'] = spo2 is not None
#     chans_used['eeg_central_signal'] = eeg_c
#     chans_used['eeg_central_fs'] = fs_eeg_c

#     # --- EEG (central): same 12-metric spectral suite as v1, across early/mid/late
#     #     thirds of the recording. This channel is also handed to Philosopher's Stone. ---
#     if eeg_c is not None and fs_eeg_c:
#         for w_name, w_sig in zip(WINDOWS, _split_windows(eeg_c, fs_eeg_c)):
#             for k, v in _eeg_spectrum(w_sig, fs_eeg_c).items():
#                 out[f'eeg_c_{w_name}_{k}'] = v
#         for k, v in _eeg_spectrum(eeg_c, fs_eeg_c).items():
#             out[f'eeg_c_{k}'] = v  # whole-night summary, used by the AHI/SWA interaction term

#     # --- EEG (frontal): NOT a full duplicate feature block (that is what made v2/v6's
#     #     dimensionality blow up relative to sample size) - just a compact
#     #     frontal-vs-central contrast, which is what the cognitive-aging literature on
#     #     frontal slow-wave activity actually motivates. ---
#     if eeg_f is not None and fs_eeg_f and eeg_c is not None and fs_eeg_c:
#         f_spec = _eeg_spectrum(eeg_f, fs_eeg_f)
#         c_spec = _eeg_spectrum(eeg_c, fs_eeg_c)
#         out['eeg_frontal_delta_rel'] = f_spec.get('delta_rel', np.nan)
#         out['eeg_frontal_minus_central_delta_rel'] = (
#             f_spec.get('delta_rel', np.nan) - c_spec.get('delta_rel', np.nan)
#         )
#         out['eeg_frontal_minus_central_slowing_ratio'] = (
#             f_spec.get('slowing_ratio', np.nan) - c_spec.get('slowing_ratio', np.nan)
#         )

#     # --- EOG: compact activity/rapid-eye-movement-density proxy per window (mirrors
#     #     how EMG is summarized below, kept deliberately small - CAISR's REM
#     #     probability already captures most stage-level REM information, so this adds
#     #     only the *within-REM eye-movement density* signal that raw EOG uniquely
#     #     offers, rather than re-deriving REM staging from scratch). ---
#     if eog is not None and fs_eog:
#         for w_name, w_sig in zip(WINDOWS, _split_windows(eog, fs_eog)):
#             out[f'eog_{w_name}_rms'] = _safe_rms(w_sig)
#             out[f'eog_{w_name}_rapid_movement_density'] = _high_freq_activity_rate(w_sig, fs_eog)

#     # --- Chin EMG: RMS per window (proxy for muscle tone / REM atonia) ---
#     if chin is not None and fs_chin:
#         for w_name, w_sig in zip(WINDOWS, _split_windows(chin, fs_chin)):
#             out[f'emg_chin_rms_{w_name}'] = _safe_rms(w_sig)
#         out['emg_chin_rms_mean'] = _safe_rms(chin)

#     # --- Leg EMG: RMS + a simple movement-rate proxy (thresholded envelope crossings) ---
#     if leg is not None and fs_leg:
#         for w_name, w_sig in zip(WINDOWS, _split_windows(leg, fs_leg)):
#             out[f'emg_leg_rms_{w_name}'] = _safe_rms(w_sig)
#         out['emg_leg_movement_rate'] = _high_freq_activity_rate(leg, fs_leg)

#     # --- ECG: simple HRV proxy per window (same approach as v1: peak-interval CV) ---
#     if ecg is not None and fs_ecg:
#         for w_name, w_sig in zip(WINDOWS, _split_windows(ecg, fs_ecg)):
#             out[f'ecg_hrv_proxy_{w_name}'] = _hrv_proxy(w_sig, fs_ecg)
#         out['ecg_hrv_proxy_mean'] = _hrv_proxy(ecg, fs_ecg)

#     # --- Respiration: airflow frequency + effort RMS per window ---
#     if airflow is not None and fs_air:
#         for w_name, w_sig in zip(WINDOWS, _split_windows(airflow, fs_air)):
#             out[f'resp_airflow_freq_{w_name}'] = _resp_spectrum(w_sig, fs_air)
#     if effort is not None and fs_eff:
#         for w_name, w_sig in zip(WINDOWS, _split_windows(effort, fs_eff)):
#             out[f'resp_effort_rms_{w_name}'] = _safe_rms(w_sig)
#         out['resp_effort_rms_mean'] = _safe_rms(effort)

#     # --- SpO2: desaturation-drop proxy per window (same approach as v1) ---
#     if spo2 is not None and fs_spo2:
#         for w_name, w_sig in zip(WINDOWS, _split_windows(spo2, fs_spo2)):
#             out[f'spo2_drop_{w_name}'] = _spo2_drop_proxy(w_sig)
#         out['spo2_drop_mean'] = _spo2_drop_proxy(spo2)

#     return out, chans_used


# def _split_windows(sig, fs, n=3):
#     sig = np.asarray(sig, dtype=float)
#     if fs is None or fs <= 0 or len(sig) < n:
#         return [sig] * n
#     chunks = np.array_split(sig, n)
#     return chunks


# def _safe_rms(sig):
#     sig = np.asarray(sig, dtype=float)
#     sig = sig[np.isfinite(sig)]
#     if len(sig) == 0:
#         return np.nan
#     return float(np.sqrt(np.mean(sig ** 2)))


# def _high_freq_activity_rate(sig, fs):
#     """Zero-crossing-rate-based proxy for rapid movement/activity density; used for both
#     EOG (rapid-eye-movement density) and leg EMG (movement rate)."""
#     sig = np.asarray(sig, dtype=float)
#     sig = sig[np.isfinite(sig)]
#     if len(sig) < 2 or fs is None or fs <= 0:
#         return np.nan
#     sig = sig - np.nanmean(sig)
#     crossings = np.sum(np.diff(np.sign(sig)) != 0)
#     duration_min = len(sig) / float(fs) / 60.0
#     if duration_min <= 0:
#         return np.nan
#     return float(crossings / duration_min)


# def _hrv_proxy(sig, fs):
#     """Lightweight heart-rate-variability proxy: coefficient of variation of the
#     inter-peak interval from simple threshold-crossing peak detection (not a clinical-
#     grade R-peak detector, but consistent across records and cheap)."""
#     sig = np.asarray(sig, dtype=float)
#     sig = sig[np.isfinite(sig)]
#     if len(sig) < 10 or fs is None or fs <= 0:
#         return np.nan
#     sig = (sig - np.nanmean(sig)) / (np.nanstd(sig) + 1e-9)
#     thresh = 1.5
#     above = sig > thresh
#     peak_idx = np.where(above[1:] & ~above[:-1])[0]
#     if len(peak_idx) < 3:
#         return np.nan
#     intervals = np.diff(peak_idx) / float(fs)
#     intervals = intervals[(intervals > 0.3) & (intervals < 2.0)]  # plausible RR range
#     if len(intervals) < 3:
#         return np.nan
#     return float(np.std(intervals) / (np.mean(intervals) + 1e-9))


# def _resp_spectrum(sig, fs):
#     """Dominant breathing frequency via FFT peak in the 0.05-1.0 Hz band."""
#     sig = np.asarray(sig, dtype=float)
#     sig = sig[np.isfinite(sig)]
#     if len(sig) < 16 or fs is None or fs <= 0:
#         return np.nan
#     sig = sig - np.nanmean(sig)
#     freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
#     power = np.abs(np.fft.rfft(sig)) ** 2
#     band = (freqs >= 0.05) & (freqs <= 1.0)
#     if not np.any(band):
#         return np.nan
#     return float(freqs[band][np.argmax(power[band])])


# def _spo2_drop_proxy(sig):
#     sig = np.asarray(sig, dtype=float)
#     sig = sig[np.isfinite(sig)]
#     if len(sig) < 2:
#         return np.nan
#     baseline = np.nanpercentile(sig, 95)
#     drops = baseline - sig
#     drops = drops[drops > 0]
#     if len(drops) == 0:
#         return 0.0
#     return float(np.nanmean(drops))


# def _eeg_spectrum(sig, fs):
#     """Same 12-metric band-power / spectral-shape summary as v1 (delta/theta/alpha/
#     sigma/beta/total power, relative delta/theta/alpha, spindle-band density, spectral
#     edge 95%, and a slow/fast slowing ratio)."""
#     out = {k: np.nan for k in EEG_METRICS}
#     sig = np.asarray(sig, dtype=float)
#     sig = sig[np.isfinite(sig)]
#     if len(sig) < 32 or fs is None or fs <= 0:
#         return out
#     sig = sig - np.nanmean(sig)
#     freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
#     power = np.abs(np.fft.rfft(sig)) ** 2

#     def band_power(lo, hi):
#         m = (freqs >= lo) & (freqs < hi)
#         return float(np.sum(power[m])) if np.any(m) else 0.0

#     delta = band_power(0.5, 4)
#     theta = band_power(4, 8)
#     alpha = band_power(8, 13)
#     sigma = band_power(11, 16)
#     beta = band_power(13, 30)
#     total = band_power(0.5, 30) + 1e-9

#     out['delta_power'] = delta
#     out['theta_power'] = theta
#     out['alpha_power'] = alpha
#     out['sigma_power'] = sigma
#     out['beta_power'] = beta
#     out['total_power'] = total
#     out['delta_rel'] = delta / total
#     out['theta_rel'] = theta / total
#     out['alpha_rel'] = alpha / total
#     out['spindle_density'] = sigma / total
#     out['slowing_ratio'] = (delta + theta) / (alpha + beta + 1e-9)

#     cum = np.cumsum(power[(freqs >= 0.5) & (freqs <= 30)])
#     if len(cum) > 0 and cum[-1] > 0:
#         f_band = freqs[(freqs >= 0.5) & (freqs <= 30)]
#         idx95 = np.searchsorted(cum, 0.95 * cum[-1])
#         out['spectral_edge95'] = float(f_band[min(idx95, len(f_band) - 1)])
#     return out


# # =====================================================================================
# # PREPROCESSING: age-residualization / interactions / impute / scale
# # =====================================================================================
# # Split into fit/apply on purpose. _fit_preprocessing is called twice in train_model:
# # once per LOSO fold (fit on that fold's training sites ONLY, to get an honest held-out
# # estimate) and once on 100% of the data (for the artifact that actually gets saved).
# # 'age' is used here purely as a regression covariate; it is never added to final_cols,
# # so it can never reach the classifier as an input. This is the one thing every
# # above-v1 regression (v2, v3.1, v4) got wrong to varying degrees - see the module
# # docstring.

# def _compute_interactions(df, interaction_pairs):
#     df = df.copy()
#     made = []
#     for a, b, name in interaction_pairs:
#         if a in df.columns and b in df.columns:
#             df[name] = df[a] * df[b]
#             made.append(name)
#         else:
#             df[name] = np.nan
#             made.append(name)
#     return df, made


# def _fit_preprocessing(df, resid_candidate_cols, direct_cols, interaction_pairs):
#     df2, interaction_names = _compute_interactions(df, interaction_pairs)

#     age_resid_models = {}
#     resid_cols_made = []
#     for col in resid_candidate_cols:
#         if col not in df2.columns:
#             continue
#         sub = df2[[col, 'age']].dropna()
#         if len(sub) < 10 or sub[col].nunique() < 2:
#             continue
#         lr = LinearRegression()
#         lr.fit(sub[['age']].values, sub[col].values)
#         age_resid_models[col] = lr
#         df2[f'{col}_resid'] = df2[col] - lr.predict(df2[['age']].fillna(sub['age'].median()).values)
#         resid_cols_made.append(f'{col}_resid')

#     direct_present = [c for c in direct_cols if c in df2.columns]
#     final_cols = direct_present + interaction_names + resid_cols_made

#     X_raw = df2[final_cols].apply(pd.to_numeric, errors='coerce').values.astype(float)
#     imputer = SimpleImputer(strategy='median')
#     X_imp = imputer.fit_transform(X_raw)
#     if X_imp.shape[0] > 0 and np.any(np.all(np.isnan(X_raw), axis=0)):
#         X_imp = np.nan_to_num(X_imp, nan=0.0)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_imp)

#     prep = {
#         'age_resid_models': age_resid_models,
#         'resid_candidate_cols': resid_candidate_cols,
#         'direct_cols': direct_present,
#         'interaction_pairs': interaction_pairs,
#         'final_cols': final_cols,
#         'imputer': imputer,
#         'scaler': scaler,
#     }
#     return prep, X_scaled


# def _apply_preprocessing(df, prep):
#     df2, _ = _compute_interactions(df, prep['interaction_pairs'])
#     for col, lr in prep['age_resid_models'].items():
#         if col in df2.columns:
#             ages = df2['age'].values.reshape(-1, 1)
#             age_fill = np.nanmedian(ages) if np.any(np.isfinite(ages)) else 65.0
#             ages_filled = np.where(np.isfinite(ages), ages, age_fill)
#             df2[f'{col}_resid'] = df2[col] - lr.predict(ages_filled)
#         else:
#             df2[f'{col}_resid'] = np.nan

#     for c in prep['final_cols']:
#         if c not in df2.columns:
#             df2[c] = np.nan

#     X_raw = df2[prep['final_cols']].apply(pd.to_numeric, errors='coerce').values.astype(float)
#     X_imp = prep['imputer'].transform(X_raw)
#     X_scaled = prep['scaler'].transform(X_imp)
#     return X_scaled


# def _poison_filter(df, candidate_cols, min_site_n=15, verbose=False):
#     """
#     Drops any feature whose Spearman-style sign of correlation with the label flips
#     across training sites (computed per-site, using only sites with at least
#     min_site_n usable rows). Unchanged in spirit from v1: a feature that predicts in
#     opposite directions at different sites is much more likely to be a site/protocol
#     artifact than real physiology, and will not be trustworthy on a genuinely new,
#     unseen site.
#     """
#     kept = []
#     dropped = []
#     sites = df['site'].unique()
#     for col in candidate_cols:
#         if col not in df.columns:
#             continue
#         signs = []
#         for s in sites:
#             sub = df.loc[df['site'] == s, [col, 'label']].dropna()
#             if len(sub) < min_site_n or sub[col].nunique() < 2 or sub['label'].nunique() < 2:
#                 continue
#             corr = np.corrcoef(sub[col].values.astype(float), sub['label'].values.astype(float))[0, 1]
#             if np.isfinite(corr) and abs(corr) > 1e-6:
#                 signs.append(np.sign(corr))
#         if len(signs) >= 2 and len(set(signs)) > 1:
#             dropped.append(col)
#         else:
#             kept.append(col)
#     if verbose and dropped:
#         print(f'  Poison filter dropped {len(dropped)}/{len(candidate_cols)} features '
#               f'with sign-flipping site correlation: {dropped[:15]}{"..." if len(dropped) > 15 else ""}')
#     return kept


# # =====================================================================================
# # MODEL ZOO
# # =====================================================================================
# # Deliberately the same small candidate set v1 used. v2/v3.1/v4 added xgboost/catboost/
# # histgradientboosting on top of this; there is no clean evidence in v1-v6's results
# # that the larger zoo helped (those versions are confounded with the age-leakage
# # problem), and a bigger candidate pool selected via only 3 noisy LOSO folds is a
# # bigger multiple-comparisons risk, not a safer one. If you want to try more model
# # types, do it as its own controlled experiment (e.g. via the local validation script
# # in README.md) rather than folding it silently into this file.

# def _make_fresh_model(name):
#     if name == 'logistic':
#         return LogisticRegression(C=0.1, penalty='l2', class_weight='balanced',
#                                    max_iter=2000, solver='lbfgs')
#     elif name == 'lightgbm':
#         if not _HAVE_LGB:
#             raise ImportError('lightgbm not available')
#         return lgb.LGBMClassifier(
#             n_estimators=150, max_depth=4, learning_rate=0.03, num_leaves=15,
#             min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
#             reg_alpha=1.0, reg_lambda=1.0, class_weight='balanced', verbosity=-1,
#         )
#     elif name == 'stack':
#         base = [
#             ('lr', LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000)),
#             ('svc', SVC(C=0.5, kernel='rbf', probability=True, class_weight='balanced')),
#         ]
#         return StackingClassifier(
#             estimators=base,
#             final_estimator=LogisticRegression(C=0.01, penalty='l2', max_iter=2000),
#             cv=3, n_jobs=1, passthrough=False,
#         )
#     elif name == 'mega':
#         estimators = [('logistic', LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000))]
#         if _HAVE_LGB:
#             estimators.append(('lightgbm', lgb.LGBMClassifier(
#                 n_estimators=150, max_depth=4, learning_rate=0.03, num_leaves=15,
#                 min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
#                 reg_alpha=1.0, reg_lambda=1.0, class_weight='balanced', verbosity=-1,
#             )))
#         estimators.append(('svc', SVC(C=0.5, kernel='rbf', probability=True, class_weight='balanced')))
#         return VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
#     else:
#         raise ValueError(f'Unknown model name: {name}')


# # =====================================================================================
# # LOSO-CV MODEL SELECTION
# # =====================================================================================

# def _run_loso_evaluation(df, resid_candidate_cols, direct_cols, interaction_pairs, sites, verbose=False):
#     """
#     For each training site, refit preprocessing on the OTHER sites only, then evaluate
#     every candidate model (trained on the other sites) on the held-out site's
#     age-conditioned AUROC at both AGE_GAP_PRIMARY and AGE_GAP_SECONDARY. Site-fold
#     scores are combined with a weighted average (weight = number of valid age-matched
#     pairs in that fold), not a plain mean, so a tiny fold (e.g. I0002, which can be
#     under 100 records) doesn't get equal say to a fold with thousands of records.
#     """
#     summary = {name: {'gap2_scores': [], 'gap2_weights': [], 'gap1_scores': [], 'gap1_weights': [],
#                        'per_site': {}} for name in CANDIDATE_MODELS}

#     for test_site in sites:
#         train_mask = (df['site'] != test_site).values
#         test_mask = (df['site'] == test_site).values
#         n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
#         if n_test < 5 or n_train < 20:
#             if verbose:
#                 print(f'  LOSO: skipping site {test_site} (n_train={n_train}, n_test={n_test}, too small).')
#             continue
#         df_train, df_test = df.loc[train_mask], df.loc[test_mask]
#         y_test = df_test['label'].values
#         if len(np.unique(y_test)) < 2:
#             if verbose:
#                 print(f'  LOSO: skipping site {test_site} (only one class present in held-out fold).')
#             continue

#         try:
#             prep_fold, X_train = _fit_preprocessing(df_train, resid_candidate_cols, direct_cols, interaction_pairs)
#             X_test = _apply_preprocessing(df_test, prep_fold)
#         except Exception as e:
#             if verbose:
#                 print(f'  LOSO: preprocessing failed for held-out site {test_site}: {e}')
#             continue
#         y_train = df_train['label'].values
#         ages_test = df_test['age'].values

#         for name in CANDIDATE_MODELS:
#             try:
#                 m = _make_fresh_model(name)
#                 m.fit(X_train, y_train)
#                 p = m.predict_proba(X_test)[:, 1]
#                 s2, w2 = _age_auroc(y_test, p, ages_test, gap=AGE_GAP_PRIMARY, return_weight=True)
#                 s1, w1 = _age_auroc(y_test, p, ages_test, gap=AGE_GAP_SECONDARY, return_weight=True)
#                 if np.isfinite(s2):
#                     summary[name]['gap2_scores'].append(s2)
#                     summary[name]['gap2_weights'].append(max(w2, 1))
#                 if np.isfinite(s1):
#                     summary[name]['gap1_scores'].append(s1)
#                     summary[name]['gap1_weights'].append(max(w1, 1))
#                 summary[name]['per_site'][test_site] = {'gap2': s2, 'gap1': s1, 'n_test': n_test}
#             except Exception as e:
#                 if verbose:
#                     print(f'  LOSO: fit/eval failed for {name} on held-out site {test_site}: {e}')
#                 continue

#     for name in CANDIDATE_MODELS:
#         g2, w2 = summary[name]['gap2_scores'], summary[name]['gap2_weights']
#         g1, w1 = summary[name]['gap1_scores'], summary[name]['gap1_weights']
#         summary[name]['gap2_weighted'] = float(np.average(g2, weights=w2)) if g2 else np.nan
#         summary[name]['gap1_weighted'] = float(np.average(g1, weights=w1)) if g1 else np.nan
#         summary[name]['n_folds'] = len(g2)

#     if verbose:
#         print('  LOSO-CV summary (weighted mean across held-out sites):')
#         for name in CANDIDATE_MODELS:
#             print(f"    {name:10s}  gap=2: {summary[name]['gap2_weighted']:.4f}   "
#                   f"gap=1: {summary[name]['gap1_weighted']:.4f}   (folds: {summary[name]['n_folds']})")
#         gap_diff = {n: abs(summary[n]['gap2_weighted'] - summary[n]['gap1_weighted'])
#                     for n in CANDIDATE_MODELS if np.isfinite(summary[n]['gap2_weighted']) and np.isfinite(summary[n]['gap1_weighted'])}
#         if gap_diff and max(gap_diff.values()) > 0.03:
#             print('  NOTE: gap=1 and gap=2 disagree by >0.03 for at least one candidate. The website '
#                   'states delta=2 but the current example evaluate_model.py calls gap=1 - see module '
#                   'docstring. Consider confirming on the forum before finalizing which model to submit.')
#     return summary


# def _select_best_model(loso_summary, verbose=False):
#     """
#     Picks the candidate with the best weighted gap=2 LOSO score (matching the written
#     metric definition). If the top candidates are within LOSO_TIE_EPSILON of each
#     other, the simplest one wins the tie - a small, explicit bias toward simplicity
#     given how noisy a 3-site LOSO signal is, rather than an implicit one like v1's.
#     """
#     if not loso_summary:
#         if verbose:
#             print('  No LOSO summary available (too few sites); defaulting to logistic.')
#         return 'logistic'
#     scored = [(name, loso_summary[name].get('gap2_weighted', np.nan)) for name in CANDIDATE_MODELS]
#     scored = [(n, s) for n, s in scored if np.isfinite(s)]
#     if not scored:
#         if verbose:
#             print('  All LOSO scores were NaN; defaulting to logistic.')
#         return 'logistic'
#     best_score = max(s for _, s in scored)
#     near_best = [n for n, s in scored if best_score - s <= LOSO_TIE_EPSILON]
#     chosen = min(near_best, key=lambda n: MODEL_COMPLEXITY_RANK.get(n, 99))
#     if verbose:
#         print(f'  Candidates within {LOSO_TIE_EPSILON} of best ({best_score:.4f}): {near_best} -> chose {chosen}')
#     return chosen


# # =====================================================================================
# # METRICS - re-derived line-by-line against physionetchallenges/python-example-2026's
# # evaluate_model.py (0.5-credit tie handling included), not just the website prose. See
# # module docstring for the gap=1-vs-gap=2 discrepancy this file surfaces.
# # =====================================================================================

# def _age_auroc(y_true, y_prob, ages, gap=2.0, return_weight=False):
#     """
#     s_C = Pr(z_i >= z_j : i positive, j negative, |age_i - age_j| <= gap), with 0.5
#     credit for exact ties - vectorized, but mathematically identical to the official
#     nested-loop compute_auroc_age.
#     """
#     y_true = np.asarray(y_true)
#     y_prob = np.asarray(y_prob, dtype=float)
#     ages = np.asarray(ages, dtype=float)
#     pos_mask = y_true == 1
#     neg_mask = y_true == 0
#     ap, an = ages[pos_mask], ages[neg_mask]
#     pp, pn = y_prob[pos_mask], y_prob[neg_mask]
#     if len(ap) == 0 or len(an) == 0:
#         return (np.nan, 0) if return_weight else np.nan
#     age_diff = np.abs(ap[:, None] - an[None, :])
#     within = age_diff <= gap
#     denom = int(np.sum(within))
#     if denom == 0:
#         return (np.nan, 0) if return_weight else np.nan
#     prob_diff = pp[:, None] - pn[None, :]
#     wins = (prob_diff > 0) & within
#     ties = (prob_diff == 0) & within
#     numer = np.sum(wins) + 0.5 * np.sum(ties)
#     score = float(numer / denom)
#     return (score, denom) if return_weight else score


# def _compute_prevalence_map(eval_ages, prevalence_labels, prevalence_ages, gap):
#     """Age -> local prevalence of the positive class in the prevalence reference
#     population (the training set's demographics.csv, per the Challenge forum,
#     2026-06-26), matched within `gap` years. No additive smoothing here, matching the
#     official compute_prevalence exactly; compute_reward below applies the official's
#     own (very loose) clipping."""
#     prevalence_labels = np.asarray(prevalence_labels, dtype=float)
#     prevalence_ages = np.asarray(prevalence_ages, dtype=float)
#     unique_ages = np.unique(eval_ages[np.isfinite(eval_ages)])
#     age_to_prevalence = {}
#     for age in unique_ages:
#         mask = np.abs(prevalence_ages - age) <= gap
#         if np.any(mask):
#             age_to_prevalence[age] = float(np.mean(prevalence_labels[mask]))
#     return age_to_prevalence


# def _compute_reward(y_true, y_pred_binary, ages, age_to_prevalence, min_p=1e-6, max_p=1 - 1e-6):
#     """Prevalence-based reward metric (NOT the ranking metric - see module docstring).
#     Matches the official compute_reward exactly."""
#     y_true = np.asarray(y_true)
#     y_pred_binary = np.asarray(y_pred_binary)
#     ages = np.asarray(ages, dtype=float)
#     scores = []
#     for i in range(len(y_true)):
#         if not np.isfinite(ages[i]) or ages[i] not in age_to_prevalence:
#             continue
#         p = min(max(age_to_prevalence[ages[i]], min_p), max_p)
#         if y_true[i] == 1 and y_pred_binary[i] == 1:
#             scores.append(1.0 / p - 1.0)
#         elif y_true[i] == 0 and y_pred_binary[i] == 1:
#             scores.append(-1.0)
#         elif y_true[i] == 1 and y_pred_binary[i] == 0:
#             scores.append(-1.0)
#         elif y_true[i] == 0 and y_pred_binary[i] == 0:
#             scores.append(1.0 / (1.0 - p) - 1.0)
#     return float(np.mean(scores)) if scores else np.nan


# # =====================================================================================
# # SMALL UTILITIES
# # =====================================================================================

# def _lookup_patient_row(demo_df, patient_id, session_id):
#     """In-memory replacement for the official load_demographics(), which re-reads
#     demographics.csv from disk on every call - fine once, costly across a few thousand
#     records. Same matching semantics (bids_folder + session_id), operating on the
#     dataframe loaded once in train_model."""
#     mask = (demo_df[HEADERS['bids_folder']] == patient_id) & (demo_df[HEADERS['session_id']] == session_id)
#     sub = demo_df.loc[mask]
#     if sub.empty:
#         return {}
#     return sub.iloc[0].to_dict()


# def _lookup_label(demo_df, patient_id):
#     """In-memory replacement for load_diagnoses() with slightly more permissive string
#     matching (handles 'True'/'T'/'Yes'/'Y'/'1' variants), matching what v2/v3.1 already
#     did correctly. Raises if the patient/label can't be found, exactly like the
#     official function, so 'Other' or malformed rows are skipped upstream in train_model
#     the same way they always were."""
#     mask = demo_df[HEADERS['bids_folder']] == patient_id
#     sub = demo_df.loc[mask]
#     if sub.empty:
#         raise ValueError(f'Patient {patient_id} not found in demographics.csv.')
#     val = sub[HEADERS['label']].values[0]
#     if pd.isna(val):
#         raise ValueError('Cognitive impairment diagnosis is missing.')
#     if isinstance(val, (bool, np.bool_)):
#         return 1 if bool(val) else 0
#     val_str = str(val).strip().casefold()
#     if val_str in ('true', 't', 'yes', 'y', '1'):
#         return 1
#     if val_str in ('false', 'f', 'no', 'n', '0'):
#         return 0
#     raise ValueError(f'Unrecognized label value: {val!r}')


# def _unpack_record(record):
#     """Defensively handles either a (patient_id, site_id, session_id) tuple or a dict
#     keyed by HEADERS, since the exact calling convention for run_model's `record`
#     argument can vary slightly by Challenge-year run_challenge_models.py version."""
#     if isinstance(record, dict):
#         return record[HEADERS['bids_folder']], record[HEADERS['site_id']], record[HEADERS['session_id']]
#     if isinstance(record, (tuple, list)) and len(record) >= 3:
#         return record[0], record[1], record[2]
#     raise ValueError(f'Unrecognized record format passed to run_model: {record!r}')


# def _sanitize(features):
#     """Coerces every feature value to a finite float or np.nan (guards against stray
#     inf/None/non-numeric values reaching the imputer/scaler)."""
#     out = {}
#     for k, v in features.items():
#         try:
#             v = float(v)
#             out[k] = v if np.isfinite(v) else np.nan
#         except (TypeError, ValueError):
#             out[k] = np.nan
#     return out

# #!/usr/bin/env python

# # Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# # change or remove non-required functions, and add your own functions.

# """
# PhysioNet Challenge 2026 - V1 + SleepFM + Best Single Model Selection
# Targets: Age-Conditioned AUROC + Prevalence Reward
# Model saved: whichever of {logistic, lightgbm, mega, mega_no_stack} has highest LOSO age-AUROC
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

# # SleepFM
# import torch
# import torch.nn as nn

# warnings.filterwarnings("ignore")
# from helper_code import *


# # CONFIGURATION


# # Model selection strategy:
# # "auto"   = internal LOSO-style CV selects best model per-site
# # "logistic" = force Logistic Regression (C=0.5, best on I0002)
# # "lightgbm" = force LightGBM (best tree on I0006)  
# # "mega"     = average ensemble of logistic + lightgbm + stack
# # "mega_no_stack" = average ensemble of logistic + lightgbm only
# FINAL_MODEL = "auto"

# # number of CV folds for internal model selection
# N_CV_FOLDS = 5

# # window names for temporal feature extraction
# WINDOWS = ['early', 'mid', 'late']

# # EEG spectral metrics
# EEG_METRICS = ['delta','theta','alpha','sigma','beta','alpha_theta',
#                'theta_beta','slowing','delta_sigma','entropy','sef50','sef90']

# # SleepFM paths
# SLEEPFM_BASE_PATH = os.path.join(os.path.dirname(__file__), 'sleepfm')
# SLEEPFM_MODEL_PATH = os.path.join(SLEEPFM_BASE_PATH, 'checkpoints', 'model_base')
# SLEEPFM_CONFIG_PATH = os.path.join(SLEEPFM_MODEL_PATH, 'config.json')
# SLEEPFM_CHANNEL_GROUPS_PATH = os.path.join(SLEEPFM_BASE_PATH, 'configs', 'channel_groups_challenge.json')


# class SleepFMFeatureExtractor:
#     """SleepFM extractor with memory-safe EDF handling."""

#     def __init__(self, verbose=False):
#         self.verbose = verbose
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = None
#         self.model_config = None
#         self.channel_groups = None
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

#             if not os.path.exists(SLEEPFM_CHANNEL_GROUPS_PATH):
#                 if self.verbose:
#                     print(f"Channel groups not found at {SLEEPFM_CHANNEL_GROUPS_PATH}")
#                 return

#             with open(SLEEPFM_CHANNEL_GROUPS_PATH, 'r') as f:
#                 self.channel_groups = json.load(f)

#             repo_root = os.path.dirname(__file__)
#             if repo_root not in sys.path:
#                 sys.path.insert(0, repo_root)

#             try:
#                 from sleepfm.models.models import SetTransformer
#                 model_class = SetTransformer
#             except ImportError as e:
#                 if self.verbose:
#                     print(f"Could not import SleepFM model: {e}")
#                 self.model = None
#                 return

#             in_channels = self.model_config.get('in_channels', 1)
#             patch_size = self.model_config.get('patch_size', 640)
#             embed_dim = self.model_config.get('embed_dim', 128)
#             num_heads = self.model_config.get('num_heads', 8)
#             num_layers = self.model_config.get('num_layers', 6)
#             pooling_head = self.model_config.get('pooling_head', 8)
#             dropout = self.model_config.get('dropout', 0.0)

#             self.model = model_class(
#                 in_channels=in_channels,
#                 patch_size=patch_size,
#                 embed_dim=embed_dim,
#                 num_heads=num_heads,
#                 num_layers=num_layers,
#                 pooling_head=pooling_head,
#                 dropout=dropout,
#                 max_seq_length=128
#             )

#             weights_path = os.path.join(SLEEPFM_MODEL_PATH, 'best.pt')
#             if not os.path.exists(weights_path):
#                 if self.verbose:
#                     print(f"SleepFM weights not found at {weights_path}")
#                 self.model = None
#                 return

#             checkpoint = torch.load(weights_path, map_location=self.device)
#             state_dict = checkpoint.get('state_dict', checkpoint)

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

#     def _edf_to_hdf5(self, edf_path, temp_dir):
#         """Convert EDF to HDF5 WITHOUT loading entire file into memory."""
#         try:
#             import mne
#             import h5py

#             # CRITICAL: preload=False means header only
#             raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

#             # Resample in-place (affects header, not full load)
#             if abs(raw.info['sfreq'] - 128.0) > 0.5:
#                 raw.resample(128.0)

#             subject_id = os.path.splitext(os.path.basename(edf_path))[0]
#             hdf5_path = os.path.join(temp_dir, f"{subject_id}_psg.hdf5")

#             with h5py.File(hdf5_path, 'w') as f:
#                 for mod, channel_list in self.channel_groups.items():
#                     mod_group = f.create_group(mod)
#                     matched_channels = []
#                     matched_data = []

#                     for ch_name in raw.ch_names:
#                         ch_lower = ch_name.lower().replace(' ', '').replace('_', '').replace('-', '')
#                         for target in channel_list:
#                             target_lower = target.lower().replace(' ', '').replace('_', '').replace('-', '')
#                             if target_lower in ch_lower or ch_lower in target_lower:
#                                 # Load only this ONE channel from disk
#                                 data = raw.get_data(picks=ch_name)[0]
#                                 matched_channels.append(ch_name)
#                                 matched_data.append(data)
#                                 break

#                     if len(matched_data) > 0:
#                         data_array = np.stack(matched_data, axis=0)
#                         mod_group.create_dataset('data', data=data_array)
#                         mod_group.create_dataset('channels', data=np.array(matched_channels, dtype='S'))
#                         mod_group.attrs['fs'] = 128.0

#                         # Free immediately
#                         del matched_data, data_array

#             return hdf5_path

#         except Exception as e:
#             if self.verbose:
#                 print(f"EDF to HDF5 conversion failed: {e}")
#             return None

#     def _generate_embeddings(self, hdf5_path):
#         """Generate embeddings by streaming HDF5 chunks."""
#         if self.model is None:
#             return None

#         try:
#             import h5py

#             with h5py.File(hdf5_path, 'r') as f:
#                 all_embeddings = {}

#                 for mod in self.channel_groups.keys():
#                     if mod not in f:
#                         all_embeddings[mod] = None
#                         continue

#                     mod_group = f[mod]
#                     if 'data' not in mod_group:
#                         all_embeddings[mod] = None
#                         continue

#                     # Stream from h5py - don't load entire dataset
#                     dataset = mod_group['data']
#                     C, T = dataset.shape

#                     patch_size = 640
#                     n_patches = T // patch_size
#                     if n_patches == 0:
#                         all_embeddings[mod] = None
#                         continue

#                     max_patches_per_chunk = 128
#                     chunk_embeddings = []

#                     for chunk_start in range(0, n_patches, max_patches_per_chunk):
#                         chunk_end = min(chunk_start + max_patches_per_chunk, n_patches)
#                         chunk_samples_start = chunk_start * patch_size
#                         chunk_samples_end = chunk_end * patch_size

#                         # Stream only this chunk
#                         chunk_data = dataset[:, chunk_samples_start:chunk_samples_end]
#                         x = torch.from_numpy(chunk_data).float().to(self.device)
#                         x = x.unsqueeze(0)
#                         mask = torch.ones(1, C, dtype=torch.bool).to(self.device)

#                         with torch.no_grad():
#                             pooled, tokens = self.model(x, mask)

#                         chunk_embeddings.append(tokens.cpu().numpy())
#                         del x, mask, tokens, pooled, chunk_data

#                     if chunk_embeddings:
#                         all_embeddings[mod] = np.concatenate(chunk_embeddings, axis=1)
#                     else:
#                         all_embeddings[mod] = None

#                 return all_embeddings

#         except Exception as e:
#             if self.verbose:
#                 print(f"Embedding generation failed: {e}")
#             return None

#     def extract_features(self, edf_path):
#         """Main entry point: EDF -> features."""
#         if self.model is None:
#             return None

#         with tempfile.TemporaryDirectory() as temp_dir:
#             hdf5_path = self._edf_to_hdf5(edf_path, temp_dir)
#             if hdf5_path is None:
#                 return None

#             embeddings = self._generate_embeddings(hdf5_path)
#             if embeddings is None:
#                 return None

#             features = {}
#             for mod in self.channel_groups.keys():
#                 emb = embeddings.get(mod)
#                 if emb is not None and emb.size > 0:
#                     emb_mod = emb[0]

#                     features[f'sleepfm_{mod}_mean'] = float(np.mean(emb_mod))
#                     features[f'sleepfm_{mod}_std'] = float(np.std(emb_mod))
#                     features[f'sleepfm_{mod}_max'] = float(np.max(emb_mod))
#                     features[f'sleepfm_{mod}_min'] = float(np.min(emb_mod))
#                     features[f'sleepfm_{mod}_median'] = float(np.median(emb_mod))

#                     n_dims = min(8, emb_mod.shape[1])
#                     for i in range(n_dims):
#                         features[f'sleepfm_{mod}_d{i}_mean'] = float(np.mean(emb_mod[:, i]))
#                         features[f'sleepfm_{mod}_d{i}_std'] = float(np.std(emb_mod[:, i]))
#                         features[f'sleepfm_{mod}_d{i}_max'] = float(np.max(emb_mod[:, i]))
#                         features[f'sleepfm_{mod}_d{i}_min'] = float(np.min(emb_mod[:, i]))

#                     if emb_mod.shape[0] > 1:
#                         features[f'sleepfm_{mod}_temporal_std'] = float(np.std(np.mean(emb_mod, axis=1)))
#                         features[f'sleepfm_{mod}_temporal_range'] = float(
#                             np.max(np.mean(emb_mod, axis=1)) - np.min(np.mean(emb_mod, axis=1))
#                         )
#                     else:
#                         features[f'sleepfm_{mod}_temporal_std'] = 0.0
#                         features[f'sleepfm_{mod}_temporal_range'] = 0.0
#                 else:
#                     features[f'sleepfm_{mod}_mean'] = np.nan
#                     features[f'sleepfm_{mod}_std'] = np.nan
#                     features[f'sleepfm_{mod}_max'] = np.nan
#                     features[f'sleepfm_{mod}_min'] = np.nan
#                     features[f'sleepfm_{mod}_median'] = np.nan
#                     for i in range(8):
#                         features[f'sleepfm_{mod}_d{i}_mean'] = np.nan
#                         features[f'sleepfm_{mod}_d{i}_std'] = np.nan
#                         features[f'sleepfm_{mod}_d{i}_max'] = np.nan
#                         features[f'sleepfm_{mod}_d{i}_min'] = np.nan
#                     features[f'sleepfm_{mod}_temporal_std'] = np.nan
#                     features[f'sleepfm_{mod}_temporal_range'] = np.nan

#             return features


# # REQUIRED FUNCTIONS (DO NOT CHANGE SIGNATURES)


# def train_model(data_folder, model_folder, verbose):
#     """
#     Train model on official challenge data.
#     Uses site-aware LOSO CV for model selection to match hidden test conditions.
#     """
#     if verbose:
#         print('Finding Challenge data...')

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

#     # PHASE 1: Extract features from all records:
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

#             # Add SleepFM features
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

#     # PHASE 2: Poison filtering (site-directionality tolerance):
#     if verbose:
#         print('Filtering site-poisonous features...')

#     sites = df['site'].unique()
#     poison_features = []

#     for col in feature_names:
#         if col.startswith('inter_'):
#             continue  # keep interactions (designed to be stable)
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
#             # TOLERANCE filter: severe inversion (>0.15 magnitude swing across 0)
#             if cmax > 0.05 and cmin < -0.05 and (cmax - cmin) > 0.15:
#                 poison_features.append(col)

#     kept = [f for f in feature_names if f not in poison_features]
#     if verbose:
#         print(f"  Kept {len(kept)} | Dropped {len(poison_features)} poison")

#     # PHASE 3: Build interaction features:
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

#     # PHASE 4: Age residualization:
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

#     # PHASE 5: Impute & scale:
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

#     # PHASE 6: Train all candidate models:
#     if verbose:
#         print('Training candidate models...')

#     candidates = {}

#     # 1. Logistic Regression (C=0.5 — LOSO winner on I0002: s_C 0.7376)
#     candidates['logistic'] = LogisticRegression(
#         C=0.5, penalty='l2', solver='liblinear',
#         class_weight='balanced', max_iter=1000, random_state=42
#     )
#     candidates['logistic'].fit(Xs, y)

#     # 2. LightGBM (best tree on I0006: s_C 0.7054)
#     if lgb is not None:
#         candidates['lightgbm'] = lgb.LGBMClassifier(
#             n_estimators=200, max_depth=4, learning_rate=0.05,
#             random_state=42, n_jobs=1, verbose=-1,
#             class_weight='balanced'
#         )
#         candidates['lightgbm'].fit(Xs, y)

#     # 3. Mega Ensemble (average of logistic + lightgbm + stack probabilities)
#     # Implemented as a VotingClassifier with soft voting
#     estimators_for_mega = [('logistic', candidates['logistic'])]
#     if lgb is not None:
#         estimators_for_mega.append(('lightgbm', candidates['lightgbm']))
#     estimators_for_mega.append(('stack', StackingClassifier(
#         estimators=[
#             ('lr1', LogisticRegression(C=0.05, class_weight='balanced', 
#                                         solver='liblinear', max_iter=1000)),
#             ('svm', SVC(probability=True, C=0.1, kernel='linear', 
#                         class_weight='balanced'))
#         ],
#         final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
#         n_jobs=1
#     )))

#     candidates['mega'] = VotingClassifier(
#         estimators=estimators_for_mega,
#         voting='soft',
#         n_jobs=1
#     )
#     # Fit mega ensemble (just for API consistency, it averages probs)
#     candidates['mega'].fit(Xs, y)

#     # 4. Mega No Stack (average of logistic + lightgbm only)
#     estimators_for_mega_no_stack = [('logistic', candidates['logistic'])]
#     if lgb is not None:
#         estimators_for_mega_no_stack.append(('lightgbm', candidates['lightgbm']))

#     candidates['mega_no_stack'] = VotingClassifier(
#         estimators=estimators_for_mega_no_stack,
#         voting='soft',
#         n_jobs=1
#     )
#     candidates['mega_no_stack'].fit(Xs, y)

#     # PHASE 7: LOSO CV to select best model by age-AUROC
#     names = list(candidates.keys())
#     oof_probs = {name: np.full(len(y), np.nan) for name in names}

#     if verbose:
#         print('Running site-aware LOSO CV...')

#     for test_site in sites:
#         train_mask = sites_arr != test_site
#         test_mask = sites_arr == test_site

#         if np.sum(test_mask) < 5 or np.sum(train_mask) < 20:
#             continue
#         if len(np.unique(y[test_mask])) < 2:
#             continue

#         X_tr, X_val = Xs[train_mask], Xs[test_mask]
#         y_tr, y_val = y[train_mask], y[test_mask]
#         a_val = ages[test_mask]
#         test_idx = np.where(test_mask)[0]

#         for name in names:
#             m = _make_fresh_model(name)
#             m.fit(X_tr, y_tr)
#             p = m.predict_proba(X_val)[:, 1]
#             oof_probs[name][test_idx] = p

#     # Report individual scores and pick best
#     if verbose:
#         for name in names:
#             valid = ~np.isnan(oof_probs[name])
#             if np.sum(valid) > 0:
#                 sc = _age_auroc(y[valid], oof_probs[name][valid], ages[valid], delta=2.0)
#                 print(f"  {name:20s}: LOSO age-AUROC = {sc:.4f}")

#     best_score = -np.inf
#     best_name = None
#     for name in names:
#         valid = ~np.isnan(oof_probs[name])
#         if np.sum(valid) > 0:
#             sc = _age_auroc(y[valid], oof_probs[name][valid], ages[valid], delta=2.0)
#             if sc > best_score:
#                 best_score = sc
#                 best_name = name

#     if verbose:
#         print(f"Best model: {best_name} with LOSO age-AUROC = {best_score:.4f}")

#     # PHASE 8: Save artifact:
#     os.makedirs(model_folder, exist_ok=True)

#     artifact = {
#         'model': candidates[best_name],
#         'model_name': best_name,
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

#     # Add SleepFM features
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

#     # builds interaction features
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

#     # Single best model prediction
#     prob = model_artifact['model'].predict_proba(Xs)[0, 1]

#     # Threshold at 0.5 for binary for prevalence-aware, could tune but 0.5 is standard
#     binary = int(prob >= 0.5)

#     return binary, prob


# # FEATURE EXTRACTION 


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


# # UTILITIES

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
#     elif name == 'mega_no_stack':
#         estimators = [('logistic', LogisticRegression(C=0.5, penalty='l2', solver='liblinear',
#                                                        class_weight='balanced', max_iter=1000, random_state=42))]
#         if lgb is not None:
#             estimators.append(('lightgbm', lgb.LGBMClassifier(n_estimators=200, max_depth=4, 
#                                                                learning_rate=0.05, random_state=42,
#                                                                n_jobs=1, verbose=-1, class_weight='balanced')))
#         return VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
#     else:
#         raise ValueError(f"Unknown model name: {name}")

# #!/usr/bin/env python
# """
# PhysioNet Challenge 2026 - Age-Conditioned AUROC Optimized v4
# Wins by: age-residualized features + stage-conditional spectral + ensemble weight optimization
# """

# import os
# import warnings
# import numpy as np
# import pandas as pd
# import scipy.stats
# import scipy.signal
# from tqdm import tqdm
# import joblib

# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import StackingClassifier
# from sklearn.svm import SVC
# from scipy.optimize import minimize

# try:
#     import lightgbm as lgb
# except ImportError:
#     lgb = None

# warnings.filterwarnings("ignore")
# from helper_code import *

# # -------------------------------------------------------------------------
# # CONFIGURATION
# # -------------------------------------------------------------------------
# FINAL_MODEL = "auto"
# WINDOWS = ['early', 'mid', 'late']
# EEG_METRICS = ['delta','theta','alpha','sigma','beta','alpha_theta',
#                'theta_beta','slowing','delta_sigma','entropy','sef50','sef90',
#                'hjorth_activity','hjorth_mobility','hjorth_complexity']

# # -------------------------------------------------------------------------
# # REQUIRED FUNCTIONS
# # -------------------------------------------------------------------------
# def train_model(data_folder, model_folder, verbose):
#     if verbose:
#         print('Finding Challenge data...')
    
#     patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
#     patient_metadata_list = find_patients(patient_data_file)
#     num_records = len(patient_metadata_list)
    
#     if num_records == 0:
#         raise FileNotFoundError('No data provided.')
    
#     if verbose:
#         print(f'Found {num_records} records. Extracting features...')
    
#     # PHASE 1: Extract features
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
    
#     # PHASE 2: Temporal-difference features (late vs early)
#     for m in EEG_METRICS + ['emg_rms','ecg_hrv','ecg_mean_hr','resp_freq',
#                             'resp_effort','spo2_drop','spo2_mean','spo2_min']:
#         e = f'physio_early_{m}'
#         l = f'physio_late_{m}'
#         if e in df.columns and l in df.columns:
#             df[f'physio_delta_{m}'] = df[l] - df[e]
#             feature_names.append(f'physio_delta_{m}')
    
#     # PHASE 3: Poison filtering (directional site instability)
#     if verbose:
#         print('Filtering site-poisonous features...')
    
#     sites = df['site'].unique()
#     poison_features = []
    
#     for col in feature_names:
#         if col.startswith('inter_') or col.startswith('physio_delta_'):
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
    
#     # PHASE 4: Interaction features
#     interactions = [
#         ('resp_caisr_ahi', 'physio_late_spo2_drop', 'inter_AHI_SpO2'),
#         ('caisr_prob_w_mean', 'physio_early_eeg_delta', 'inter_WASO_SWA'),
#         ('caisr_prob_r_mean', 'physio_late_emg_rms', 'inter_REM_EMG_Atonia'),
#         ('physio_late_ecg_hrv', 'physio_late_resp_effort', 'inter_HRV_RespEffort'),
#         ('physio_late_eeg_theta_beta', 'physio_late_resp_freq', 'inter_ThetaBeta_RespFreq'),
#         ('physio_mid_eeg_delta_sigma', 'physio_mid_spo2_drop', 'inter_mid_DeltaSigma_SpO2'),
#         ('physio_mid_eeg_theta_beta', 'physio_mid_resp_effort', 'inter_mid_ThetaBeta_Effort'),
#     ]
    
#     age_interactions = []
#     if 'age' in kept:
#         age_interactions.append(('age', 'age', 'age_sq'))
#         for partner in ['resp_caisr_ahi', 'caisr_n3_pct', 'caisr_prob_n3_mean',
#                         'physio_early_eeg_delta', 'physio_late_eeg_entropy', 'bmi']:
#             if partner in kept:
#                 age_interactions.append(('age', partner, f'inter_age_{partner}'))
    
#     for f1, f2, name in interactions + age_interactions:
#         if f1 in kept and f2 in kept:
#             df[name] = df[f1] * df[f2] if f1 != f2 else df[f1] ** 2
#             if name not in kept:
#                 kept.append(name)
    
#     # PHASE 5: Age residualization (skip demographics & age interactions)
#     if verbose:
#         print('Age-residualizing physiology features...')
    
#     demo_cols = {'age', 'sex_male', 'sex_female', 'race_white', 'race_black',
#                  'race_asian', 'race_other', 'race_unavailable'}
#     resid_cols = []
#     age_resid_models = {}
    
#     for col in kept:
#         if col in demo_cols or col.startswith('inter_age_') or col == 'age_sq':
#             continue
#         sub = df.dropna(subset=[col, 'age'])
#         if len(sub) > 10:
#             lr = LinearRegression()
#             lr.fit(sub[['age']].values, sub[col].values)
#             df[f"{col}_resid"] = df[col] - lr.predict(df[['age']].values)
#             resid_cols.append(f"{col}_resid")
#             age_resid_models[col] = lr
    
#     # PHASE 6: Impute & scale
#     final_cols = list(demo_cols.intersection(kept)) + [c for c in kept if c.startswith('inter_')] + ['age_sq'] + resid_cols
#     # Ensure all exist
#     for col in final_cols:
#         if col not in df.columns:
#             df[col] = np.nan
#         med = df[col].median()
#         if np.isnan(med):
#             med = 0.0
#         df[col] = df[col].fillna(med)
    
#     X = df[final_cols].values
#     y = df['label'].values
#     ages = df['age'].values
#     sites_arr = df['site'].values
    
#     imputer = SimpleImputer(strategy='median')
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(imputer.fit_transform(X))
    
#     # PHASE 7: Train candidate models
#     if verbose:
#         print('Training candidate models...')
    
#     candidates = {}
    
#     candidates['logistic'] = LogisticRegression(
#         C=0.5, penalty='l2', solver='liblinear',
#         class_weight='balanced', max_iter=1000, random_state=42)
#     candidates['logistic'].fit(Xs, y)
    
#     candidates['logistic_l1'] = LogisticRegression(
#         C=0.1, penalty='l1', solver='liblinear',
#         class_weight='balanced', max_iter=1000, random_state=42)
#     candidates['logistic_l1'].fit(Xs, y)
    
#     if lgb is not None:
#         candidates['lgb_shallow'] = lgb.LGBMClassifier(
#             n_estimators=300, max_depth=3, learning_rate=0.05, num_leaves=15,
#             min_child_samples=10, reg_lambda=1,
#             random_state=42, n_jobs=1, verbose=-1,
#             class_weight='balanced')
#         candidates['lgb_shallow'].fit(Xs, y)
        
#         candidates['lgb_medium'] = lgb.LGBMClassifier(
#             n_estimators=300, max_depth=4, learning_rate=0.05, num_leaves=31,
#             min_child_samples=10, reg_lambda=1,
#             random_state=42, n_jobs=1, verbose=-1,
#             class_weight='balanced')
#         candidates['lgb_medium'].fit(Xs, y)
    
#     candidates['svm_lin'] = SVC(
#         probability=True, C=0.5, kernel='linear',
#         class_weight='balanced', random_state=42)
#     candidates['svm_lin'].fit(Xs, y)
    
#     candidates['stack'] = StackingClassifier(
#         estimators=[
#             ('lr1', LogisticRegression(C=0.05, class_weight='balanced', solver='liblinear', max_iter=1000)),
#             ('svm', SVC(probability=True, C=0.1, kernel='linear', class_weight='balanced'))
#         ],
#         final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
#         n_jobs=1, passthrough=False)
#     candidates['stack'].fit(Xs, y)
    
#     # PHASE 8: LOSO CV + ensemble weight optimization on age-AUROC
#     names = list(candidates.keys())
#     oof_probs = {name: np.full(len(y), np.nan) for name in names}
    
#     if verbose:
#         print('Running site-aware LOSO CV...')
    
#     for test_site in sites:
#         train_mask = sites_arr != test_site
#         test_mask = sites_arr == test_site
        
#         if np.sum(test_mask) < 5 or np.sum(train_mask) < 20:
#             continue
#         if len(np.unique(y[test_mask])) < 2:
#             continue
        
#         X_tr, X_val = Xs[train_mask], Xs[test_mask]
#         y_tr, y_val = y[train_mask], y[test_mask]
#         a_val = ages[test_mask]
#         test_idx = np.where(test_mask)[0]
        
#         for name in names:
#             m = _make_fresh_model(name)
#             m.fit(X_tr, y_tr)
#             p = m.predict_proba(X_val)[:, 1]
#             oof_probs[name][test_idx] = p
    
#     # Report individual scores
#     if verbose:
#         for name in names:
#             valid = ~np.isnan(oof_probs[name])
#             if np.sum(valid) > 0:
#                 sc = _age_auroc(y[valid], oof_probs[name][valid], ages[valid], delta=2.0)
#                 print(f"  {name:15s}: LOSO age-AUROC = {sc:.4f}")
    
#     # Optimize ensemble weights to maximize age-AUROC
#     def _ens_score(weights):
#         w = np.maximum(weights, 0)
#         if np.sum(w) == 0:
#             return 0.0
#         w = w / np.sum(w)
#         ens = np.zeros(len(y))
#         for i, name in enumerate(names):
#             valid = ~np.isnan(oof_probs[name])
#             ens[valid] += w[i] * oof_probs[name][valid]
#         valid = ~np.isnan(ens)
#         if np.sum(valid) == 0:
#             return 0.0
#         return _age_auroc(y[valid], ens[valid], ages[valid], delta=2.0)
    
#     # Replace the SLSQP block with this:
#     scores = []
#     for name in names:
#         valid = ~np.isnan(oof_probs[name])
#         sc = _age_auroc(y[valid], oof_probs[name][valid], ages[valid], delta=2.0)
#         scores.append(sc)

#     valid_mask = np.array(scores) > 0.50
#     scores = np.array(scores)
#     scores[~valid_mask] = -1e9  # <-- not 0.0

#     # Softmax weighting
#     exp_scores = np.exp(scores - np.max(scores))
#     best_w = exp_scores / np.sum(exp_scores)
#     ensemble_weights = dict(zip(names, best_w))
    
#     if verbose:
#         print(f"Optimized ensemble LOSO age-AUROC = {_ens_score(best_w):.4f}")
#         for name, w in zip(names, best_w):
#             print(f"  {name:15s}: weight = {w:.3f}")
    
#     # PHASE 9: Save artifact
#     os.makedirs(model_folder, exist_ok=True)
    
#     artifact = {
#         'models': candidates,
#         'ensemble_weights': ensemble_weights,
#         'scaler': scaler,
#         'imputer': imputer,
#         'final_cols': final_cols,
#         'resid_cols': resid_cols,
#         'age_resid_models': age_resid_models,
#         'interactions': interactions + age_interactions,
#         'sites_seen': list(sites),
#         'median_age': float(np.median(ages)),
#     }
    
#     joblib.dump(artifact, os.path.join(model_folder, 'model.sav'))
#     if verbose:
#         print('Training complete. Model saved.')


# def load_model(model_folder, verbose):
#     return joblib.load(os.path.join(model_folder, 'model.sav'))


# def run_model(model_artifact, record, data_folder, verbose):
#     patient_id = record[HEADERS['bids_folder']]
#     site_id = record[HEADERS['site_id']]
#     session_id = record[HEADERS['session_id']]
    
#     patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
#     patient_data = load_demographics(patient_data_file, patient_id, session_id)
    
#     feats = extract_all_features(data_folder, patient_id, site_id, session_id, patient_data)
#     if feats is None:
#         return float('nan'), float('nan')
    
#     df = pd.DataFrame([feats])
#     age = load_age(patient_data)
#     if np.isnan(age):
#         age = model_artifact.get('median_age', 65.0)
#     df['age'] = age
    
#     # Build interactions
#     for f1, f2, name in model_artifact.get('interactions', []):
#         if f1 in df.columns and f2 in df.columns:
#             df[name] = df[f1] * df[f2] if f1 != f2 else df[f1] ** 2
    
#     # Build temporal deltas
#     for m in EEG_METRICS + ['emg_rms','ecg_hrv','ecg_mean_hr','resp_freq',
#                             'resp_effort','spo2_drop','spo2_mean','spo2_min']:
#         e = f'physio_early_{m}'
#         l = f'physio_late_{m}'
#         if e in df.columns and l in df.columns:
#             df[f'physio_delta_{m}'] = df[l] - df[e]
    
#     # Age residualization
#     for col, lr in model_artifact.get('age_resid_models', {}).items():
#         if col in df.columns:
#             df[f"{col}_resid"] = df[col].values - lr.predict(np.array([[age]]))[0]
    
#     final_cols = model_artifact['final_cols']
#     for col in final_cols:
#         if col not in df.columns:
#             df[col] = float('nan')
#         med = df[col].median()
#         if np.isnan(med):
#             med = 0.0
#         df[col] = df[col].fillna(med)
    
#     X = df[final_cols].values
#     Xs = model_artifact['scaler'].transform(model_artifact['imputer'].transform(X))
    
#     # Ensemble prediction
#     weights = model_artifact['ensemble_weights']
#     prob = 0.0
#     for name, w in weights.items():
#         if w > 0 and name in model_artifact['models']:
#             prob += w * model_artifact['models'][name].predict_proba(Xs)[0, 1]
    
#     # Threshold irrelevant for AUROC ranking; use 0.5 for API compliance
#     binary = int(prob >= 0.5)
#     return binary, float(prob)


# # -------------------------------------------------------------------------
# # FEATURE EXTRACTION
# # -------------------------------------------------------------------------
# def extract_all_features(data_folder, patient_id, site_id, session_id, patient_data):
#     features = {}
    
#     # 1. Demographics
#     features['age'] = load_age(patient_data)
#     sex = load_sex(patient_data, standardize=True)
#     features['sex_male'] = 1 if sex == 'Male' else 0
#     features['sex_female'] = 1 if sex == 'Female' else 0
#     race = load_race(patient_data, standardize=True)
#     features['race_white'] = 1 if race == 'White' else 0
#     features['race_black'] = 1 if race == 'Black' else 0
#     features['race_asian'] = 1 if race == 'Asian' else 0
#     features['race_other'] = 1 if race == 'Others' else 0
#     features['race_unavailable'] = 1 if race == 'Unavailable' else 0
#     features['bmi'] = load_bmi(patient_data)
    
#     # 2. CAISR
#     caisr_path = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
#                               site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
#     caisr_features, caisr_stages = _extract_caisr(caisr_path)
#     features.update(caisr_features)
    
#     # 3. Physiology
#     physio_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
#                                site_id, f"{patient_id}_ses-{session_id}.edf")
#     features.update(_extract_physio(physio_path, stages=caisr_stages))
    
#     return features


# def _extract_caisr(edf_path):
#     """Return (features_dict, stages_array_or_None)."""
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
#         'caisr_n3_pct': np.nan, 'caisr_n2_pct': np.nan,
#         'caisr_n1_pct': np.nan, 'caisr_rem_pct': np.nan,
#         'caisr_wake_pct': np.nan, 'caisr_sleep_efficiency': np.nan,
#         'caisr_sleep_latency_min': np.nan, 'caisr_rem_latency_min': np.nan,
#         'caisr_waso_min': np.nan, 'caisr_rdi': np.nan,
#         'caisr_stage_prob_var': np.nan, 'caisr_n1_sleep_pct': np.nan,
#         'caisr_plm_index': np.nan,
#     }
#     stages_out = None
    
#     if not os.path.exists(edf_path):
#         return out, stages_out
    
#     try:
#         data_dict, fs_dict = load_signal_data(edf_path)
#         if not data_dict:
#             return out, stages_out
        
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
        
#         if stages is not None and len(stages) > 0:
#             stages_out = stages
#             epoch_dur_min = 0.5
#             n_epochs = len(stages)
#             tst = (n_epochs * 30) / 3600
#             out['stage_caisr_tst'] = tst
            
#             valid = stages[stages < 9]
#             if len(valid) > 0:
#                 out['caisr_wake_pct'] = float(np.mean(valid == 5))
#                 out['caisr_n1_pct'] = float(np.mean(valid == 3))
#                 out['caisr_n2_pct'] = float(np.mean(valid == 2))
#                 out['caisr_n3_pct'] = float(np.mean(valid == 1))
#                 out['caisr_rem_pct'] = float(np.mean(valid == 4))
#                 out['stage_caisr_se'] = float(np.mean((valid >= 1) & (valid <= 4)))
#                 out['caisr_sleep_efficiency'] = out['stage_caisr_se']
                
#                 sleep_idx = np.where(np.isin(valid, [1, 2, 3, 4]))[0]
#                 if len(sleep_idx) > 0:
#                     out['caisr_sleep_latency_min'] = float(sleep_idx[0] * epoch_dur_min)
#                 else:
#                     out['caisr_sleep_latency_min'] = float(len(valid) * epoch_dur_min)
                
#                 rem_idx = np.where(valid == 4)[0]
#                 if len(rem_idx) > 0:
#                     out['caisr_rem_latency_min'] = float(rem_idx[0] * epoch_dur_min)
                
#                 sleep_started = False
#                 wake_epochs = 0
#                 for s in valid:
#                     if not sleep_started and s in [1, 2, 3, 4]:
#                         sleep_started = True
#                     if sleep_started and s == 5:
#                         wake_epochs += 1
#                 out['caisr_waso_min'] = float(wake_epochs * epoch_dur_min)
                
#                 sleep_stages = valid[np.isin(valid, [1, 2, 3, 4])]
#                 if len(sleep_stages) > 0:
#                     out['caisr_n1_sleep_pct'] = float(np.mean(sleep_stages == 3))
            
#             out['stage_transition_rate'] = float(np.sum(np.diff(stages) != 0) / max(tst, 0.5))
            
#             stage_probs = []
#             for p in [pn3, pn2, pn1, pr, pw]:
#                 if p is not None and len(p) == len(stages):
#                     stage_probs.append(p)
#             if len(stage_probs) > 0:
#                 sp_arr = np.array(stage_probs)
#                 out['caisr_stage_prob_var'] = float(np.mean(np.var(sp_arr, axis=0)))
        
#         dh = max(out['stage_caisr_tst'], 0.5)
        
#         if arousal is not None:
#             out['arousal_caisr_rate'] = float(_count_events(arousal, [1]) / dh)
#         if resp is not None:
#             out['resp_caisr_ahi'] = float(_count_events(resp, [1, 2, 3, 4]) / dh)
#             tap = _count_events(resp, [2]) + _count_events(resp, [1])
#             out['resp_central_ratio'] = float(_count_events(resp, [2]) / tap) if tap > 0 else 0.0
#             out['caisr_rdi'] = out['resp_caisr_ahi'] + out.get('arousal_caisr_rate', 0)
#         if limbs is not None:
#             out['limb_caisr_rate'] = float(_count_events(limbs, [1, 2]) / dh)
#             out['limb_isolated_rate'] = float(_count_events(limbs, [1]) / dh)
#             out['limb_periodic_rate'] = float(_count_events(limbs, [2]) / dh)
#             out['caisr_plm_index'] = out['limb_periodic_rate']
        
#         if pn3 is not None: out['caisr_prob_n3_mean'] = float(np.mean(pn3))
#         if pn2 is not None: out['caisr_prob_n2_mean'] = float(np.mean(pn2))
#         if pw  is not None: out['caisr_prob_w_mean']  = float(np.mean(pw))
#         if pr  is not None: out['caisr_prob_r_mean']  = float(np.mean(pr))
        
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
#     return out, stages_out


# def _extract_physio(edf_path, stages=None):
#     """Extract physio features. If stages provided, compute stage-conditional features."""
#     out = {}
#     for w in WINDOWS:
#         for m in EEG_METRICS:
#             out[f'physio_{w}_eeg_{m}'] = np.nan
#         out[f'physio_{w}_emg_rms'] = np.nan
#         out[f'physio_{w}_ecg_hrv'] = np.nan
#         out[f'physio_{w}_ecg_mean_hr'] = np.nan
#         out[f'physio_{w}_resp_freq'] = np.nan
#         out[f'physio_{w}_resp_effort'] = np.nan
#         out[f'physio_{w}_spo2_drop'] = np.nan
#         out[f'physio_{w}_spo2_mean'] = np.nan
#         out[f'physio_{w}_spo2_min'] = np.nan
    
#     # Stage-conditional features
#     out['n3_eeg_delta'] = np.nan
#     out['n3_eeg_theta'] = np.nan
#     out['n3_eeg_sigma'] = np.nan
#     out['n3_eeg_entropy'] = np.nan
#     out['n3_eeg_sef90'] = np.nan
#     out['rem_emg_rms'] = np.nan
#     out['sleep_spo2_mean'] = np.nan
#     out['sleep_spo2_min'] = np.nan
#     out['sleep_spo2_drop'] = np.nan
    
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
        
#         dur = 0
#         for s, f in [(eeg, eeg_fs), (emg, emg_fs), (ecg, ecg_fs),
#                      (rsp, rsp_fs), (eff, eff_fs), (sp2, sp2_fs)]:
#             if s is not None and f > 0:
#                 dur = max(dur, len(s) / f)
#         if dur <= 0:
#             return out
        
#         t3 = dur / 3.0
#         bounds = {'early': (0, t3), 'mid': (t3, 2*t3), 'late': (2*t3, dur)}
        
#         for stage, (st, en) in bounds.items():
#             if eeg is not None and eeg_fs > 0:
#                 sl = eeg[int(st*eeg_fs):int(en*eeg_fs)]
#                 if len(sl) > eeg_fs * 30:
#                     sl = (sl - np.nanmean(sl)) / (np.nanstd(sl) + 1e-8)
#                     ef = _eeg_features_welch(sl, eeg_fs)
#                     for idx, m in enumerate(EEG_METRICS):
#                         out[f'physio_{stage}_eeg_{m}'] = ef[idx]
            
#             if emg is not None and emg_fs > 0:
#                 sl = emg[int(st*emg_fs):int(en*emg_fs)]
#                 if len(sl) > emg_fs * 10:
#                     out[f'physio_{stage}_emg_rms'] = float(
#                         np.sqrt(np.mean(np.square(sl - np.mean(sl)))))
            
#             if ecg is not None and ecg_fs > 0:
#                 sl = ecg[int(st*ecg_fs):int(en*ecg_fs)]
#                 if len(sl) > ecg_fs * 10:
#                     out[f'physio_{stage}_ecg_hrv'] = float(np.var(np.diff(sl))) * (ecg_fs ** 2)
#                     # Simple mean HR estimate
#                     peaks, _ = scipy.signal.find_peaks(sl, distance=int(0.5*ecg_fs),
#                                                         prominence=np.std(sl)*0.3)
#                     if len(peaks) >= 5:
#                         rr = np.diff(peaks) / ecg_fs
#                         rr = rr[(rr > 0.4) & (rr < 1.5)]
#                         if len(rr) > 0:
#                             out[f'physio_{stage}_ecg_mean_hr'] = float(60.0 / np.mean(rr))
            
#             if rsp is not None and rsp_fs > 0:
#                 sl = rsp[int(st*rsp_fs):int(en*rsp_fs)]
#                 if len(sl) > rsp_fs * 10:
#                     out[f'physio_{stage}_resp_freq'] = _resp_spectrum(sl, rsp_fs)[0]
            
#             if eff is not None and eff_fs > 0:
#                 sl = eff[int(st*eff_fs):int(en*eff_fs)]
#                 if len(sl) > eff_fs * 10:
#                     out[f'physio_{stage}_resp_effort'] = float(np.var(sl))
            
#             if sp2 is not None and sp2_fs > 0:
#                 sl = sp2[int(st*sp2_fs):int(en*sp2_fs)]
#                 if len(sl) > sp2_fs * 10:
#                     out[f'physio_{stage}_spo2_drop'] = float(
#                         np.percentile(sl, 95) - np.percentile(sl, 5))
#                     out[f'physio_{stage}_spo2_mean'] = float(np.mean(sl))
#                     out[f'physio_{stage}_spo2_min'] = float(np.min(sl))
        
#         # Stage-conditional features (the secret weapon)
#         if stages is not None:
#             # N3 EEG
#             n3_eeg = _extract_stage_signal(eeg, eeg_fs, stages, target_stage=1)
#             if len(n3_eeg) > eeg_fs * 30:
#                 n3_eeg = (n3_eeg - np.nanmean(n3_eeg)) / (np.nanstd(n3_eeg) + 1e-8)
#                 ef = _eeg_features_welch(n3_eeg, eeg_fs)
#                 out['n3_eeg_delta'] = ef[0]
#                 out['n3_eeg_theta'] = ef[1]
#                 out['n3_eeg_sigma'] = ef[3]
#                 out['n3_eeg_entropy'] = ef[9]
#                 out['n3_eeg_sef90'] = ef[11]
            
#             # REM EMG
#             rem_emg = _extract_stage_signal(emg, emg_fs, stages, target_stage=4)
#             if len(rem_emg) > emg_fs * 30:
#                 out['rem_emg_rms'] = float(
#                     np.sqrt(np.mean(np.square(rem_emg - np.mean(rem_emg)))))
            
#             # Sleep SpO2 (stages 1-4)
#             sleep_sp2 = _extract_stage_signal(sp2, sp2_fs, stages, target_stages=[1,2,3,4])
#             if len(sleep_sp2) > sp2_fs * 30:
#                 out['sleep_spo2_mean'] = float(np.mean(sleep_sp2))
#                 out['sleep_spo2_min'] = float(np.min(sleep_sp2))
#                 out['sleep_spo2_drop'] = float(
#                     np.percentile(sleep_sp2, 95) - np.percentile(sleep_sp2, 5))
    
#     except Exception:
#         pass
#     return out


# def _extract_stage_signal(signal, fs, stages, target_stage=None, target_stages=None):
#     """Extract concatenated signal segments for specified sleep stage(s)."""
#     if signal is None or stages is None or len(stages) == 0 or fs <= 0:
#         return np.array([])
#     if target_stage is not None:
#         target_stages = [target_stage]
#     if target_stages is None:
#         return np.array([])
    
#     epoch_samples = int(30 * fs)
#     segments = []
#     for i, s in enumerate(stages):
#         if s in target_stages:
#             start = i * epoch_samples
#             end = start + epoch_samples
#             if end <= len(signal):
#                 segments.append(signal[start:end])
#     if len(segments) == 0:
#         return np.array([])
#     return np.concatenate(segments)


# # -------------------------------------------------------------------------
# # SPECTRAL & UTILITIES
# # -------------------------------------------------------------------------
# def _eeg_features_welch(signal, fs):
#     if signal is None or len(signal) == 0:
#         return [np.nan] * len(EEG_METRICS)
#     try:
#         nperseg = min(30 * int(fs), len(signal))
#         if nperseg < 2 * int(fs):
#             return [np.nan] * len(EEG_METRICS)
#         freqs, psd = scipy.signal.welch(signal, fs, nperseg=nperseg,
#                                          window='hann', noverlap=nperseg//2)
        
#         ti = (freqs >= 0.5) & (freqs <= 30)
#         tp = np.sum(psd[ti])
#         if tp == 0 or np.isnan(tp):
#             return [np.nan] * len(EEG_METRICS)
        
#         delta = np.sum(psd[(freqs >= 0.5) & (freqs < 4)]) / tp
#         theta = np.sum(psd[(freqs >= 4) & (freqs < 8)]) / tp
#         alpha = np.sum(psd[(freqs >= 8) & (freqs < 12)]) / tp
#         sigma = np.sum(psd[(freqs >= 12) & (freqs < 15)]) / tp
#         beta  = np.sum(psd[(freqs >= 15) & (freqs <= 30)]) / tp
        
#         at = alpha / (theta + 1e-8)
#         tb = theta / (beta + 1e-8)
#         sl = (delta + theta) / (alpha + beta + 1e-8)
#         ds = delta / (sigma + 1e-8)
        
#         pn = psd[ti] / tp
#         ent = scipy.stats.entropy(pn, base=2)
        
#         cp = np.cumsum(psd[ti])
#         sef50 = freqs[ti][np.where(cp >= 0.50 * tp)[0][0]] if np.any(cp >= 0.5*tp) else np.nan
#         sef90 = freqs[ti][np.where(cp >= 0.90 * tp)[0][0]] if np.any(cp >= 0.9*tp) else np.nan
        
#         activity = np.var(signal)
#         d1 = np.diff(signal)
#         d2 = np.diff(d1)
#         mobility = np.sqrt(np.var(d1) / (activity + 1e-8))
#         complexity = np.sqrt(np.var(d2) / (np.var(d1) + 1e-8)) / (mobility + 1e-8)
        
#         return [delta, theta, alpha, sigma, beta, at, tb, sl, ds, ent, sef50, sef90,
#                 activity, mobility, complexity]
#     except Exception:
#         return [np.nan] * len(EEG_METRICS)


# def _resp_spectrum(signal, fs):
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


# def _find_sig(labels, data_dict, fs_dict, target):
#     manifest = {
#         'eeg': ['c3-m2','c4-m1','c3','c4','f3-m2','f4-m1','f3','f4',
#                 'o1-m2','o2-m1','eeg',
#                 'c3-avg','c4-avg','f3-avg','f4-avg','o1-avg','o2-avg',
#                 'e1-m2','e2-m1','e1-avg','e2-avg','e2-m2',
#                 'c3-a2','c4-a1','c3:m2','c4:m1','f3:m2','f4:m1',
#                 'o1-a2','o2-a1','o1:m2','o2:m1',
#                 'loc-a2','roc-a1',
#                 'e1','e2','loc','roc'],
#         'emg': ['chin1-chin2','chin','emg.subm','emg','chin1','emg1','chin2','emg2',
#                 'chin-a','chin-l','chin-r',
#                 'china','chinl','chinr',
#                 'chin emg',
#                 'submental','submentalis','chin-emg'],
#         'ecg': ['ecg','ekg','ecg-la','ecg-v1','ecg i','ecg ii','ecg1',
#                 'ecg2','ekg-l','ekg-r',
#                 'ecg iii','lead ii','lead iii','v1','v2','v3','v4','v5','v6'],
#         'resp_airflow': ['airflow','flow','thermal','thermistor','nasal_pressure',
#                          'nasal','nasaloral','cannula','c-flow','c press','c-pres',
#                          'npt','ptaf','cpap flow','flow_dr'],
#         'resp_effort': ['abd','abdomen','chest','thorax','effort abd','effort tho',
#                         'abdominal','thoracic','effort_abd','effort_tho',
#                         'respitrace abdom','respitrace chest','thoracic'],
#         'spo2': ['spo2','sao2','osat','o2sat','oximetry','pulse ox','pulse_ox'],
#         'eog': ['e1-m2','e2-m1','e1','e2','eog','loc','roc','eog-l','eog-r',
#                 'loc-a2','roc-a1','e1:m2','e2:m1']
#     }
#     for t in manifest.get(target, []):
#         for lbl in labels:
#             if t in lbl:
#                 return data_dict[lbl], fs_dict.get(lbl, 1.0)
#     return None, 1.0


# def _sanitize(sig):
#     if sig is None or len(sig) == 0:
#         return sig
#     mn, mx = np.min(sig), np.max(sig)
#     if mx > 1.0001 or mn < -0.0001:
#         d = mx - mn
#         if d > 1e-6:
#             sig = (sig - mn) / d
#     return np.clip(sig, 0.0, 1.0)


# def _count_events(arr, codes):
#     if arr is None or len(arr) == 0:
#         return 0
#     b = np.isin(arr, codes).astype(int)
#     d = np.diff(b)
#     return max(np.sum(d == 1) + (1 if b[0] == 1 else 0), 0)


# def _age_auroc(y_true, y_prob, ages, delta=2.0):
#     y_true, y_prob, ages = np.asarray(y_true), np.asarray(y_prob), np.asarray(ages)
#     pos = np.where(y_true == 1)[0]
#     neg = np.where(y_true == 0)[0]
#     numer, denom = 0, 0
#     for i in pos:
#         for j in neg:
#             if abs(ages[i] - ages[j]) <= delta:
#                 if y_prob[i] > y_prob[j]:
#                     numer += 1
#                 elif y_prob[i] == y_prob[j]:
#                     numer += 0.5
#                 denom += 1
#     return numer / denom if denom > 0 else np.nan


# def _make_fresh_model(name):
#     if name == 'logistic':
#         return LogisticRegression(C=0.5, penalty='l2', solver='liblinear',
#                                    class_weight='balanced', max_iter=1000, random_state=42)
#     elif name == 'logistic_l1':
#         return LogisticRegression(C=0.1, penalty='l1', solver='liblinear',
#                                    class_weight='balanced', max_iter=1000, random_state=42)
#     elif name == 'lgb_shallow' and lgb is not None:
#         return lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, num_leaves=15,
#                                    min_child_samples=10, reg_lambda=1,
#                                    random_state=42, n_jobs=1, verbose=-1,
#                                    class_weight='balanced')
#     elif name == 'lgb_medium' and lgb is not None:
#         return lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, num_leaves=31,
#                                    min_child_samples=10, reg_lambda=1,
#                                    random_state=42, n_jobs=1, verbose=-1,
#                                    class_weight='balanced')
#     elif name == 'svm_lin':
#         return SVC(probability=True, C=0.5, kernel='linear',
#                    class_weight='balanced', random_state=42)
#     elif name == 'stack':
#         return StackingClassifier(
#             estimators=[
#                 ('lr1', LogisticRegression(C=0.05, class_weight='balanced', solver='liblinear', max_iter=1000)),
#                 ('svm', SVC(probability=True, C=0.1, kernel='linear', class_weight='balanced'))
#             ],
#             final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
#             n_jobs=1, passthrough=False
#         )
#     else:
#         raise ValueError(f"Unknown model name: {name}")
    

# #!/usr/bin/env python
# PhysioNet Challenge 2026 - Age-Conditioned AUROC Optimized v4
# Wins by: age-residualized features + stage-conditional spectral + ensemble weight optimization
# """

# import os
# import warnings
# import numpy as np
# import pandas as pd
# import scipy.stats
# import scipy.signal
# from tqdm import tqdm
# import joblib

# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import StackingClassifier
# from sklearn.svm import SVC
# from scipy.optimize import minimize

# try:
#     import lightgbm as lgb
# except ImportError:
#     lgb = None

# warnings.filterwarnings("ignore")
# from helper_code import *

# # -------------------------------------------------------------------------
# # CONFIGURATION
# # -------------------------------------------------------------------------
# FINAL_MODEL = "auto"
# WINDOWS = ['early', 'mid', 'late']
# EEG_METRICS = ['delta','theta','alpha','sigma','beta','alpha_theta',
#                'theta_beta','slowing','delta_sigma','entropy','sef50','sef90',
#                'hjorth_activity','hjorth_mobility','hjorth_complexity']

# # -------------------------------------------------------------------------
# # REQUIRED FUNCTIONS
# # -------------------------------------------------------------------------
# def train_model(data_folder, model_folder, verbose):
#     if verbose:
#         print('Finding Challenge data...')
    
#     patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
#     patient_metadata_list = find_patients(patient_data_file)
#     num_records = len(patient_metadata_list)
    
#     if num_records == 0:
#         raise FileNotFoundError('No data provided.')
    
#     if verbose:
#         print(f'Found {num_records} records. Extracting features...')
    
#     # PHASE 1: Extract features
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
    
#     # PHASE 2: Temporal-difference features (late vs early)
#     for m in EEG_METRICS + ['emg_rms','ecg_hrv','ecg_mean_hr','resp_freq',
#                             'resp_effort','spo2_drop','spo2_mean','spo2_min']:
#         e = f'physio_early_{m}'
#         l = f'physio_late_{m}'
#         if e in df.columns and l in df.columns:
#             df[f'physio_delta_{m}'] = df[l] - df[e]
#             feature_names.append(f'physio_delta_{m}')
    
#     # PHASE 3: Poison filtering (directional site instability)
#     if verbose:
#         print('Filtering site-poisonous features...')
    
#     sites = df['site'].unique()
#     poison_features = []
    
#     for col in feature_names:
#         if col.startswith('inter_') or col.startswith('physio_delta_'):
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
    
#     # PHASE 4: Interaction features
#     interactions = [
#         ('resp_caisr_ahi', 'physio_late_spo2_drop', 'inter_AHI_SpO2'),
#         ('caisr_prob_w_mean', 'physio_early_eeg_delta', 'inter_WASO_SWA'),
#         ('caisr_prob_r_mean', 'physio_late_emg_rms', 'inter_REM_EMG_Atonia'),
#         ('physio_late_ecg_hrv', 'physio_late_resp_effort', 'inter_HRV_RespEffort'),
#         ('physio_late_eeg_theta_beta', 'physio_late_resp_freq', 'inter_ThetaBeta_RespFreq'),
#         ('physio_mid_eeg_delta_sigma', 'physio_mid_spo2_drop', 'inter_mid_DeltaSigma_SpO2'),
#         ('physio_mid_eeg_theta_beta', 'physio_mid_resp_effort', 'inter_mid_ThetaBeta_Effort'),
#     ]
    
#     age_interactions = []
#     if 'age' in kept:
#         age_interactions.append(('age', 'age', 'age_sq'))
#         for partner in ['resp_caisr_ahi', 'caisr_n3_pct', 'caisr_prob_n3_mean',
#                         'physio_early_eeg_delta', 'physio_late_eeg_entropy', 'bmi']:
#             if partner in kept:
#                 age_interactions.append(('age', partner, f'inter_age_{partner}'))
    
#     for f1, f2, name in interactions + age_interactions:
#         if f1 in kept and f2 in kept:
#             df[name] = df[f1] * df[f2] if f1 != f2 else df[f1] ** 2
#             if name not in kept:
#                 kept.append(name)
    
#     # PHASE 5: Age residualization (skip demographics & age interactions)
#     if verbose:
#         print('Age-residualizing physiology features...')
    
#     demo_cols = {'age', 'sex_male', 'sex_female', 'race_white', 'race_black',
#                  'race_asian', 'race_other', 'race_unavailable'}
#     resid_cols = []
#     age_resid_models = {}
    
#     for col in kept:
#         if col in demo_cols or col.startswith('inter_age_') or col == 'age_sq':
#             continue
#         sub = df.dropna(subset=[col, 'age'])
#         if len(sub) > 10:
#             lr = LinearRegression()
#             lr.fit(sub[['age']].values, sub[col].values)
#             df[f"{col}_resid"] = df[col] - lr.predict(df[['age']].values)
#             resid_cols.append(f"{col}_resid")
#             age_resid_models[col] = lr
    
#     # PHASE 6: Impute & scale
#     final_cols = list(demo_cols.intersection(kept)) + [c for c in kept if c.startswith('inter_')] + ['age_sq'] + resid_cols
#     # Ensure all exist
#     for col in final_cols:
#         if col not in df.columns:
#             df[col] = np.nan
#         med = df[col].median()
#         if np.isnan(med):
#             med = 0.0
#         df[col] = df[col].fillna(med)
    
#     X = df[final_cols].values
#     y = df['label'].values
#     ages = df['age'].values
#     sites_arr = df['site'].values
    
#     imputer = SimpleImputer(strategy='median')
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(imputer.fit_transform(X))
    
#     # PHASE 7: Train candidate models
#     if verbose:
#         print('Training candidate models...')
    
#     candidates = {}
    
#     candidates['logistic'] = LogisticRegression(
#         C=0.5, penalty='l2', solver='liblinear',
#         class_weight='balanced', max_iter=1000, random_state=42)
#     candidates['logistic'].fit(Xs, y)
    
#     candidates['logistic_l1'] = LogisticRegression(
#         C=0.1, penalty='l1', solver='liblinear',
#         class_weight='balanced', max_iter=1000, random_state=42)
#     candidates['logistic_l1'].fit(Xs, y)
    
#     if lgb is not None:
#         candidates['lgb_shallow'] = lgb.LGBMClassifier(
#             n_estimators=300, max_depth=3, learning_rate=0.05, num_leaves=15,
#             min_child_samples=10, reg_lambda=1,
#             random_state=42, n_jobs=1, verbose=-1,
#             class_weight='balanced')
#         candidates['lgb_shallow'].fit(Xs, y)
        
#         candidates['lgb_medium'] = lgb.LGBMClassifier(
#             n_estimators=300, max_depth=4, learning_rate=0.05, num_leaves=31,
#             min_child_samples=10, reg_lambda=1,
#             random_state=42, n_jobs=1, verbose=-1,
#             class_weight='balanced')
#         candidates['lgb_medium'].fit(Xs, y)
    
#     candidates['svm_lin'] = SVC(
#         probability=True, C=0.5, kernel='linear',
#         class_weight='balanced', random_state=42)
#     candidates['svm_lin'].fit(Xs, y)
    
#     candidates['stack'] = StackingClassifier(
#         estimators=[
#             ('lr1', LogisticRegression(C=0.05, class_weight='balanced', solver='liblinear', max_iter=1000)),
#             ('svm', SVC(probability=True, C=0.1, kernel='linear', class_weight='balanced'))
#         ],
#         final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
#         n_jobs=1, passthrough=False)
#     candidates['stack'].fit(Xs, y)
    
#     # PHASE 8: LOSO CV + ensemble weight optimization on age-AUROC
#     names = list(candidates.keys())
#     oof_probs = {name: np.full(len(y), np.nan) for name in names}
    
#     if verbose:
#         print('Running site-aware LOSO CV...')
    
#     for test_site in sites:
#         train_mask = sites_arr != test_site
#         test_mask = sites_arr == test_site
        
#         if np.sum(test_mask) < 5 or np.sum(train_mask) < 20:
#             continue
#         if len(np.unique(y[test_mask])) < 2:
#             continue
        
#         X_tr, X_val = Xs[train_mask], Xs[test_mask]
#         y_tr, y_val = y[train_mask], y[test_mask]
#         a_val = ages[test_mask]
#         test_idx = np.where(test_mask)[0]
        
#         for name in names:
#             m = _make_fresh_model(name)
#             m.fit(X_tr, y_tr)
#             p = m.predict_proba(X_val)[:, 1]
#             oof_probs[name][test_idx] = p
    
#     # Report individual scores
#     if verbose:
#         for name in names:
#             valid = ~np.isnan(oof_probs[name])
#             if np.sum(valid) > 0:
#                 sc = _age_auroc(y[valid], oof_probs[name][valid], ages[valid], delta=2.0)
#                 print(f"  {name:15s}: LOSO age-AUROC = {sc:.4f}")
    
#     # Optimize ensemble weights to maximize age-AUROC
#     def _ens_score(weights):
#         w = np.maximum(weights, 0)
#         if np.sum(w) == 0:
#             return 0.0
#         w = w / np.sum(w)
#         ens = np.zeros(len(y))
#         for i, name in enumerate(names):
#             valid = ~np.isnan(oof_probs[name])
#             ens[valid] += w[i] * oof_probs[name][valid]
#         valid = ~np.isnan(ens)
#         if np.sum(valid) == 0:
#             return 0.0
#         return _age_auroc(y[valid], ens[valid], ages[valid], delta=2.0)
    
#     x0 = np.ones(len(names)) / len(names)
#     bounds = [(0.0, 1.0)] * len(names)
#     cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
#     result = minimize(lambda w: -_ens_score(w), x0,
#                       method='SLSQP', bounds=bounds, constraints=cons,
#                       options={'maxiter': 200})
    
#     best_w = np.maximum(result.x, 0)
#     best_w = best_w / np.sum(best_w)
#     ensemble_weights = dict(zip(names, best_w))
    
#     if verbose:
#         print(f"Optimized ensemble LOSO age-AUROC = {_ens_score(best_w):.4f}")
#         for name, w in zip(names, best_w):
#             print(f"  {name:15s}: weight = {w:.3f}")
    
#     # PHASE 9: Save artifact
#     os.makedirs(model_folder, exist_ok=True)
    
#     artifact = {
#         'models': candidates,
#         'ensemble_weights': ensemble_weights,
#         'scaler': scaler,
#         'imputer': imputer,
#         'final_cols': final_cols,
#         'resid_cols': resid_cols,
#         'age_resid_models': age_resid_models,
#         'interactions': interactions + age_interactions,
#         'sites_seen': list(sites),
#         'median_age': float(np.median(ages)),
#     }
    
#     joblib.dump(artifact, os.path.join(model_folder, 'model.sav'))
#     if verbose:
#         print('Training complete. Model saved.')


# def load_model(model_folder, verbose):
#     return joblib.load(os.path.join(model_folder, 'model.sav'))


# def run_model(model_artifact, record, data_folder, verbose):
#     patient_id = record[HEADERS['bids_folder']]
#     site_id = record[HEADERS['site_id']]
#     session_id = record[HEADERS['session_id']]
    
#     patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
#     patient_data = load_demographics(patient_data_file, patient_id, session_id)
    
#     feats = extract_all_features(data_folder, patient_id, site_id, session_id, patient_data)
#     if feats is None:
#         return float('nan'), float('nan')
    
#     df = pd.DataFrame([feats])
#     age = load_age(patient_data)
#     if np.isnan(age):
#         age = model_artifact.get('median_age', 65.0)
#     df['age'] = age
    
#     # Build interactions
#     for f1, f2, name in model_artifact.get('interactions', []):
#         if f1 in df.columns and f2 in df.columns:
#             df[name] = df[f1] * df[f2] if f1 != f2 else df[f1] ** 2
    
#     # Build temporal deltas
#     for m in EEG_METRICS + ['emg_rms','ecg_hrv','ecg_mean_hr','resp_freq',
#                             'resp_effort','spo2_drop','spo2_mean','spo2_min']:
#         e = f'physio_early_{m}'
#         l = f'physio_late_{m}'
#         if e in df.columns and l in df.columns:
#             df[f'physio_delta_{m}'] = df[l] - df[e]
    
#     # Age residualization
#     for col, lr in model_artifact.get('age_resid_models', {}).items():
#         if col in df.columns:
#             df[f"{col}_resid"] = df[col].values - lr.predict(np.array([[age]]))[0]
    
#     final_cols = model_artifact['final_cols']
#     for col in final_cols:
#         if col not in df.columns:
#             df[col] = float('nan')
#         med = df[col].median()
#         if np.isnan(med):
#             med = 0.0
#         df[col] = df[col].fillna(med)
    
#     X = df[final_cols].values
#     Xs = model_artifact['scaler'].transform(model_artifact['imputer'].transform(X))
    
#     # Ensemble prediction
#     weights = model_artifact['ensemble_weights']
#     prob = 0.0
#     for name, w in weights.items():
#         if w > 0 and name in model_artifact['models']:
#             prob += w * model_artifact['models'][name].predict_proba(Xs)[0, 1]
    
#     # Threshold irrelevant for AUROC ranking; use 0.5 for API compliance
#     binary = int(prob >= 0.5)
#     return binary, float(prob)


# # -------------------------------------------------------------------------
# # FEATURE EXTRACTION
# # -------------------------------------------------------------------------
# def extract_all_features(data_folder, patient_id, site_id, session_id, patient_data):
#     features = {}
    
#     # 1. Demographics
#     features['age'] = load_age(patient_data)
#     sex = load_sex(patient_data, standardize=True)
#     features['sex_male'] = 1 if sex == 'Male' else 0
#     features['sex_female'] = 1 if sex == 'Female' else 0
#     race = load_race(patient_data, standardize=True)
#     features['race_white'] = 1 if race == 'White' else 0
#     features['race_black'] = 1 if race == 'Black' else 0
#     features['race_asian'] = 1 if race == 'Asian' else 0
#     features['race_other'] = 1 if race == 'Others' else 0
#     features['race_unavailable'] = 1 if race == 'Unavailable' else 0
#     features['bmi'] = load_bmi(patient_data)
    
#     # 2. CAISR
#     caisr_path = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
#                               site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
#     caisr_features, caisr_stages = _extract_caisr(caisr_path)
#     features.update(caisr_features)
    
#     # 3. Physiology
#     physio_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
#                                site_id, f"{patient_id}_ses-{session_id}.edf")
#     features.update(_extract_physio(physio_path, stages=caisr_stages))
    
#     return features


# def _extract_caisr(edf_path):
#     """Return (features_dict, stages_array_or_None)."""
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
#         'caisr_n3_pct': np.nan, 'caisr_n2_pct': np.nan,
#         'caisr_n1_pct': np.nan, 'caisr_rem_pct': np.nan,
#         'caisr_wake_pct': np.nan, 'caisr_sleep_efficiency': np.nan,
#         'caisr_sleep_latency_min': np.nan, 'caisr_rem_latency_min': np.nan,
#         'caisr_waso_min': np.nan, 'caisr_rdi': np.nan,
#         'caisr_stage_prob_var': np.nan, 'caisr_n1_sleep_pct': np.nan,
#         'caisr_plm_index': np.nan,
#     }
#     stages_out = None
    
#     if not os.path.exists(edf_path):
#         return out, stages_out
    
#     try:
#         data_dict, fs_dict = load_signal_data(edf_path)
#         if not data_dict:
#             return out, stages_out
        
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
        
#         if stages is not None and len(stages) > 0:
#             stages_out = stages
#             epoch_dur_min = 0.5
#             n_epochs = len(stages)
#             tst = (n_epochs * 30) / 3600
#             out['stage_caisr_tst'] = tst
            
#             valid = stages[stages < 9]
#             if len(valid) > 0:
#                 out['caisr_wake_pct'] = float(np.mean(valid == 5))
#                 out['caisr_n1_pct'] = float(np.mean(valid == 3))
#                 out['caisr_n2_pct'] = float(np.mean(valid == 2))
#                 out['caisr_n3_pct'] = float(np.mean(valid == 1))
#                 out['caisr_rem_pct'] = float(np.mean(valid == 4))
#                 out['stage_caisr_se'] = float(np.mean((valid >= 1) & (valid <= 4)))
#                 out['caisr_sleep_efficiency'] = out['stage_caisr_se']
                
#                 sleep_idx = np.where(np.isin(valid, [1, 2, 3, 4]))[0]
#                 if len(sleep_idx) > 0:
#                     out['caisr_sleep_latency_min'] = float(sleep_idx[0] * epoch_dur_min)
#                 else:
#                     out['caisr_sleep_latency_min'] = float(len(valid) * epoch_dur_min)
                
#                 rem_idx = np.where(valid == 4)[0]
#                 if len(rem_idx) > 0:
#                     out['caisr_rem_latency_min'] = float(rem_idx[0] * epoch_dur_min)
                
#                 sleep_started = False
#                 wake_epochs = 0
#                 for s in valid:
#                     if not sleep_started and s in [1, 2, 3, 4]:
#                         sleep_started = True
#                     if sleep_started and s == 5:
#                         wake_epochs += 1
#                 out['caisr_waso_min'] = float(wake_epochs * epoch_dur_min)
                
#                 sleep_stages = valid[np.isin(valid, [1, 2, 3, 4])]
#                 if len(sleep_stages) > 0:
#                     out['caisr_n1_sleep_pct'] = float(np.mean(sleep_stages == 3))
            
#             out['stage_transition_rate'] = float(np.sum(np.diff(stages) != 0) / max(tst, 0.5))
            
#             stage_probs = []
#             for p in [pn3, pn2, pn1, pr, pw]:
#                 if p is not None and len(p) == len(stages):
#                     stage_probs.append(p)
#             if len(stage_probs) > 0:
#                 sp_arr = np.array(stage_probs)
#                 out['caisr_stage_prob_var'] = float(np.mean(np.var(sp_arr, axis=0)))
        
#         dh = max(out['stage_caisr_tst'], 0.5)
        
#         if arousal is not None:
#             out['arousal_caisr_rate'] = float(_count_events(arousal, [1]) / dh)
#         if resp is not None:
#             out['resp_caisr_ahi'] = float(_count_events(resp, [1, 2, 3, 4]) / dh)
#             tap = _count_events(resp, [2]) + _count_events(resp, [1])
#             out['resp_central_ratio'] = float(_count_events(resp, [2]) / tap) if tap > 0 else 0.0
#             out['caisr_rdi'] = out['resp_caisr_ahi'] + out.get('arousal_caisr_rate', 0)
#         if limbs is not None:
#             out['limb_caisr_rate'] = float(_count_events(limbs, [1, 2]) / dh)
#             out['limb_isolated_rate'] = float(_count_events(limbs, [1]) / dh)
#             out['limb_periodic_rate'] = float(_count_events(limbs, [2]) / dh)
#             out['caisr_plm_index'] = out['limb_periodic_rate']
        
#         if pn3 is not None: out['caisr_prob_n3_mean'] = float(np.mean(pn3))
#         if pn2 is not None: out['caisr_prob_n2_mean'] = float(np.mean(pn2))
#         if pw  is not None: out['caisr_prob_w_mean']  = float(np.mean(pw))
#         if pr  is not None: out['caisr_prob_r_mean']  = float(np.mean(pr))
        
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
#     return out, stages_out


# def _extract_physio(edf_path, stages=None):
#     """Extract physio features. If stages provided, compute stage-conditional features."""
#     out = {}
#     for w in WINDOWS:
#         for m in EEG_METRICS:
#             out[f'physio_{w}_eeg_{m}'] = np.nan
#         out[f'physio_{w}_emg_rms'] = np.nan
#         out[f'physio_{w}_ecg_hrv'] = np.nan
#         out[f'physio_{w}_ecg_mean_hr'] = np.nan
#         out[f'physio_{w}_resp_freq'] = np.nan
#         out[f'physio_{w}_resp_effort'] = np.nan
#         out[f'physio_{w}_spo2_drop'] = np.nan
#         out[f'physio_{w}_spo2_mean'] = np.nan
#         out[f'physio_{w}_spo2_min'] = np.nan
    
#     # Stage-conditional features
#     out['n3_eeg_delta'] = np.nan
#     out['n3_eeg_theta'] = np.nan
#     out['n3_eeg_sigma'] = np.nan
#     out['n3_eeg_entropy'] = np.nan
#     out['n3_eeg_sef90'] = np.nan
#     out['rem_emg_rms'] = np.nan
#     out['sleep_spo2_mean'] = np.nan
#     out['sleep_spo2_min'] = np.nan
#     out['sleep_spo2_drop'] = np.nan
    
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
        
#         dur = 0
#         for s, f in [(eeg, eeg_fs), (emg, emg_fs), (ecg, ecg_fs),
#                      (rsp, rsp_fs), (eff, eff_fs), (sp2, sp2_fs)]:
#             if s is not None and f > 0:
#                 dur = max(dur, len(s) / f)
#         if dur <= 0:
#             return out
        
#         t3 = dur / 3.0
#         bounds = {'early': (0, t3), 'mid': (t3, 2*t3), 'late': (2*t3, dur)}
        
#         for stage, (st, en) in bounds.items():
#             if eeg is not None and eeg_fs > 0:
#                 sl = eeg[int(st*eeg_fs):int(en*eeg_fs)]
#                 if len(sl) > eeg_fs * 30:
#                     sl = (sl - np.nanmean(sl)) / (np.nanstd(sl) + 1e-8)
#                     ef = _eeg_features_welch(sl, eeg_fs)
#                     for idx, m in enumerate(EEG_METRICS):
#                         out[f'physio_{stage}_eeg_{m}'] = ef[idx]
            
#             if emg is not None and emg_fs > 0:
#                 sl = emg[int(st*emg_fs):int(en*emg_fs)]
#                 if len(sl) > emg_fs * 10:
#                     out[f'physio_{stage}_emg_rms'] = float(
#                         np.sqrt(np.mean(np.square(sl - np.mean(sl)))))
            
#             if ecg is not None and ecg_fs > 0:
#                 sl = ecg[int(st*ecg_fs):int(en*ecg_fs)]
#                 if len(sl) > ecg_fs * 10:
#                     out[f'physio_{stage}_ecg_hrv'] = float(np.var(np.diff(sl)))
#                     # Simple mean HR estimate
#                     peaks, _ = scipy.signal.find_peaks(sl, distance=int(0.5*ecg_fs),
#                                                         prominence=np.std(sl)*0.3)
#                     if len(peaks) >= 5:
#                         rr = np.diff(peaks) / ecg_fs
#                         rr = rr[(rr > 0.4) & (rr < 1.5)]
#                         if len(rr) > 0:
#                             out[f'physio_{stage}_ecg_mean_hr'] = float(60.0 / np.mean(rr))
            
#             if rsp is not None and rsp_fs > 0:
#                 sl = rsp[int(st*rsp_fs):int(en*rsp_fs)]
#                 if len(sl) > rsp_fs * 10:
#                     out[f'physio_{stage}_resp_freq'] = _resp_spectrum(sl, rsp_fs)[0]
            
#             if eff is not None and eff_fs > 0:
#                 sl = eff[int(st*eff_fs):int(en*eff_fs)]
#                 if len(sl) > eff_fs * 10:
#                     out[f'physio_{stage}_resp_effort'] = float(np.var(sl))
            
#             if sp2 is not None and sp2_fs > 0:
#                 sl = sp2[int(st*sp2_fs):int(en*sp2_fs)]
#                 if len(sl) > sp2_fs * 10:
#                     out[f'physio_{stage}_spo2_drop'] = float(
#                         np.percentile(sl, 95) - np.percentile(sl, 5))
#                     out[f'physio_{stage}_spo2_mean'] = float(np.mean(sl))
#                     out[f'physio_{stage}_spo2_min'] = float(np.min(sl))
        
#         # Stage-conditional features (the secret weapon)
#         if stages is not None:
#             # N3 EEG
#             n3_eeg = _extract_stage_signal(eeg, eeg_fs, stages, target_stage=1)
#             if len(n3_eeg) > eeg_fs * 30:
#                 n3_eeg = (n3_eeg - np.nanmean(n3_eeg)) / (np.nanstd(n3_eeg) + 1e-8)
#                 ef = _eeg_features_welch(n3_eeg, eeg_fs)
#                 out['n3_eeg_delta'] = ef[0]
#                 out['n3_eeg_theta'] = ef[1]
#                 out['n3_eeg_sigma'] = ef[3]
#                 out['n3_eeg_entropy'] = ef[9]
#                 out['n3_eeg_sef90'] = ef[11]
            
#             # REM EMG
#             rem_emg = _extract_stage_signal(emg, emg_fs, stages, target_stage=4)
#             if len(rem_emg) > emg_fs * 30:
#                 out['rem_emg_rms'] = float(
#                     np.sqrt(np.mean(np.square(rem_emg - np.mean(rem_emg)))))
            
#             # Sleep SpO2 (stages 1-4)
#             sleep_sp2 = _extract_stage_signal(sp2, sp2_fs, stages, target_stages=[1,2,3,4])
#             if len(sleep_sp2) > sp2_fs * 30:
#                 out['sleep_spo2_mean'] = float(np.mean(sleep_sp2))
#                 out['sleep_spo2_min'] = float(np.min(sleep_sp2))
#                 out['sleep_spo2_drop'] = float(
#                     np.percentile(sleep_sp2, 95) - np.percentile(sleep_sp2, 5))
    
#     except Exception:
#         pass
#     return out


# def _extract_stage_signal(signal, fs, stages, target_stage=None, target_stages=None):
#     """Extract concatenated signal segments for specified sleep stage(s)."""
#     if signal is None or stages is None or len(stages) == 0 or fs <= 0:
#         return np.array([])
#     if target_stage is not None:
#         target_stages = [target_stage]
#     if target_stages is None:
#         return np.array([])
    
#     epoch_samples = int(30 * fs)
#     segments = []
#     for i, s in enumerate(stages):
#         if s in target_stages:
#             start = i * epoch_samples
#             end = start + epoch_samples
#             if end <= len(signal):
#                 segments.append(signal[start:end])
#     if len(segments) == 0:
#         return np.array([])
#     return np.concatenate(segments)


# # -------------------------------------------------------------------------
# # SPECTRAL & UTILITIES
# # -------------------------------------------------------------------------
# def _eeg_features_welch(signal, fs):
#     if signal is None or len(signal) == 0:
#         return [np.nan] * len(EEG_METRICS)
#     try:
#         nperseg = min(30 * int(fs), len(signal))
#         if nperseg < 2 * int(fs):
#             return [np.nan] * len(EEG_METRICS)
#         freqs, psd = scipy.signal.welch(signal, fs, nperseg=nperseg,
#                                          window='hann', noverlap=nperseg//2)
        
#         ti = (freqs >= 0.5) & (freqs <= 30)
#         tp = np.sum(psd[ti])
#         if tp == 0 or np.isnan(tp):
#             return [np.nan] * len(EEG_METRICS)
        
#         delta = np.sum(psd[(freqs >= 0.5) & (freqs < 4)]) / tp
#         theta = np.sum(psd[(freqs >= 4) & (freqs < 8)]) / tp
#         alpha = np.sum(psd[(freqs >= 8) & (freqs < 12)]) / tp
#         sigma = np.sum(psd[(freqs >= 12) & (freqs < 15)]) / tp
#         beta  = np.sum(psd[(freqs >= 15) & (freqs <= 30)]) / tp
        
#         at = alpha / (theta + 1e-8)
#         tb = theta / (beta + 1e-8)
#         sl = (delta + theta) / (alpha + beta + 1e-8)
#         ds = delta / (sigma + 1e-8)
        
#         pn = psd[ti] / tp
#         ent = scipy.stats.entropy(pn, base=2)
        
#         cp = np.cumsum(psd[ti])
#         sef50 = freqs[ti][np.where(cp >= 0.50 * tp)[0][0]] if np.any(cp >= 0.5*tp) else np.nan
#         sef90 = freqs[ti][np.where(cp >= 0.90 * tp)[0][0]] if np.any(cp >= 0.9*tp) else np.nan
        
#         activity = np.var(signal)
#         d1 = np.diff(signal)
#         d2 = np.diff(d1)
#         mobility = np.sqrt(np.var(d1) / (activity + 1e-8))
#         complexity = np.sqrt(np.var(d2) / (np.var(d1) + 1e-8)) / (mobility + 1e-8)
        
#         return [delta, theta, alpha, sigma, beta, at, tb, sl, ds, ent, sef50, sef90,
#                 activity, mobility, complexity]
#     except Exception:
#         return [np.nan] * len(EEG_METRICS)


# def _resp_spectrum(signal, fs):
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


# def _find_sig(labels, data_dict, fs_dict, target):
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
#     if sig is None or len(sig) == 0:
#         return sig
#     mn, mx = np.min(sig), np.max(sig)
#     if mx > 1.0001 or mn < -0.0001:
#         d = mx - mn
#         if d > 1e-6:
#             sig = (sig - mn) / d
#     return np.clip(sig, 0.0, 1.0)


# def _count_events(arr, codes):
#     if arr is None or len(arr) == 0:
#         return 0
#     b = np.isin(arr, codes).astype(int)
#     d = np.diff(b)
#     return max(np.sum(d == 1) + (1 if b[0] == 1 else 0), 0)


# def _age_auroc(y_true, y_prob, ages, delta=2.0):
#     y_true, y_prob, ages = np.asarray(y_true), np.asarray(y_prob), np.asarray(ages)
#     pos = np.where(y_true == 1)[0]
#     neg = np.where(y_true == 0)[0]
#     numer, denom = 0, 0
#     for i in pos:
#         for j in neg:
#             if abs(ages[i] - ages[j]) <= delta:
#                 if y_prob[i] > y_prob[j]:
#                     numer += 1
#                 elif y_prob[i] == y_prob[j]:
#                     numer += 0.5
#                 denom += 1
#     return numer / denom if denom > 0 else np.nan


# def _make_fresh_model(name):
#     if name == 'logistic':
#         return LogisticRegression(C=0.5, penalty='l2', solver='liblinear',
#                                    class_weight='balanced', max_iter=1000, random_state=42)
#     elif name == 'logistic_l1':
#         return LogisticRegression(C=0.1, penalty='l1', solver='liblinear',
#                                    class_weight='balanced', max_iter=1000, random_state=42)
#     elif name == 'lgb_shallow' and lgb is not None:
#         return lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, num_leaves=15,
#                                    min_child_samples=10, reg_lambda=1,
#                                    random_state=42, n_jobs=1, verbose=-1,
#                                    class_weight='balanced')
#     elif name == 'lgb_medium' and lgb is not None:
#         return lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, num_leaves=31,
#                                    min_child_samples=10, reg_lambda=1,
#                                    random_state=42, n_jobs=1, verbose=-1,
#                                    class_weight='balanced')
#     elif name == 'svm_lin':
#         return SVC(probability=True, C=0.5, kernel='linear',
#                    class_weight='balanced', random_state=42)
#     elif name == 'stack':
#         return StackingClassifier(
#             estimators=[
#                 ('lr1', LogisticRegression(C=0.05, class_weight='balanced', solver='liblinear', max_iter=1000)),
#                 ('svm', SVC(probability=True, C=0.1, kernel='linear', class_weight='balanced'))
#             ],
#             final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
#             n_jobs=1, passthrough=False
#         )
#     else:
#         raise ValueError(f"Unknown model name: {name}")



# current 0.5581 auroc local
# #Extract: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 194/194 [2:24:50<00:00, 44.80s/rec, id=sub-I0006179002878]
# Extracted 194 records, 231 raw features.
# Prevalence: 0.088 | Sites: 3
# Filtering site-poisonous features...
#   Kept 199 | Dropped 32 poison
# Age-residualizing features...
# Training candidate models...
# Running site-aware LOSO CV for model selection...
#   logistic    : LOSO age-AUROC = 0.5390 (n=3 folds)
#   lightgbm    : LOSO age-AUROC = 0.5581 (n=3 folds)
#   stack       : LOSO age-AUROC = 0.4045 (n=3 folds)
#   mega        : LOSO age-AUROC = 0.5371 (n=3 folds)
# Selected: lightgbm (LOSO age-AUROC = 0.5581)
# Training complete. Model saved.
# Training complete. Model saved.
# # # #!/usr/bin/env python



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

# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import StackingClassifier, VotingClassifier
# from sklearn.svm import SVC

# try:
#     import lightgbm as lgb
# except ImportError:
#     lgb = None

# import torch
# import torch.nn as nn

# warnings.filterwarnings("ignore")
# from helper_code import *

# FINAL_MODEL = "auto"
# N_CV_FOLDS = 5
# WINDOWS = ['early', 'mid', 'late']
# EEG_METRICS = ['delta','theta','alpha','sigma','beta','alpha_theta',
#                'theta_beta','slowing','delta_sigma','entropy','sef50','sef90']

# SLEEPFM_BASE_PATH = os.path.join(os.path.dirname(__file__), 'sleepfm')
# SLEEPFM_MODEL_PATH = os.path.join(SLEEPFM_BASE_PATH, 'checkpoints', 'model_base')
# SLEEPFM_CONFIG_PATH = os.path.join(SLEEPFM_MODEL_PATH, 'config.json')
# SLEEPFM_CHANNEL_GROUPS_PATH = os.path.join(SLEEPFM_BASE_PATH, 'configs', 'channel_groups_challenge.json')


# class SleepFMFeatureExtractor:
#     """SleepFM extractor with memory-safe EDF handling."""

#     def __init__(self, verbose=False):
#         self.verbose = verbose
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = None
#         self.model_config = None
#         self.channel_groups = None
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

#             if not os.path.exists(SLEEPFM_CHANNEL_GROUPS_PATH):
#                 if self.verbose:
#                     print(f"Channel groups not found at {SLEEPFM_CHANNEL_GROUPS_PATH}")
#                 return

#             with open(SLEEPFM_CHANNEL_GROUPS_PATH, 'r') as f:
#                 self.channel_groups = json.load(f)

#             repo_root = os.path.dirname(__file__)
#             if repo_root not in sys.path:
#                 sys.path.insert(0, repo_root)

#             try:
#                 from sleepfm.models.models import SetTransformer
#                 model_class = SetTransformer
#             except ImportError as e:
#                 if self.verbose:
#                     print(f"Could not import SleepFM model: {e}")
#                 self.model = None
#                 return

#             in_channels = self.model_config.get('in_channels', 1)
#             patch_size = self.model_config.get('patch_size', 640)
#             embed_dim = self.model_config.get('embed_dim', 128)
#             num_heads = self.model_config.get('num_heads', 8)
#             num_layers = self.model_config.get('num_layers', 6)
#             pooling_head = self.model_config.get('pooling_head', 8)
#             dropout = self.model_config.get('dropout', 0.0)

#             self.model = model_class(
#                 in_channels=in_channels,
#                 patch_size=patch_size,
#                 embed_dim=embed_dim,
#                 num_heads=num_heads,
#                 num_layers=num_layers,
#                 pooling_head=pooling_head,
#                 dropout=dropout,
#                 max_seq_length=128
#             )

#             weights_path = os.path.join(SLEEPFM_MODEL_PATH, 'best.pt')
#             if not os.path.exists(weights_path):
#                 if self.verbose:
#                     print(f"SleepFM weights not found at {weights_path}")
#                 self.model = None
#                 return

#             checkpoint = torch.load(weights_path, map_location=self.device)
#             state_dict = checkpoint.get('state_dict', checkpoint)

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

#     def _edf_to_hdf5(self, edf_path, temp_dir):
#         """Convert EDF to HDF5 WITHOUT loading entire file into memory."""
#         try:
#             import mne
#             import h5py

#             # CRITICAL: preload=False means header only
#             raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

#             # Resample in-place (affects header, not full load)
#             if abs(raw.info['sfreq'] - 128.0) > 0.5:
#                 raw.resample(128.0)

#             subject_id = os.path.splitext(os.path.basename(edf_path))[0]
#             hdf5_path = os.path.join(temp_dir, f"{subject_id}_psg.hdf5")

#             with h5py.File(hdf5_path, 'w') as f:
#                 for mod, channel_list in self.channel_groups.items():
#                     mod_group = f.create_group(mod)
#                     matched_channels = []
#                     matched_data = []

#                     for ch_name in raw.ch_names:
#                         ch_lower = ch_name.lower().replace(' ', '_').replace('-', '_')
#                         for target in channel_list:
#                             target_lower = target.lower().replace('-', '_')
#                             if target_lower in ch_lower or ch_lower in target_lower:
#                                 # Load only this ONE channel from disk
#                                 data = raw.get_data(picks=ch_name)[0]
#                                 matched_channels.append(ch_name)
#                                 matched_data.append(data)
#                                 break

#                     if len(matched_data) > 0:
#                         data_array = np.stack(matched_data, axis=0)
#                         mod_group.create_dataset('data', data=data_array)
#                         mod_group.create_dataset('channels', data=np.array(matched_channels, dtype='S'))
#                         mod_group.attrs['fs'] = 128.0
                        
#                         # Free immediately
#                         del matched_data, data_array

#             return hdf5_path

#         except Exception as e:
#             if self.verbose:
#                 print(f"EDF to HDF5 conversion failed: {e}")
#             return None

#     def _generate_embeddings(self, hdf5_path):
#         """Generate embeddings by streaming HDF5 chunks."""
#         if self.model is None:
#             return None

#         try:
#             import h5py

#             with h5py.File(hdf5_path, 'r') as f:
#                 all_embeddings = {}

#                 for mod in self.channel_groups.keys():
#                     if mod not in f:
#                         all_embeddings[mod] = None
#                         continue

#                     mod_group = f[mod]
#                     if 'data' not in mod_group:
#                         all_embeddings[mod] = None
#                         continue

#                     # Stream from h5py - don't load entire dataset
#                     dataset = mod_group['data']
#                     C, T = dataset.shape

#                     patch_size = 640
#                     n_patches = T // patch_size
#                     if n_patches == 0:
#                         all_embeddings[mod] = None
#                         continue

#                     max_patches_per_chunk = 128
#                     chunk_embeddings = []

#                     for chunk_start in range(0, n_patches, max_patches_per_chunk):
#                         chunk_end = min(chunk_start + max_patches_per_chunk, n_patches)
#                         chunk_samples_start = chunk_start * patch_size
#                         chunk_samples_end = chunk_end * patch_size

#                         # Stream only this chunk
#                         chunk_data = dataset[:, chunk_samples_start:chunk_samples_end]
#                         x = torch.from_numpy(chunk_data).float().to(self.device)
#                         x = x.unsqueeze(0)
#                         mask = torch.ones(1, C, dtype=torch.bool).to(self.device)

#                         with torch.no_grad():
#                             pooled, tokens = self.model(x, mask)

#                         chunk_embeddings.append(tokens.cpu().numpy())
#                         del x, mask, tokens, pooled, chunk_data

#                     if chunk_embeddings:
#                         all_embeddings[mod] = np.concatenate(chunk_embeddings, axis=1)
#                     else:
#                         all_embeddings[mod] = None

#                 return all_embeddings

#         except Exception as e:
#             if self.verbose:
#                 print(f"Embedding generation failed: {e}")
#             return None

#     def extract_features(self, edf_path):
#         """Main entry point: EDF -> features."""
#         if self.model is None:
#             return None

#         with tempfile.TemporaryDirectory() as temp_dir:
#             hdf5_path = self._edf_to_hdf5(edf_path, temp_dir)
#             if hdf5_path is None:
#                 return None

#             embeddings = self._generate_embeddings(hdf5_path)
#             if embeddings is None:
#                 return None

#             features = {}
#             for mod in self.channel_groups.keys():
#                 emb = embeddings.get(mod)
#                 if emb is not None and emb.size > 0:
#                     emb_mod = emb[0]

#                     features[f'sleepfm_{mod}_mean'] = float(np.mean(emb_mod))
#                     features[f'sleepfm_{mod}_std'] = float(np.std(emb_mod))
#                     features[f'sleepfm_{mod}_max'] = float(np.max(emb_mod))
#                     features[f'sleepfm_{mod}_min'] = float(np.min(emb_mod))
#                     features[f'sleepfm_{mod}_median'] = float(np.median(emb_mod))

#                     n_dims = min(8, emb_mod.shape[1])
#                     for i in range(n_dims):
#                         features[f'sleepfm_{mod}_d{i}_mean'] = float(np.mean(emb_mod[:, i]))
#                         features[f'sleepfm_{mod}_d{i}_std'] = float(np.std(emb_mod[:, i]))
#                         features[f'sleepfm_{mod}_d{i}_max'] = float(np.max(emb_mod[:, i]))
#                         features[f'sleepfm_{mod}_d{i}_min'] = float(np.min(emb_mod[:, i]))

#                     if emb_mod.shape[0] > 1:
#                         features[f'sleepfm_{mod}_temporal_std'] = float(np.std(np.mean(emb_mod, axis=1)))
#                         features[f'sleepfm_{mod}_temporal_range'] = float(
#                             np.max(np.mean(emb_mod, axis=1)) - np.min(np.mean(emb_mod, axis=1))
#                         )
#                     else:
#                         features[f'sleepfm_{mod}_temporal_std'] = 0.0
#                         features[f'sleepfm_{mod}_temporal_range'] = 0.0
#                 else:
#                     features[f'sleepfm_{mod}_mean'] = np.nan
#                     features[f'sleepfm_{mod}_std'] = np.nan
#                     features[f'sleepfm_{mod}_max'] = np.nan
#                     features[f'sleepfm_{mod}_min'] = np.nan
#                     features[f'sleepfm_{mod}_median'] = np.nan
#                     for i in range(8):
#                         features[f'sleepfm_{mod}_d{i}_mean'] = np.nan
#                         features[f'sleepfm_{mod}_d{i}_std'] = np.nan
#                         features[f'sleepfm_{mod}_d{i}_max'] = np.nan
#                         features[f'sleepfm_{mod}_d{i}_min'] = np.nan
#                     features[f'sleepfm_{mod}_temporal_std'] = np.nan
#                     features[f'sleepfm_{mod}_temporal_range'] = np.nan

#             return features


# def train_model(data_folder, model_folder, verbose):
#     if verbose:
#         print('Finding Challenge data...')

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
#     return joblib.load(os.path.join(model_folder, 'model.sav'))


# def run_model(model_artifact, record, data_folder, verbose):
#     patient_id = record[HEADERS['bids_folder']]
#     site_id = record[HEADERS['site_id']]
#     session_id = record[HEADERS['session_id']]

#     patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
#     patient_data = load_demographics(patient_data_file, patient_id, session_id)

#     feats = extract_all_features(data_folder, patient_id, site_id, session_id, patient_data)
#     if feats is None:
#         return float('nan'), float('nan')

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

#     for f1, f2, name in model_artifact.get('interactions', []):
#         if f1 in df.columns and f2 in df.columns:
#             df[name] = df[f1] * df[f2]

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


# def extract_all_features(data_folder, patient_id, site_id, session_id, patient_data):
#     features = {}

#     features['age'] = load_age(patient_data)
#     sex = load_sex(patient_data, standardize=True)
#     features['sex_male'] = 1 if sex == 'Male' else 0
#     race = load_race(patient_data, standardize=True)
#     features['race_white'] = 1 if race == 'White' else 0
#     features['race_black'] = 1 if race == 'Black' else 0
#     features['race_asian'] = 1 if race == 'Asian' else 0
#     features['race_other'] = 1 if race == 'Others' else 0
#     features['bmi'] = load_bmi(patient_data)

#     caisr_path = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
#                               site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
#     features.update(_extract_caisr(caisr_path))

#     physio_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
#                                site_id, f"{patient_id}_ses-{session_id}.edf")
#     features.update(_extract_physio(physio_path))

#     return features


# def _extract_caisr(edf_path):
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

#         if stages is not None and len(stages) > 0:
#             tst = (len(stages) * 30) / 3600
#             out['stage_caisr_tst'] = tst
#             out['stage_caisr_se'] = float(np.sum(np.isin(stages, [1,2,3,4])) / len(stages))
#             out['stage_transition_rate'] = float(np.sum(np.diff(stages) != 0) / max(tst, 0.5))

#         dh = max(out['stage_caisr_tst'], 0.5)

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

#         if pn3 is not None: out['caisr_prob_n3_mean'] = float(np.mean(pn3))
#         if pn2 is not None: out['caisr_prob_n2_mean'] = float(np.mean(pn2))
#         if pw  is not None: out['caisr_prob_w_mean']  = float(np.mean(pw))
#         if pr  is not None: out['caisr_prob_r_mean']  = float(np.mean(pr))

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
#             if eeg is not None and eeg_fs > 0:
#                 sl = eeg[int(st*eeg_fs):int(en*eeg_fs)]
#                 if len(sl) > eeg_fs * 10:
#                     sl = (sl - np.nanmean(sl)) / (np.nanstd(sl) + 1e-8)
#                     ef = _eeg_spectrum(sl, eeg_fs)
#                     for idx, m in enumerate(EEG_METRICS):
#                         out[f'physio_{stage}_eeg_{m}'] = ef[idx]

#             if emg is not None and emg_fs > 0:
#                 sl = emg[int(st*emg_fs):int(en*emg_fs)]
#                 if len(sl) > emg_fs * 10:
#                     out[f'physio_{stage}_emg_rms'] = float(
#                         np.sqrt(np.mean(np.square(sl - np.mean(sl)))))

#             if ecg is not None and ecg_fs > 0:
#                 sl = ecg[int(st*ecg_fs):int(en*ecg_fs)]
#                 if len(sl) > ecg_fs * 10:
#                     out[f'physio_{stage}_ecg_hrv'] = float(np.var(np.diff(sl)))

#             if rsp is not None and rsp_fs > 0:
#                 sl = rsp[int(st*rsp_fs):int(en*rsp_fs)]
#                 if len(sl) > rsp_fs * 10:
#                     out[f'physio_{stage}_resp_freq'] = _resp_spectrum(sl, rsp_fs)[0]

#             if eff is not None and eff_fs > 0:
#                 sl = eff[int(st*eff_fs):int(en*eff_fs)]
#                 if len(sl) > eff_fs * 10:
#                     out[f'physio_{stage}_resp_effort'] = float(np.var(sl))

#             if sp2 is not None and sp2_fs > 0:
#                 sl = sp2[int(st*sp2_fs):int(en*sp2_fs)]
#                 if len(sl) > sp2_fs * 10:
#                     out[f'physio_{stage}_spo2_drop'] = float(
#                         np.percentile(sl, 95) - np.percentile(sl, 5))

#     except Exception:
#         pass
#     return out


# def _find_sig(labels, data_dict, fs_dict, target):
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
#     if sig is None or len(sig) == 0:
#         return sig
#     mn, mx = np.min(sig), np.max(sig)
#     if mx > 1.0001 or mn < -0.0001:
#         d = mx - mn
#         if d > 1e-6:
#             sig = (sig - mn) / d
#     return np.clip(sig, 0.0, 1.0)


# def _count_events(arr, codes):
#     if arr is None or len(arr) == 0:
#         return 0
#     b = np.isin(arr, codes).astype(int)
#     d = np.diff(b)
#     return max(np.sum(d == 1) + (1 if b[0] == 1 else 0), 0)


# def _eeg_spectrum(signal, fs):
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


