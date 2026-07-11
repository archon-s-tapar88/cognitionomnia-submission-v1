#!/usr/bin/env python
"""
PhysioNet Challenge 2026 - Age-Conditioned AUROC Optimized v4
Wins by: age-residualized features + stage-conditional spectral + ensemble weight optimization
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
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from scipy.optimize import minimize

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

warnings.filterwarnings("ignore")
from helper_code import *

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
FINAL_MODEL = "auto"
WINDOWS = ['early', 'mid', 'late']
EEG_METRICS = ['delta','theta','alpha','sigma','beta','alpha_theta',
               'theta_beta','slowing','delta_sigma','entropy','sef50','sef90',
               'hjorth_activity','hjorth_mobility','hjorth_complexity']

# -------------------------------------------------------------------------
# REQUIRED FUNCTIONS
# -------------------------------------------------------------------------
def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding Challenge data...')
    
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)
    
    if num_records == 0:
        raise FileNotFoundError('No data provided.')
    
    if verbose:
        print(f'Found {num_records} records. Extracting features...')
    
    # PHASE 1: Extract features
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
    
    # PHASE 2: Temporal-difference features (late vs early)
    for m in EEG_METRICS + ['emg_rms','ecg_hrv','ecg_mean_hr','resp_freq',
                            'resp_effort','spo2_drop','spo2_mean','spo2_min']:
        e = f'physio_early_{m}'
        l = f'physio_late_{m}'
        if e in df.columns and l in df.columns:
            df[f'physio_delta_{m}'] = df[l] - df[e]
            feature_names.append(f'physio_delta_{m}')
    
    # PHASE 3: Poison filtering (directional site instability)
    if verbose:
        print('Filtering site-poisonous features...')
    
    sites = df['site'].unique()
    poison_features = []
    
    for col in feature_names:
        if col.startswith('inter_') or col.startswith('physio_delta_'):
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
    
    # PHASE 4: Interaction features
    interactions = [
        ('resp_caisr_ahi', 'physio_late_spo2_drop', 'inter_AHI_SpO2'),
        ('caisr_prob_w_mean', 'physio_early_eeg_delta', 'inter_WASO_SWA'),
        ('caisr_prob_r_mean', 'physio_late_emg_rms', 'inter_REM_EMG_Atonia'),
        ('physio_late_ecg_hrv', 'physio_late_resp_effort', 'inter_HRV_RespEffort'),
        ('physio_late_eeg_theta_beta', 'physio_late_resp_freq', 'inter_ThetaBeta_RespFreq'),
        ('physio_mid_eeg_delta_sigma', 'physio_mid_spo2_drop', 'inter_mid_DeltaSigma_SpO2'),
        ('physio_mid_eeg_theta_beta', 'physio_mid_resp_effort', 'inter_mid_ThetaBeta_Effort'),
    ]
    
    age_interactions = []
    if 'age' in kept:
        age_interactions.append(('age', 'age', 'age_sq'))
        for partner in ['resp_caisr_ahi', 'caisr_n3_pct', 'caisr_prob_n3_mean',
                        'physio_early_eeg_delta', 'physio_late_eeg_entropy', 'bmi']:
            if partner in kept:
                age_interactions.append(('age', partner, f'inter_age_{partner}'))
    
    for f1, f2, name in interactions + age_interactions:
        if f1 in kept and f2 in kept:
            df[name] = df[f1] * df[f2] if f1 != f2 else df[f1] ** 2
            if name not in kept:
                kept.append(name)
    
    # PHASE 5: Age residualization (skip demographics & age interactions)
    if verbose:
        print('Age-residualizing physiology features...')
    
    demo_cols = {'age', 'sex_male', 'sex_female', 'race_white', 'race_black',
                 'race_asian', 'race_other', 'race_unavailable'}
    resid_cols = []
    age_resid_models = {}
    
    for col in kept:
        if col in demo_cols or col.startswith('inter_age_') or col == 'age_sq':
            continue
        sub = df.dropna(subset=[col, 'age'])
        if len(sub) > 10:
            lr = LinearRegression()
            lr.fit(sub[['age']].values, sub[col].values)
            df[f"{col}_resid"] = df[col] - lr.predict(df[['age']].values)
            resid_cols.append(f"{col}_resid")
            age_resid_models[col] = lr
    
    # PHASE 6: Impute & scale
    final_cols = list(demo_cols.intersection(kept)) + [c for c in kept if c.startswith('inter_')] + ['age_sq'] + resid_cols
    # Ensure all exist
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan
        med = df[col].median()
        if np.isnan(med):
            med = 0.0
        df[col] = df[col].fillna(med)
    
    X = df[final_cols].values
    y = df['label'].values
    ages = df['age'].values
    sites_arr = df['site'].values
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xs = scaler.fit_transform(imputer.fit_transform(X))
    
    # PHASE 7: Train candidate models
    if verbose:
        print('Training candidate models...')
    
    candidates = {}
    
    candidates['logistic'] = LogisticRegression(
        C=0.5, penalty='l2', solver='liblinear',
        class_weight='balanced', max_iter=1000, random_state=42)
    candidates['logistic'].fit(Xs, y)
    
    candidates['logistic_l1'] = LogisticRegression(
        C=0.1, penalty='l1', solver='liblinear',
        class_weight='balanced', max_iter=1000, random_state=42)
    candidates['logistic_l1'].fit(Xs, y)
    
    if lgb is not None:
        candidates['lgb_shallow'] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, reg_lambda=1,
            random_state=42, n_jobs=1, verbose=-1,
            class_weight='balanced')
        candidates['lgb_shallow'].fit(Xs, y)
        
        candidates['lgb_medium'] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, reg_lambda=1,
            random_state=42, n_jobs=1, verbose=-1,
            class_weight='balanced')
        candidates['lgb_medium'].fit(Xs, y)
    
    candidates['svm_lin'] = SVC(
        probability=True, C=0.5, kernel='linear',
        class_weight='balanced', random_state=42)
    candidates['svm_lin'].fit(Xs, y)
    
    candidates['stack'] = StackingClassifier(
        estimators=[
            ('lr1', LogisticRegression(C=0.05, class_weight='balanced', solver='liblinear', max_iter=1000)),
            ('svm', SVC(probability=True, C=0.1, kernel='linear', class_weight='balanced'))
        ],
        final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
        n_jobs=1, passthrough=False)
    candidates['stack'].fit(Xs, y)
    
    # PHASE 8: LOSO CV + ensemble weight optimization on age-AUROC
    names = list(candidates.keys())
    oof_probs = {name: np.full(len(y), np.nan) for name in names}
    
    if verbose:
        print('Running site-aware LOSO CV...')
    
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
        test_idx = np.where(test_mask)[0]
        
        for name in names:
            m = _make_fresh_model(name)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_val)[:, 1]
            oof_probs[name][test_idx] = p
    
    # Report individual scores
    if verbose:
        for name in names:
            valid = ~np.isnan(oof_probs[name])
            if np.sum(valid) > 0:
                sc = _age_auroc(y[valid], oof_probs[name][valid], ages[valid], delta=2.0)
                print(f"  {name:15s}: LOSO age-AUROC = {sc:.4f}")
    
    # Optimize ensemble weights to maximize age-AUROC
    def _ens_score(weights):
        w = np.maximum(weights, 0)
        if np.sum(w) == 0:
            return 0.0
        w = w / np.sum(w)
        ens = np.zeros(len(y))
        for i, name in enumerate(names):
            valid = ~np.isnan(oof_probs[name])
            ens[valid] += w[i] * oof_probs[name][valid]
        valid = ~np.isnan(ens)
        if np.sum(valid) == 0:
            return 0.0
        return _age_auroc(y[valid], ens[valid], ages[valid], delta=2.0)
    
    # Replace the SLSQP block with this:
    scores = []
    for name in names:
        valid = ~np.isnan(oof_probs[name])
        sc = _age_auroc(y[valid], oof_probs[name][valid], ages[valid], delta=2.0)
        scores.append(sc)

    valid_mask = np.array(scores) > 0.50
    scores = np.array(scores)
    scores[~valid_mask] = 0.0

    # Softmax weighting
    exp_scores = np.exp(scores - np.max(scores))
    best_w = exp_scores / np.sum(exp_scores)
    ensemble_weights = dict(zip(names, best_w))
    
    if verbose:
        print(f"Optimized ensemble LOSO age-AUROC = {_ens_score(best_w):.4f}")
        for name, w in zip(names, best_w):
            print(f"  {name:15s}: weight = {w:.3f}")
    
    # PHASE 9: Save artifact
    os.makedirs(model_folder, exist_ok=True)
    
    artifact = {
        'models': candidates,
        'ensemble_weights': ensemble_weights,
        'scaler': scaler,
        'imputer': imputer,
        'final_cols': final_cols,
        'resid_cols': resid_cols,
        'age_resid_models': age_resid_models,
        'interactions': interactions + age_interactions,
        'sites_seen': list(sites),
        'median_age': float(np.median(ages)),
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
    if np.isnan(age):
        age = model_artifact.get('median_age', 65.0)
    df['age'] = age
    
    # Build interactions
    for f1, f2, name in model_artifact.get('interactions', []):
        if f1 in df.columns and f2 in df.columns:
            df[name] = df[f1] * df[f2] if f1 != f2 else df[f1] ** 2
    
    # Build temporal deltas
    for m in EEG_METRICS + ['emg_rms','ecg_hrv','ecg_mean_hr','resp_freq',
                            'resp_effort','spo2_drop','spo2_mean','spo2_min']:
        e = f'physio_early_{m}'
        l = f'physio_late_{m}'
        if e in df.columns and l in df.columns:
            df[f'physio_delta_{m}'] = df[l] - df[e]
    
    # Age residualization
    for col, lr in model_artifact.get('age_resid_models', {}).items():
        if col in df.columns:
            df[f"{col}_resid"] = df[col].values - lr.predict(np.array([[age]]))[0]
    
    final_cols = model_artifact['final_cols']
    for col in final_cols:
        if col not in df.columns:
            df[col] = float('nan')
        med = df[col].median()
        if np.isnan(med):
            med = 0.0
        df[col] = df[col].fillna(med)
    
    X = df[final_cols].values
    Xs = model_artifact['scaler'].transform(model_artifact['imputer'].transform(X))
    
    # Ensemble prediction
    weights = model_artifact['ensemble_weights']
    prob = 0.0
    for name, w in weights.items():
        if w > 0 and name in model_artifact['models']:
            prob += w * model_artifact['models'][name].predict_proba(Xs)[0, 1]
    
    # Threshold irrelevant for AUROC ranking; use 0.5 for API compliance
    binary = int(prob >= 0.5)
    return binary, float(prob)


# -------------------------------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------------------------------
def extract_all_features(data_folder, patient_id, site_id, session_id, patient_data):
    features = {}
    
    # 1. Demographics
    features['age'] = load_age(patient_data)
    sex = load_sex(patient_data, standardize=True)
    features['sex_male'] = 1 if sex == 'Male' else 0
    features['sex_female'] = 1 if sex == 'Female' else 0
    race = load_race(patient_data, standardize=True)
    features['race_white'] = 1 if race == 'White' else 0
    features['race_black'] = 1 if race == 'Black' else 0
    features['race_asian'] = 1 if race == 'Asian' else 0
    features['race_other'] = 1 if race == 'Others' else 0
    features['race_unavailable'] = 1 if race == 'Unavailable' else 0
    features['bmi'] = load_bmi(patient_data)
    
    # 2. CAISR
    caisr_path = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                              site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    caisr_features, caisr_stages = _extract_caisr(caisr_path)
    features.update(caisr_features)
    
    # 3. Physiology
    physio_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                               site_id, f"{patient_id}_ses-{session_id}.edf")
    features.update(_extract_physio(physio_path, stages=caisr_stages))
    
    return features


def _extract_caisr(edf_path):
    """Return (features_dict, stages_array_or_None)."""
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
        'caisr_n3_pct': np.nan, 'caisr_n2_pct': np.nan,
        'caisr_n1_pct': np.nan, 'caisr_rem_pct': np.nan,
        'caisr_wake_pct': np.nan, 'caisr_sleep_efficiency': np.nan,
        'caisr_sleep_latency_min': np.nan, 'caisr_rem_latency_min': np.nan,
        'caisr_waso_min': np.nan, 'caisr_rdi': np.nan,
        'caisr_stage_prob_var': np.nan, 'caisr_n1_sleep_pct': np.nan,
        'caisr_plm_index': np.nan,
    }
    stages_out = None
    
    if not os.path.exists(edf_path):
        return out, stages_out
    
    try:
        data_dict, fs_dict = load_signal_data(edf_path)
        if not data_dict:
            return out, stages_out
        
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
        
        if stages is not None and len(stages) > 0:
            stages_out = stages
            epoch_dur_min = 0.5
            n_epochs = len(stages)
            tst = (n_epochs * 30) / 3600
            out['stage_caisr_tst'] = tst
            
            valid = stages[stages < 9]
            if len(valid) > 0:
                out['caisr_wake_pct'] = float(np.mean(valid == 5))
                out['caisr_n1_pct'] = float(np.mean(valid == 3))
                out['caisr_n2_pct'] = float(np.mean(valid == 2))
                out['caisr_n3_pct'] = float(np.mean(valid == 1))
                out['caisr_rem_pct'] = float(np.mean(valid == 4))
                out['stage_caisr_se'] = float(np.mean((valid >= 1) & (valid <= 4)))
                out['caisr_sleep_efficiency'] = out['stage_caisr_se']
                
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
            
            out['stage_transition_rate'] = float(np.sum(np.diff(stages) != 0) / max(tst, 0.5))
            
            stage_probs = []
            for p in [pn3, pn2, pn1, pr, pw]:
                if p is not None and len(p) == len(stages):
                    stage_probs.append(p)
            if len(stage_probs) > 0:
                sp_arr = np.array(stage_probs)
                out['caisr_stage_prob_var'] = float(np.mean(np.var(sp_arr, axis=0)))
        
        dh = max(out['stage_caisr_tst'], 0.5)
        
        if arousal is not None:
            out['arousal_caisr_rate'] = float(_count_events(arousal, [1]) / dh)
        if resp is not None:
            out['resp_caisr_ahi'] = float(_count_events(resp, [1, 2, 3, 4]) / dh)
            tap = _count_events(resp, [2]) + _count_events(resp, [1])
            out['resp_central_ratio'] = float(_count_events(resp, [2]) / tap) if tap > 0 else 0.0
            out['caisr_rdi'] = out['resp_caisr_ahi'] + out.get('arousal_caisr_rate', 0)
        if limbs is not None:
            out['limb_caisr_rate'] = float(_count_events(limbs, [1, 2]) / dh)
            out['limb_isolated_rate'] = float(_count_events(limbs, [1]) / dh)
            out['limb_periodic_rate'] = float(_count_events(limbs, [2]) / dh)
            out['caisr_plm_index'] = out['limb_periodic_rate']
        
        if pn3 is not None: out['caisr_prob_n3_mean'] = float(np.mean(pn3))
        if pn2 is not None: out['caisr_prob_n2_mean'] = float(np.mean(pn2))
        if pw  is not None: out['caisr_prob_w_mean']  = float(np.mean(pw))
        if pr  is not None: out['caisr_prob_r_mean']  = float(np.mean(pr))
        
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
    return out, stages_out


def _extract_physio(edf_path, stages=None):
    """Extract physio features. If stages provided, compute stage-conditional features."""
    out = {}
    for w in WINDOWS:
        for m in EEG_METRICS:
            out[f'physio_{w}_eeg_{m}'] = np.nan
        out[f'physio_{w}_emg_rms'] = np.nan
        out[f'physio_{w}_ecg_hrv'] = np.nan
        out[f'physio_{w}_ecg_mean_hr'] = np.nan
        out[f'physio_{w}_resp_freq'] = np.nan
        out[f'physio_{w}_resp_effort'] = np.nan
        out[f'physio_{w}_spo2_drop'] = np.nan
        out[f'physio_{w}_spo2_mean'] = np.nan
        out[f'physio_{w}_spo2_min'] = np.nan
    
    # Stage-conditional features
    out['n3_eeg_delta'] = np.nan
    out['n3_eeg_theta'] = np.nan
    out['n3_eeg_sigma'] = np.nan
    out['n3_eeg_entropy'] = np.nan
    out['n3_eeg_sef90'] = np.nan
    out['rem_emg_rms'] = np.nan
    out['sleep_spo2_mean'] = np.nan
    out['sleep_spo2_min'] = np.nan
    out['sleep_spo2_drop'] = np.nan
    
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
        
        dur = 0
        for s, f in [(eeg, eeg_fs), (emg, emg_fs), (ecg, ecg_fs),
                     (rsp, rsp_fs), (eff, eff_fs), (sp2, sp2_fs)]:
            if s is not None and f > 0:
                dur = max(dur, len(s) / f)
        if dur <= 0:
            return out
        
        t3 = dur / 3.0
        bounds = {'early': (0, t3), 'mid': (t3, 2*t3), 'late': (2*t3, dur)}
        
        for stage, (st, en) in bounds.items():
            if eeg is not None and eeg_fs > 0:
                sl = eeg[int(st*eeg_fs):int(en*eeg_fs)]
                if len(sl) > eeg_fs * 30:
                    sl = (sl - np.nanmean(sl)) / (np.nanstd(sl) + 1e-8)
                    ef = _eeg_features_welch(sl, eeg_fs)
                    for idx, m in enumerate(EEG_METRICS):
                        out[f'physio_{stage}_eeg_{m}'] = ef[idx]
            
            if emg is not None and emg_fs > 0:
                sl = emg[int(st*emg_fs):int(en*emg_fs)]
                if len(sl) > emg_fs * 10:
                    out[f'physio_{stage}_emg_rms'] = float(
                        np.sqrt(np.mean(np.square(sl - np.mean(sl)))))
            
            if ecg is not None and ecg_fs > 0:
                sl = ecg[int(st*ecg_fs):int(en*ecg_fs)]
                if len(sl) > ecg_fs * 10:
                    out[f'physio_{stage}_ecg_hrv'] = float(np.var(np.diff(sl)))
                    # Simple mean HR estimate
                    peaks, _ = scipy.signal.find_peaks(sl, distance=int(0.5*ecg_fs),
                                                        prominence=np.std(sl)*0.3)
                    if len(peaks) >= 5:
                        rr = np.diff(peaks) / ecg_fs
                        rr = rr[(rr > 0.4) & (rr < 1.5)]
                        if len(rr) > 0:
                            out[f'physio_{stage}_ecg_mean_hr'] = float(60.0 / np.mean(rr))
            
            if rsp is not None and rsp_fs > 0:
                sl = rsp[int(st*rsp_fs):int(en*rsp_fs)]
                if len(sl) > rsp_fs * 10:
                    out[f'physio_{stage}_resp_freq'] = _resp_spectrum(sl, rsp_fs)[0]
            
            if eff is not None and eff_fs > 0:
                sl = eff[int(st*eff_fs):int(en*eff_fs)]
                if len(sl) > eff_fs * 10:
                    out[f'physio_{stage}_resp_effort'] = float(np.var(sl))
            
            if sp2 is not None and sp2_fs > 0:
                sl = sp2[int(st*sp2_fs):int(en*sp2_fs)]
                if len(sl) > sp2_fs * 10:
                    out[f'physio_{stage}_spo2_drop'] = float(
                        np.percentile(sl, 95) - np.percentile(sl, 5))
                    out[f'physio_{stage}_spo2_mean'] = float(np.mean(sl))
                    out[f'physio_{stage}_spo2_min'] = float(np.min(sl))
        
        # Stage-conditional features (the secret weapon)
        if stages is not None:
            # N3 EEG
            n3_eeg = _extract_stage_signal(eeg, eeg_fs, stages, target_stage=1)
            if len(n3_eeg) > eeg_fs * 30:
                n3_eeg = (n3_eeg - np.nanmean(n3_eeg)) / (np.nanstd(n3_eeg) + 1e-8)
                ef = _eeg_features_welch(n3_eeg, eeg_fs)
                out['n3_eeg_delta'] = ef[0]
                out['n3_eeg_theta'] = ef[1]
                out['n3_eeg_sigma'] = ef[3]
                out['n3_eeg_entropy'] = ef[9]
                out['n3_eeg_sef90'] = ef[11]
            
            # REM EMG
            rem_emg = _extract_stage_signal(emg, emg_fs, stages, target_stage=4)
            if len(rem_emg) > emg_fs * 30:
                out['rem_emg_rms'] = float(
                    np.sqrt(np.mean(np.square(rem_emg - np.mean(rem_emg)))))
            
            # Sleep SpO2 (stages 1-4)
            sleep_sp2 = _extract_stage_signal(sp2, sp2_fs, stages, target_stages=[1,2,3,4])
            if len(sleep_sp2) > sp2_fs * 30:
                out['sleep_spo2_mean'] = float(np.mean(sleep_sp2))
                out['sleep_spo2_min'] = float(np.min(sleep_sp2))
                out['sleep_spo2_drop'] = float(
                    np.percentile(sleep_sp2, 95) - np.percentile(sleep_sp2, 5))
    
    except Exception:
        pass
    return out


def _extract_stage_signal(signal, fs, stages, target_stage=None, target_stages=None):
    """Extract concatenated signal segments for specified sleep stage(s)."""
    if signal is None or stages is None or len(stages) == 0 or fs <= 0:
        return np.array([])
    if target_stage is not None:
        target_stages = [target_stage]
    if target_stages is None:
        return np.array([])
    
    epoch_samples = int(30 * fs)
    segments = []
    for i, s in enumerate(stages):
        if s in target_stages:
            start = i * epoch_samples
            end = start + epoch_samples
            if end <= len(signal):
                segments.append(signal[start:end])
    if len(segments) == 0:
        return np.array([])
    return np.concatenate(segments)


# -------------------------------------------------------------------------
# SPECTRAL & UTILITIES
# -------------------------------------------------------------------------
def _eeg_features_welch(signal, fs):
    if signal is None or len(signal) == 0:
        return [np.nan] * len(EEG_METRICS)
    try:
        nperseg = min(30 * int(fs), len(signal))
        if nperseg < 2 * int(fs):
            return [np.nan] * len(EEG_METRICS)
        freqs, psd = scipy.signal.welch(signal, fs, nperseg=nperseg,
                                         window='hann', noverlap=nperseg//2)
        
        ti = (freqs >= 0.5) & (freqs <= 30)
        tp = np.sum(psd[ti])
        if tp == 0 or np.isnan(tp):
            return [np.nan] * len(EEG_METRICS)
        
        delta = np.sum(psd[(freqs >= 0.5) & (freqs < 4)]) / tp
        theta = np.sum(psd[(freqs >= 4) & (freqs < 8)]) / tp
        alpha = np.sum(psd[(freqs >= 8) & (freqs < 12)]) / tp
        sigma = np.sum(psd[(freqs >= 12) & (freqs < 15)]) / tp
        beta  = np.sum(psd[(freqs >= 15) & (freqs <= 30)]) / tp
        
        at = alpha / (theta + 1e-8)
        tb = theta / (beta + 1e-8)
        sl = (delta + theta) / (alpha + beta + 1e-8)
        ds = delta / (sigma + 1e-8)
        
        pn = psd[ti] / tp
        ent = scipy.stats.entropy(pn, base=2)
        
        cp = np.cumsum(psd[ti])
        sef50 = freqs[ti][np.where(cp >= 0.50 * tp)[0][0]] if np.any(cp >= 0.5*tp) else np.nan
        sef90 = freqs[ti][np.where(cp >= 0.90 * tp)[0][0]] if np.any(cp >= 0.9*tp) else np.nan
        
        activity = np.var(signal)
        d1 = np.diff(signal)
        d2 = np.diff(d1)
        mobility = np.sqrt(np.var(d1) / (activity + 1e-8))
        complexity = np.sqrt(np.var(d2) / (np.var(d1) + 1e-8)) / (mobility + 1e-8)
        
        return [delta, theta, alpha, sigma, beta, at, tb, sl, ds, ent, sef50, sef90,
                activity, mobility, complexity]
    except Exception:
        return [np.nan] * len(EEG_METRICS)


def _resp_spectrum(signal, fs):
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


def _find_sig(labels, data_dict, fs_dict, target):
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
        return LogisticRegression(C=0.5, penalty='l2', solver='liblinear',
                                   class_weight='balanced', max_iter=1000, random_state=42)
    elif name == 'logistic_l1':
        return LogisticRegression(C=0.1, penalty='l1', solver='liblinear',
                                   class_weight='balanced', max_iter=1000, random_state=42)
    elif name == 'lgb_shallow' and lgb is not None:
        return lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, num_leaves=15,
                                   min_child_samples=10, reg_lambda=1,
                                   random_state=42, n_jobs=1, verbose=-1,
                                   class_weight='balanced')
    elif name == 'lgb_medium' and lgb is not None:
        return lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=10, reg_lambda=1,
                                   random_state=42, n_jobs=1, verbose=-1,
                                   class_weight='balanced')
    elif name == 'svm_lin':
        return SVC(probability=True, C=0.5, kernel='linear',
                   class_weight='balanced', random_state=42)
    elif name == 'stack':
        return StackingClassifier(
            estimators=[
                ('lr1', LogisticRegression(C=0.05, class_weight='balanced', solver='liblinear', max_iter=1000)),
                ('svm', SVC(probability=True, C=0.1, kernel='linear', class_weight='balanced'))
            ],
            final_estimator=LogisticRegression(C=0.1, solver='liblinear', max_iter=1000),
            n_jobs=1, passthrough=False
        )
    else:
        raise ValueError(f"Unknown model name: {name}")
    

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
# # #!/usr/bin/env python

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


