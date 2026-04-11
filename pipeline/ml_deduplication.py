"""
ML-Based Incident Deduplication
================================
Transforms incident deduplication from static rules into a learning system.

    f(Report A, Report B) → P(same incident)  in [0, 1]

Architecture
------------
1. Input representation  — structured text from report fields
2. LaBSE embeddings      — reuses pipeline/embedder.py model (optionally fine-tuned)
3. Pairwise features     — semantic + geo + temporal + metadata signals
4. Classifier            — XGBoost (preferred) / GradientBoosting / LR fallback
5. Fine-tuning (opt.)    — ContrastiveLoss on LaBSE for cross-lingual alignment

Drop-in integration
-------------------
  ml_similarity(report, uir) → same dict as combined_similarity()
  Swap into clustering.py via set_similarity_model() — DBSCAN unchanged.

Why ML genuinely wins here (vs fixed weights)
---------------------------------------------
  • Learns sem×geo interaction: high semantic + mismatched geography is a
    much weaker merge signal than either alone — impossible with fixed weights
  • Cross-lingual pairs: after fine-tuning, Sinhala ↔ Tamil reports of the
    same flood share high cosine similarity; the classifier exploits the
    cross_lingual feature to boost confidence
  • Hard negatives teach the model that same-location / same-type ≠ same
    incident (adjacent_critical scenario)
  • Operator corrections flow directly into retraining via FeedbackLogger
"""

import json
import math
import pickle
import random
import time
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Optional

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, roc_auc_score,
                                 precision_recall_fscore_support)
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False

# ── geo / temporal helpers (same as clustering.py, no circular import) ────────

def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _geo_sim(r_a: dict, r_b: dict) -> float:
    la, loa = r_a.get('lat'), r_a.get('lng')
    lb, lob = r_b.get('lat'), r_b.get('lng')
    if la is None or lb is None:
        return 0.0
    return 1.0 - min(_haversine_km(la, loa, lb, lob) / 5.0, 1.0)

def _time_sim(r_a: dict, r_b: dict) -> float:
    ta = r_a.get('timestamp', r_a.get('receive_time'))
    tb = r_b.get('timestamp', r_b.get('receive_time',
         r_b.get('last_updated', r_b.get('created_at'))))
    if ta is None or tb is None:
        return 0.5
    return math.exp(-0.05 * abs((ta - tb).total_seconds()) / 60)

_URGENCY = {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}


def _location_str_sim(r_a: dict, r_b: dict) -> float:
    """
    Fuzzy string similarity between location_raw fields.
    Uses token_sort_ratio (handles word reordering).
    Key signal for same_street_neighbors: "No 12, Kelaniya Road" vs
    "No 54, Kelaniya Road" → ~0.80, not 1.0, despite identical geocoded coords.
    Returns 0.5 if either location is empty (unknown).
    """
    loc_a = str(r_a.get('location_raw') or '').strip()
    loc_b = str(r_b.get('location_raw') or '').strip()
    if not loc_a or not loc_b:
        return 0.5
    try:
        from rapidfuzz.fuzz import token_sort_ratio
        return token_sort_ratio(loc_a.lower(), loc_b.lower()) / 100.0
    except Exception:
        # character-set overlap fallback
        sa, sb = set(loc_a.lower()), set(loc_b.lower())
        return len(sa & sb) / max(len(sa | sb), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PAIR GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class PairGenerator:
    """
    Generates labeled (report_a, report_b, label) training pairs from ground truth.

    label=1  — same incident  (positive)
    label=0  — different incidents (negative, including hard negatives)

    Hard negatives make the classifier learn difficult boundary cases:
      - Same location, different incidents (adjacent_critical scenario)
      - Same incident_type but unrelated events
      - Close in time but different location
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate(self, reports: list, neg_ratio: float = 3.0) -> list:
        """
        Returns list of (report_a, report_b, label) tuples.
        neg_ratio: number of negative pairs per positive pair.
        """
        reports = [r for r in reports if r.get('embedding') is not None
                   and r.get('_ground_truth', {}).get('incident_id', 'UNKNOWN') != 'UNKNOWN']

        positives = self._positive_pairs(reports)
        n_neg_target = int(len(positives) * neg_ratio)
        hard_neg = self._hard_negatives(reports)
        easy_neg = self._random_negatives(reports, max(0, n_neg_target - len(hard_neg)))
        negatives = hard_neg + easy_neg

        print(f"  [PairGen] Positives: {len(positives)}, "
              f"Hard neg: {len(hard_neg)}, Easy neg: {len(easy_neg)}, "
              f"Total: {len(positives)+len(negatives)}")
        return positives + negatives

    def _positive_pairs(self, reports: list) -> list:
        """All within-incident pairs (same incident_id)."""
        by_incident: dict = {}
        for r in reports:
            iid = r['_ground_truth']['incident_id']
            by_incident.setdefault(iid, []).append(r)

        pairs = []
        for reps in by_incident.values():
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    pairs.append((reps[i], reps[j], 1))
        return pairs

    def _hard_negatives(self, reports: list) -> list:
        """
        Three types of hard negatives that expose the most common false-positive
        patterns:
          A) Same location, different incident
          B) Same incident_type, different incident, similar time
          C) Different location, close in time (< 20 min)
        """
        pairs = []
        by_incident: dict = {}
        for r in reports:
            iid = r['_ground_truth']['incident_id']
            by_incident.setdefault(iid, []).append(r)

        incident_ids = list(by_incident.keys())

        # A) Same location, different incident
        by_loc: dict = {}
        for iid, reps in by_incident.items():
            loc = reps[0].get('_ground_truth', {}).get('actual_location', '')
            by_loc.setdefault(loc, []).append(iid)
        for loc, iids in by_loc.items():
            if len(iids) < 2:
                continue
            for i in range(min(3, len(iids))):
                for j in range(i + 1, min(4, len(iids))):
                    r_a = by_incident[iids[i]][0]
                    r_b = by_incident[iids[j]][0]
                    pairs.append((r_a, r_b, 0))

        # B) Same incident_type, different incident
        by_type: dict = {}
        for r in reports:
            by_type.setdefault(r.get('incident_type', ''), []).append(r)
        for itype, reps in by_type.items():
            different = [(r1, r2) for r1 in reps[:20] for r2 in reps[:20]
                         if r1['_ground_truth']['incident_id'] != r2['_ground_truth']['incident_id']]
            sampled = random.sample(different, min(50, len(different)))
            for r1, r2 in sampled:
                pairs.append((r1, r2, 0))

        # C) Close in time, different location
        timestamped = [(r, r.get('timestamp', r.get('receive_time')))
                       for r in reports if r.get('timestamp') or r.get('receive_time')]
        timestamped.sort(key=lambda x: x[1] or '')
        for i in range(len(timestamped) - 1):
            r1, t1 = timestamped[i]
            r2, t2 = timestamped[i + 1]
            if t1 is None or t2 is None:
                continue
            gap_min = abs((t1 - t2).total_seconds()) / 60
            if (gap_min < 20
                    and r1['_ground_truth']['incident_id'] != r2['_ground_truth']['incident_id']
                    and r1.get('_ground_truth', {}).get('actual_location') !=
                        r2.get('_ground_truth', {}).get('actual_location')):
                pairs.append((r1, r2, 0))
                if len([p for p in pairs if p[2] == 0]) > 300:
                    break

        return pairs

    def _random_negatives(self, reports: list, n: int) -> list:
        """Random cross-incident pairs."""
        by_incident: dict = {}
        for r in reports:
            by_incident.setdefault(r['_ground_truth']['incident_id'], []).append(r)
        incident_ids = list(by_incident.keys())

        pairs = []
        attempts = 0
        while len(pairs) < n and attempts < n * 10:
            attempts += 1
            i1, i2 = random.sample(incident_ids, 2)
            r1 = random.choice(by_incident[i1])
            r2 = random.choice(by_incident[i2])
            pairs.append((r1, r2, 0))
        return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2–3 — ML DEDUPLICATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MLDeduplicator:
    """
    Trainable incident deduplication model.

    Features extracted per pair:
      sem_sim            — cosine similarity of LaBSE embeddings
      geo_sim            — geographic similarity (1 - min(dist_km/5, 1))
      time_sim           — temporal similarity (exp(-0.05 * delta_min))
      type_match         — 1 if same incident_type
      urgency_diff       — |urgency_a - urgency_b| normalized to [0,1]
      cross_lingual      — 1 if reports are in different languages
      both_critical      — 1 if both reports are CRITICAL
      sem_geo            — sem_sim × geo_sim  (interaction term)
      sem_time           — sem_sim × time_sim (interaction term)
      location_str_sim   — fuzzy string similarity of location_raw fields
      geo_str_divergence — geo_sim × (1 - location_str_sim): high when same
                           geocoded coords but different address strings —
                           key signal for same_street_neighbors scenario

    The interaction terms are what make ML genuinely better than fixed weights:
    sem_sim=0.8 + geo_sim=0.9 → very high merge probability
    sem_sim=0.8 + geo_sim=0.1 → much lower despite high semantic similarity
    geo_sim=1.0 + location_str_sim=0.80 → geo_str_divergence=0.20 → likely different houses
    """

    FEATURE_NAMES = [
        'sem_sim', 'geo_sim', 'time_sim', 'type_match',
        'urgency_diff', 'cross_lingual', 'both_critical',
        'sem_geo', 'sem_time',
        'location_str_sim', 'geo_str_divergence',
    ]

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.model_path = self.data_dir / 'ml_deduplicator.pkl'
        self.finetuned_path = self.data_dir / 'finetuned_labse'

        self._clf = None
        self._finetuned_embedder = None
        self._trained = False
        self._finetuned = False
        self._metrics: dict = {}
        self._comparison: dict = {}   # stores last train_compare() results

        self._load()

    # ── input text ────────────────────────────────────────────────────────────

    @staticmethod
    def build_input_text(report: dict) -> str:
        """
        Structured text for embedding:
          incident_type  location_raw  urgency  key_phrase_1  key_phrase_2 ...
        Example:
          'flood Kelaniya bridge CRITICAL ජලය පපුව දක්වා ඉහළ ගිහිං'
        """
        parts = [
            report.get('incident_type', ''),
            report.get('location_raw', ''),
            report.get('urgency', ''),
        ]
        parts += report.get('key_phrases', [])[:3]
        return ' '.join(x for x in parts if x).strip()

    # ── embeddings ────────────────────────────────────────────────────────────

    def get_embedding(self, report: dict) -> np.ndarray:
        """Use fine-tuned model if available, else the pipeline embedding."""
        if self._finetuned_embedder is not None and report.get('embedding') is None:
            text = self.build_input_text(report)
            return self._finetuned_embedder.encode(text, normalize_embeddings=True)
        # Reports already have embeddings from pipeline/embedder.py
        emb = report.get('embedding')
        if emb is not None:
            return emb
        # Fallback: zero vector
        return np.zeros(768, dtype=np.float32)

    # ── feature extraction ────────────────────────────────────────────────────

    def extract_features(self, r_a: dict, r_b: dict) -> list:
        """
        11-dimensional feature vector. Works with individual reports or
        UIR pseudo-reports (see ml_similarity).
        """
        emb_a = self.get_embedding(r_a)
        emb_b = self.get_embedding(r_b)
        sem_sim = float(np.dot(emb_a, emb_b))
        geo_sim = _geo_sim(r_a, r_b)
        time_sim = _time_sim(r_a, r_b)

        type_match = float(
            r_a.get('incident_type', '') == r_b.get('incident_type', '')
            and r_a.get('incident_type', '') != ''
        )

        u_a = _URGENCY.get(r_a.get('urgency', ''), 1)
        u_b = _URGENCY.get(r_b.get('urgency', ''), 1)
        urgency_diff = abs(u_a - u_b) / 3.0  # normalize to [0,1]

        lang_a = r_a.get('_ground_truth', {}).get('language', '')
        lang_b = r_b.get('_ground_truth', {}).get('language', '')
        cross_lingual = float(lang_a != lang_b and lang_a != '' and lang_b != '')

        both_critical = float(
            r_a.get('urgency') == 'CRITICAL' and r_b.get('urgency') == 'CRITICAL'
        )

        # Address-string similarity — distinguishes same-street neighbors
        # who share identical geocoded coords but have different house numbers
        loc_str_sim = _location_str_sim(r_a, r_b)
        geo_str_divergence = geo_sim * (1.0 - loc_str_sim)

        return [
            sem_sim, geo_sim, time_sim, type_match,
            urgency_diff, cross_lingual, both_critical,
            sem_sim * geo_sim,    # interaction: learned by XGBoost
            sem_sim * time_sim,   # interaction: learned by XGBoost
            loc_str_sim,          # address string similarity
            geo_str_divergence,   # high → same coords but different address
        ]

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, reports: list, method: str = 'auto') -> dict:
        """
        Train the pairwise classifier.

        method: 'xgboost' | 'random_forest' | 'lightgbm' |
                'gradientboost' | 'logistic' | 'auto'
          auto → XGBoost if available, else GradientBoosting, else Logistic

        Returns evaluation metrics dict.
        """
        if not _SKLEARN:
            print("  [MLDedup] scikit-learn not available — cannot train")
            return {}

        print("  [MLDedup] Generating training pairs...")
        gen = PairGenerator()
        pairs = gen.generate(reports, neg_ratio=3.0)

        if len(pairs) < 50:
            print(f"  [MLDedup] Only {len(pairs)} pairs — need more data to train")
            return {}

        X = np.array([self.extract_features(a, b) for a, b, _ in pairs])
        y = np.array([lbl for _, _, lbl in pairs])

        print(f"  [MLDedup] Training on {len(y)} pairs "
              f"({y.sum()} positive, {(~y.astype(bool)).sum()} negative)")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = self._build_classifier(method, y_tr)
        clf.fit(X_tr, y_tr)
        self._clf = clf
        self._trained = True

        metrics = self._evaluate(clf, method, X_val, y_val, len(y_tr))
        self._metrics = metrics

        self.save()
        return self._metrics

    def _build_classifier(self, method: str, y_tr: np.ndarray):
        """Instantiate the chosen classifier."""
        if method == 'auto':
            method = 'xgboost' if _XGB else 'lightgbm' if _LGB else 'gradientboost'

        if method == 'xgboost' and _XGB:
            n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
            return xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                scale_pos_weight=n_neg / max(n_pos, 1),
                use_label_encoder=False, eval_metric='logloss',
                random_state=42, verbosity=0,
            )
        if method == 'lightgbm' and _LGB:
            n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
            return lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=42, verbosity=-1,
            )
        if method == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200, max_depth=None,
                class_weight='balanced', random_state=42, n_jobs=-1,
            )
        if method == 'gradientboost':
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=42,
            )
        # logistic / fallback
        return LogisticRegression(C=1.0, max_iter=500, class_weight='balanced',
                                  random_state=42)

    def _evaluate(self, clf, method: str, X_val: np.ndarray,
                  y_val: np.ndarray, n_train: int) -> dict:
        """Score a fitted classifier and return a metrics dict."""
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        p, r, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(y_val, y_prob)
        except Exception:
            auc = 0.0

        print(f"  [MLDedup] Trained ({method}) | "
              f"P={p:.3f} R={r:.3f} F1={f1:.3f} AUC={auc:.3f}")

        result = {
            'method': method,
            'n_train': n_train, 'n_val': len(y_val),
            'precision': round(float(p), 4),
            'recall': round(float(r), 4),
            'f1': round(float(f1), 4),
            'auc_roc': round(float(auc), 4),
        }

        if hasattr(clf, 'feature_importances_'):
            fi = dict(zip(self.FEATURE_NAMES, clf.feature_importances_))
            fi_sorted = sorted(fi.items(), key=lambda x: -x[1])
            print("  [MLDedup] Feature importance:")
            for feat, imp in fi_sorted[:5]:
                bar = '█' * int(imp * 30)
                print(f"    {feat:15s} {imp:.3f}  {bar}")
            result['feature_importance'] = {k: round(float(v), 4) for k, v in fi.items()}

        return result

    # ── multi-algorithm comparison ─────────────────────────────────────────────

    def train_compare(self, reports: list) -> dict:
        """
        Train XGBoost, Random Forest, and LightGBM on the same train/val split,
        compare their metrics side-by-side, then activate the best model
        (highest AUC) as the live classifier.

        Returns a dict with per-algorithm results and the chosen winner.
        """
        if not _SKLEARN:
            return {"error": "scikit-learn not available"}

        print("  [MLDedup] Generating training pairs for comparison...")
        gen = PairGenerator()
        pairs = gen.generate(reports, neg_ratio=3.0)

        if len(pairs) < 50:
            return {"error": f"only {len(pairs)} pairs — need more data"}

        X = np.array([self.extract_features(a, b) for a, b, _ in pairs])
        y = np.array([lbl for _, _, lbl in pairs])
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        candidates = []
        if _XGB:
            candidates.append('xgboost')
        candidates.append('random_forest')
        if _LGB:
            candidates.append('lightgbm')

        print(f"\n  {'='*55}")
        print(f"  ALGORITHM COMPARISON  ({len(y_tr)} train / {len(y_val)} val pairs)")
        print(f"  {'='*55}")

        results = {}
        best_method, best_clf, best_auc = None, None, -1.0

        for method in candidates:
            print(f"\n  ── {method} ──")
            clf = self._build_classifier(method, y_tr)
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            elapsed = time.time() - t0
            m = self._evaluate(clf, method, X_val, y_val, len(y_tr))
            m['train_seconds'] = round(elapsed, 2)
            results[method] = m

            if m['auc_roc'] > best_auc:
                best_auc = m['auc_roc']
                best_method = method
                best_clf = clf

        # Print comparison table
        print(f"\n  {'─'*55}")
        print(f"  {'Algorithm':15s}  {'F1':>6}  {'AUC':>6}  {'Precision':>9}  {'Recall':>6}  {'Secs':>5}")
        print(f"  {'─'*55}")
        for m_name, m in results.items():
            marker = ' ◀ best' if m_name == best_method else ''
            print(f"  {m_name:15s}  {m['f1']:6.4f}  {m['auc_roc']:6.4f}  "
                  f"{m['precision']:9.4f}  {m['recall']:6.4f}  {m['train_seconds']:5.1f}s{marker}")
        print(f"  {'─'*55}")

        # Activate the best model
        self._clf = best_clf
        self._trained = True
        self._metrics = results[best_method]
        self._comparison = {
            'algorithms': results,
            'best': best_method,
            'best_auc': best_auc,
        }

        self.save()
        return self._comparison

    # ── fine-tuning (optional) ────────────────────────────────────────────────

    def fine_tune(self, reports: list, epochs: int = 3) -> bool:
        """
        Fine-tune LaBSE on incident pairs using ContrastiveLoss.

        This is the step that gives genuine cross-lingual improvement:
        after fine-tuning, a Sinhala flood report and a Tamil flood report
        describing the same event will have higher cosine similarity than
        two different incidents in the same language.

        Requires: sentence-transformers installed + LaBSE downloaded.
        Saves fine-tuned model to data/finetuned_labse/.
        """
        try:
            from sentence_transformers import SentenceTransformer, InputExample
            from sentence_transformers.losses import OnlineContrastiveLoss
            from torch.utils.data import DataLoader
        except ImportError:
            print("  [MLDedup] sentence-transformers not available — skipping fine-tune")
            return False

        print("  [MLDedup] Loading LaBSE for fine-tuning...")
        try:
            model = SentenceTransformer('LaBSE')
        except Exception as e:
            print(f"  [MLDedup] LaBSE unavailable: {e} — skipping fine-tune")
            return False

        # Build training examples — focus on cross-lingual pairs where fine-tuning
        # makes the biggest difference
        gen = PairGenerator()
        pairs = gen.generate(reports, neg_ratio=2.0)

        train_examples = []
        for r_a, r_b, lbl in pairs:
            text_a = self.build_input_text(r_a)
            text_b = self.build_input_text(r_b)
            if text_a and text_b:
                train_examples.append(InputExample(texts=[text_a, text_b],
                                                   label=float(lbl)))

        if len(train_examples) < 50:
            print("  [MLDedup] Too few examples for fine-tuning")
            return False

        print(f"  [MLDedup] Fine-tuning on {len(train_examples)} pairs "
              f"for {epochs} epoch(s)...")
        train_dl = DataLoader(train_examples, shuffle=True, batch_size=16)
        loss = OnlineContrastiveLoss(model)

        model.fit(
            train_objectives=[(train_dl, loss)],
            epochs=epochs,
            warmup_steps=max(10, len(train_examples) // 20),
            show_progress_bar=True,
        )

        self.finetuned_path.mkdir(parents=True, exist_ok=True)
        model.save(str(self.finetuned_path))
        self._finetuned_embedder = model
        self._finetuned = True

        print(f"  [MLDedup] Fine-tuned model saved → {self.finetuned_path}")
        return True

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, r_a: dict, r_b: dict) -> float:
        """P(same incident) for a report pair. Falls back to weighted sum."""
        if self._clf is not None:
            feat = np.array([self.extract_features(r_a, r_b)])
            try:
                proba = self._clf.predict_proba(feat)[0]
                merge_idx = list(self._clf.classes_).index(1)
                return float(proba[merge_idx])
            except Exception:
                pass
        # Fallback: weighted sum
        feats = self.extract_features(r_a, r_b)
        sem, geo, time = feats[0], feats[1], feats[2]
        return 0.5 * sem + 0.3 * geo + 0.2 * time

    def ml_similarity(self, report: dict, uir: dict) -> dict:
        """
        Drop-in replacement for combined_similarity(report, uir) in clustering.py.

        Builds a pseudo-report from the UIR's centroid embedding + metadata,
        then runs the pairwise classifier. Returns the same dict format.
        """
        # Build pseudo-report from UIR so extract_features works uniformly
        uir_pseudo = {
            'embedding': uir.get('centroid_embedding'),
            'lat': uir.get('lat'),
            'lng': uir.get('lng'),
            'incident_type': uir.get('incident_type', 'unknown'),
            'urgency': uir.get('urgency', 'MEDIUM'),
            'channel': uir.get('dominant_channel', 'voice'),
            'timestamp': uir.get('last_updated', uir.get('created_at')),
            'receive_time': uir.get('created_at'),
            # Concatenate source address strings so location_str_sim can compare
            # against the incoming report's location_raw
            'location_raw': ' '.join(uir.get('location', {}).get('source_strings', [])[:3]),
            '_ground_truth': {
                'language': _infer_uir_language(uir),
            },
        }

        # Raw component scores (always computed, used for safety gate & logging)
        emb_r = report.get('embedding')
        emb_u = uir.get('centroid_embedding')
        s_sem = float(np.dot(emb_r, emb_u)) if (emb_r is not None and emb_u is not None) else 0.0
        s_geo = _geo_sim(report, uir)
        s_time = _time_sim(report, uir_pseudo)

        if self._clf is not None:
            combined = self.predict_proba(report, uir_pseudo)
        else:
            combined = 0.5 * s_sem + 0.3 * s_geo + 0.2 * s_time

        return {
            'semantic': s_sem,
            'geographic': s_geo,
            'temporal': s_time,
            'combined': combined,
            'distance': 1.0 - combined,
        }

    # ── cross-lingual evaluation ───────────────────────────────────────────────

    def evaluate_cross_lingual(self, reports: list) -> dict:
        """
        Pairwise AUC split by language combination.
        Shows exactly how much the ML model improves on cross-lingual pairs vs
        same-lingual pairs.

        Returns dict with AUC per language pair and overall improvement.
        """
        if not _SKLEARN or self._clf is None:
            return {"error": "model not trained"}

        gen = PairGenerator()
        pairs = gen.generate(reports, neg_ratio=3.0)

        results_by_lang: dict = {}
        for r_a, r_b, lbl in pairs:
            la = r_a.get('_ground_truth', {}).get('language', 'unknown')
            lb = r_b.get('_ground_truth', {}).get('language', 'unknown')
            lang_key = tuple(sorted([la, lb]))
            results_by_lang.setdefault(lang_key, ([], []))
            score = self.predict_proba(r_a, r_b)
            results_by_lang[lang_key][0].append(lbl)
            results_by_lang[lang_key][1].append(score)

        output = {}
        for lang_key, (labels, scores) in results_by_lang.items():
            if len(set(labels)) < 2:
                continue
            try:
                auc = roc_auc_score(labels, scores)
                output[f"{lang_key[0]} ↔ {lang_key[1]}"] = round(auc, 4)
            except Exception:
                pass

        return output

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'clf': self._clf,
                'trained': self._trained,
                'finetuned': self._finetuned,
                'metrics': self._metrics,
                'comparison': self._comparison,
            }, f)

    def _load(self):
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    state = pickle.load(f)
                self._clf = state.get('clf')
                self._trained = state.get('trained', False)
                self._finetuned = state.get('finetuned', False)
                self._metrics = state.get('metrics', {})
                self._comparison = state.get('comparison', {})
            except Exception:
                pass

        if self._finetuned and self.finetuned_path.exists():
            try:
                from sentence_transformers import SentenceTransformer
                self._finetuned_embedder = SentenceTransformer(str(self.finetuned_path))
                print(f"  [MLDedup] Loaded fine-tuned embedder from {self.finetuned_path}")
            except Exception:
                self._finetuned = False

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def is_finetuned(self) -> bool:
        return self._finetuned

    @property
    def metrics(self) -> dict:
        return self._metrics

    @property
    def comparison(self) -> dict:
        return self._comparison

    def status_line(self) -> str:
        if self._trained:
            m = self._metrics
            cmp = (f" | best of {len(self._comparison['algorithms'])} algos"
                   if self._comparison else "")
            return (f"[MLDedup] TRAINED ({m.get('method','?')}) | "
                    f"F1={m.get('f1','?')} AUC={m.get('auc_roc','?')}"
                    f"{cmp} | "
                    f"fine-tuned={'yes' if self._finetuned else 'no'}")
        return "[MLDedup] NOT TRAINED — using fixed-weight fallback"


# ── helpers ───────────────────────────────────────────────────────────────────

def _infer_uir_language(uir: dict) -> str:
    """Infer the dominant language of a UIR from its source reports."""
    langs = [r.get('_ground_truth', {}).get('language', '')
             for r in uir.get('source_reports', [])]
    langs = [l for l in langs if l]
    if not langs:
        return 'unknown'
    return Counter(langs).most_common(1)[0][0]


# ── module-level singleton + drop-in function ─────────────────────────────────

_DEDUPLICATOR: Optional[MLDeduplicator] = None


def get_deduplicator(data_dir: Path = None) -> Optional[MLDeduplicator]:
    return _DEDUPLICATOR


def init_deduplicator(data_dir: Path) -> MLDeduplicator:
    global _DEDUPLICATOR
    _DEDUPLICATOR = MLDeduplicator(data_dir)
    return _DEDUPLICATOR


def ml_similarity(report: dict, uir: dict) -> dict:
    """
    Module-level drop-in for combined_similarity(report, uir).

    Uses the trained MLDeduplicator if initialised (via init_deduplicator).
    Falls back to fixed-weight formula otherwise.
    Compatible with clustering.py's IncidentClusterEngine without changes.
    """
    if _DEDUPLICATOR is not None and _DEDUPLICATOR.is_trained:
        return _DEDUPLICATOR.ml_similarity(report, uir)

    # Fallback: fixed weights (same as clustering.py default)
    emb_r = report.get('embedding')
    emb_u = uir.get('centroid_embedding')
    s_sem = float(np.dot(emb_r, emb_u)) if (emb_r is not None and emb_u is not None) else 0.0
    s_geo = _geo_sim(report, uir)
    s_time = _time_sim(report, uir)
    combined = 0.5 * s_sem + 0.3 * s_geo + 0.2 * s_time
    return {'semantic': s_sem, 'geographic': s_geo, 'temporal': s_time,
            'combined': combined, 'distance': 1.0 - combined}


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pipeline.normalizer import normalize_batch, reset_counters
    from pipeline.gazetteer import resolve_batch
    from pipeline.embedder import embed_batch

    data_path = Path('data/disaster_dataset_600.json')
    if not data_path.exists():
        print("Dataset not found. Run: python generate_dataset.py")
        sys.exit(1)

    import json as _json
    with open(data_path, encoding='utf-8') as f:
        dataset = _json.load(f)

    print("Preparing reports...")
    reset_counters()
    reports = normalize_batch(dataset['reports'])
    reports = resolve_batch(reports)
    reports = embed_batch(reports)

    dedup = MLDeduplicator(Path('data'))

    print("\n" + "="*60)
    print("  STEP 1: Compare XGBoost vs Random Forest vs LightGBM")
    print("="*60)
    comparison = dedup.train_compare(reports)
    print(f"\n  Best algorithm: {comparison.get('best')}  "
          f"(AUC={comparison.get('best_auc', '?')})")

    print("\n" + "="*60)
    print("  STEP 2: Cross-lingual AUC breakdown")
    print("="*60)
    cl_auc = dedup.evaluate_cross_lingual(reports)
    for lang_pair, auc in sorted(cl_auc.items()):
        print(f"  {lang_pair:35s}  AUC = {auc:.4f}")

    print("\n" + "="*60)
    print("  STEP 3: Fine-tune LaBSE (requires model download)")
    print("="*60)
    do_finetune = '--finetune' in sys.argv
    if do_finetune:
        success = dedup.fine_tune(reports, epochs=3)
        if success:
            print("  Re-running classifier with fine-tuned embeddings...")
            reports = embed_batch(reports)  # re-embed with fine-tuned model
            metrics2 = dedup.train(reports, method='auto')
            print(f"\n  Post-finetune metrics: {metrics2}")
    else:
        print("  (skip — run with --finetune to enable)")

    print("\n" + "="*60)
    print("  STEP 4: Compare vs fixed weights (before/after)")
    print("="*60)
    print("  [fixed 0.5/0.3/0.2]  Use clustering.py as-is")
    print("  [trained classifier]  set_similarity_model(dedup) in server.py")
    print(f"\n  {dedup.status_line()}")
