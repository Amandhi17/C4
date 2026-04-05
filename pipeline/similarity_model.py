"""
Learned Similarity Model — replaces fixed weights 0.5/0.3/0.2
==============================================================
Trains a logistic regression on operator correction logs.
Falls back to fixed weights until >= MIN_TRAIN_EXAMPLES labeled pairs exist.

Why this beats fixed weights:
- Learns the *interaction* between semantic × geo × temporal
  (e.g. high semantic + mismatched geo is a stronger signal than either alone)
- Cross-lingual pairs (Sinhala ↔ Tamil) with real LaBSE shift the semantic
  distribution; learned weights adapt automatically
- Operator corrections (split/merge) provide ground truth that no hand-tuning
  can replicate

Feature vector: [s_sem, s_geo, s_time, s_sem*s_geo, s_sem*s_time]
Target: 1 = same incident (should merge), 0 = different (should not merge)
"""

import json
import pickle
import numpy as np
from pathlib import Path

MIN_TRAIN_EXAMPLES = 20
_DEFAULT_WEIGHTS = np.array([0.5, 0.3, 0.2], dtype=np.float64)


class LearnedSimilarityModel:
    """
    Wraps a LogisticRegression trained on operator feedback.
    Before MIN_TRAIN_EXAMPLES, `predict_score` returns the fixed-weight sum.
    After training, returns the model's merge probability.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.model_path = self.data_dir / "similarity_model.pkl"
        self.examples_path = self.data_dir / "similarity_examples.jsonl"

        self._weights = _DEFAULT_WEIGHTS.copy()
        self._clf = None
        self._trained = False
        self._n_examples = 0

        self._load_state()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_state(self):
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    state = pickle.load(f)
                self._weights = state.get("weights", _DEFAULT_WEIGHTS.copy())
                self._clf = state.get("clf")
                self._trained = state.get("trained", False)
                self._n_examples = state.get("n_examples", 0)
            except Exception:
                pass

    def save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "weights": self._weights,
                "clf": self._clf,
                "trained": self._trained,
                "n_examples": self._n_examples,
            }, f)

    # ── training data ─────────────────────────────────────────────────────────

    def add_example(self, semantic: float, geographic: float, temporal: float, label: int):
        """
        Persist one labeled pair.
        label=1  → reports describe the same incident (operator merged)
        label=0  → reports describe different incidents (operator split)
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        row = {"semantic": semantic, "geographic": geographic,
               "temporal": temporal, "label": label}
        with open(self.examples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self._n_examples += 1

    def _load_examples(self):
        if not self.examples_path.exists():
            return np.empty((0, 5)), np.empty(0)
        X, y = [], []
        with open(self.examples_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    X.append(self._featurize(ex["semantic"], ex["geographic"], ex["temporal"]))
                    y.append(int(ex["label"]))
                except Exception:
                    pass
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)

    @staticmethod
    def _featurize(sem: float, geo: float, time: float) -> list:
        """[sem, geo, time, sem×geo, sem×time]  — interaction terms let the model
        learn that high semantic similarity *plus* close geography is a much
        stronger merge signal than either alone."""
        return [sem, geo, time, sem * geo, sem * time]

    # ── training ──────────────────────────────────────────────────────────────

    def retrain(self) -> bool:
        """
        Fit or re-fit the classifier on all accumulated examples.
        Returns True if training succeeded.
        No-op if fewer than MIN_TRAIN_EXAMPLES examples exist.
        """
        X, y = self._load_examples()
        if len(y) < MIN_TRAIN_EXAMPLES:
            print(f"  [SimilarityModel] Need {MIN_TRAIN_EXAMPLES} examples to train "
                  f"(have {len(y)}) — using fixed weights {self._weights.tolist()}")
            return False

        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced",
                                     solver="lbfgs")
            clf.fit(X, y)
            self._clf = clf
            self._trained = True
            self._n_examples = len(y)

            # Derive interpretable main weights from the first 3 coefficients
            coef = clf.coef_[0]          # [sem, geo, time, sem×geo, sem×time]
            main_w = np.abs(coef[:3])
            total = main_w.sum()
            if total > 0:
                self._weights = main_w / total

            self.save()
            print(f"  [SimilarityModel] Trained on {len(y)} operator-labeled pairs. "
                  f"Learned weights → sem={self._weights[0]:.3f}, "
                  f"geo={self._weights[1]:.3f}, time={self._weights[2]:.3f}")
            return True

        except Exception as e:
            print(f"  [SimilarityModel] Training failed: {e}")
            return False

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_score(self, semantic: float, geographic: float, temporal: float) -> float:
        """
        Return a similarity score in [0, 1].
        - If model is trained: probability of 'same incident' from LR classifier
          (captures interaction terms LaBSE makes meaningful)
        - Otherwise: weighted sum with fixed weights [0.5, 0.3, 0.2]
        """
        if self._trained and self._clf is not None:
            feat = np.array([self._featurize(semantic, geographic, temporal)])
            try:
                proba = self._clf.predict_proba(feat)[0]
                merge_idx = list(self._clf.classes_).index(1)
                return float(proba[merge_idx])
            except Exception:
                pass
        w = self._weights
        return float(w[0] * semantic + w[1] * geographic + w[2] * temporal)

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def n_examples(self) -> int:
        return self._n_examples

    @property
    def weights(self) -> list:
        return self._weights.tolist()

    def status_line(self) -> str:
        if self._trained:
            return (f"[SimilarityModel] TRAINED on {self._n_examples} pairs | "
                    f"weights sem={self._weights[0]:.3f} geo={self._weights[1]:.3f} "
                    f"time={self._weights[2]:.3f}")
        return (f"[SimilarityModel] FIXED WEIGHTS (need {MIN_TRAIN_EXAMPLES}, "
                f"have {self._n_examples}) | weights {self._weights.tolist()}")
