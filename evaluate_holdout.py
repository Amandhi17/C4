"""
Evaluate the already-trained MLDeduplicator on a fresh held-out dataset.

Usage
-----
    # 1) Generate a holdout dataset with a different seed
    python generate_dataset.py --seed 123 --out data/holdout.json

    # 2) Score the trained model on it (no retraining)
    python evaluate_holdout.py --dataset data/holdout.json

Loads the trained classifier from data/ml_deduplicator.pkl, builds pairs from
the holdout reports, and prints F1 / AUC / Precision / Recall on data the model
has never seen during training.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.normalizer import normalize_batch, reset_counters
from pipeline.gazetteer import resolve_batch
from pipeline.embedder import embed_batch
from pipeline.ml_deduplication import MLDeduplicator, PairGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/holdout.json",
                    help="Path to held-out dataset JSON")
    ap.add_argument("--model-dir", default="data",
                    help="Directory containing ml_deduplicator.pkl")
    ap.add_argument("--pair-seed", type=int, default=999,
                    help="Seed for PairGenerator (different from training seed)")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print(f"Generate one with:")
        print(f"  python generate_dataset.py --seed 123 --out {dataset_path}")
        sys.exit(1)

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"  {len(dataset['reports'])} raw reports "
          f"(generator seed: {dataset.get('metadata', {}).get('seed', 'unknown')})")

    print("Preparing reports (normalize, resolve, embed)...")
    reset_counters()
    reports = normalize_batch(dataset["reports"])
    reports = resolve_batch(reports)
    reports = embed_batch(reports)

    model_dir = Path(args.model_dir)
    print(f"Loading trained model from {model_dir}/ml_deduplicator.pkl...")
    dedup = MLDeduplicator(model_dir)
    if not dedup.is_trained:
        print("ERROR: model is not trained. First run:")
        print("  python pipeline/ml_deduplication.py")
        sys.exit(1)
    print(f"  {dedup.status_line()}")

    print(f"\nGenerating pairs from holdout (PairGenerator seed={args.pair_seed})...")
    gen = PairGenerator(seed=args.pair_seed)
    pairs = gen.generate(reports)

    print(f"Scoring {len(pairs)} pairs against trained model (no .fit() call)...")
    y_true = np.array([lbl for _, _, lbl in pairs])
    y_score = np.array([dedup.predict_proba(a, b) for a, b, _ in pairs])
    y_pred = (y_score >= 0.5).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_score) if len(set(y_true.tolist())) > 1 else 0.0

    bar = "=" * 55
    print(f"\n  {bar}")
    print(f"  HOLDOUT EVALUATION  (model has never seen this data)")
    print(f"  {bar}")
    print(f"  Dataset:    {dataset_path}")
    print(f"  Reports:    {len(reports)}")
    print(f"  Pairs:      {len(pairs)} (pos={int(y_true.sum())}, "
          f"neg={int((y_true == 0).sum())})")
    print(f"  Precision:  {p:.4f}")
    print(f"  Recall:     {r:.4f}")
    print(f"  F1:         {f1:.4f}")
    print(f"  AUC-ROC:    {auc:.4f}")
    print(f"  {bar}")

    if dedup.metrics:
        print("\n  Train-time validation vs. holdout:")
        sep = "-" * 50
        print(f"  {sep}")
        print(f"  {'metric':<12} {'train val':>12} {'holdout':>12} {'delta':>10}")
        print(f"  {sep}")
        holdout_vals = {"precision": p, "recall": r, "f1": f1, "auc_roc": auc}
        deltas = []
        for k in ("precision", "recall", "f1", "auc_roc"):
            tv = dedup.metrics.get(k, 0.0)
            hv = holdout_vals[k]
            delta = hv - tv
            deltas.append(delta)
            print(f"  {k:<12} {tv:>12.4f} {hv:>12.4f} {delta:>+10.4f}")
        print(f"  {sep}")

        max_drop = -min(deltas)
        print()
        if max_drop > 0.05:
            print(f"  Significant drop ({max_drop:.3f}) on holdout — "
                  f"model may be overfitting to the training distribution.")
        elif max_drop > 0.02:
            print(f"  Modest drop ({max_drop:.3f}) — within typical sampling noise.")
        else:
            print(f"  Holdout metrics within {max_drop:+.4f} of training — "
                  f"generalizes well across draws.")


if __name__ == "__main__":
    main()
