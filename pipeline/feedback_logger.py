"""
Operator Correction Logger
==========================
Every time an operator splits, merges, or corrects a UIR the event is
appended to data/operator_corrections.jsonl as a gold-label training row.

split  → reports that were merged but shouldn't be  → label=0 pairs
merge  → UIRs kept separate but should be together  → label=1 pairs
correct → field-level fix (audit trail only, not used for similarity)

extract_training_pairs() converts the log into (features, label) dicts
that can be fed directly into LearnedSimilarityModel.add_example().
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class FeedbackLogger:

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.log_path = self.data_dir / "operator_corrections.jsonl"

    # ── write helpers ─────────────────────────────────────────────────────────

    def _append(self, entry: dict):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_split(self, uir_id: str, kept_source_ids: list,
                  split_source_ids: list, operator_id: str, note: str = ""):
        """
        Operator decided that some sources in this UIR belong to a different
        incident. kept + split are now separate incidents → label=0 pairs.
        """
        self._append({
            "action": "split",
            "uir_id": uir_id,
            "kept_source_ids": kept_source_ids,
            "split_source_ids": split_source_ids,
            "operator_id": operator_id,
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def log_merge(self, uir_id_a: str, uir_id_b: str,
                  sim_scores: dict, operator_id: str, note: str = ""):
        """
        Operator decided two separate UIRs are actually the same incident.
        The sim_scores at decision time become a label=1 training row.
        sim_scores = {'semantic': float, 'geographic': float, 'temporal': float}
        """
        self._append({
            "action": "merge",
            "uir_id_a": uir_id_a,
            "uir_id_b": uir_id_b,
            "sim_scores": sim_scores,
            "operator_id": operator_id,
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def log_correct(self, uir_id: str, field: str, old_value,
                    new_value, operator_id: str, note: str = ""):
        """Field-level correction — audit trail, not used for similarity training."""
        self._append({
            "action": "correct",
            "uir_id": uir_id,
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "operator_id": operator_id,
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ── read ──────────────────────────────────────────────────────────────────

    def load_corrections(self) -> list:
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        return entries

    # ── training pair extraction ───────────────────────────────────────────────

    def extract_training_pairs(self, engine) -> list:
        """
        Convert the correction log into labeled feature dicts.

        Each dict has:  {'semantic': float, 'geographic': float,
                          'temporal': float,  'label': int}

        These are ready to pass to LearnedSimilarityModel.add_example().

        Split events   → cross-pairs between kept and split source groups, label=0
        Merge events   → centroid-pair of the two UIRs using logged sim_scores, label=1
        """
        from pipeline.clustering import (
            semantic_similarity, geographic_similarity, temporal_similarity
        )

        corrections = self.load_corrections()
        uir_index = {u["uir_id"]: u for u in engine.active_uirs}
        pairs = []

        for entry in corrections:
            action = entry.get("action")

            if action == "split":
                uir = uir_index.get(entry.get("uir_id"))
                if not uir:
                    continue
                kept_ids = set(entry.get("kept_source_ids", []))
                split_ids = set(entry.get("split_source_ids", []))
                kept_reps = [r for r in uir.get("source_reports", [])
                             if r.get("source_id") in kept_ids
                             and r.get("embedding") is not None]
                split_reps = [r for r in uir.get("source_reports", [])
                              if r.get("source_id") in split_ids
                              and r.get("embedding") is not None]

                # Cross pairs — these reports should NOT have been merged
                for rk in kept_reps[:3]:
                    for rs in split_reps[:3]:
                        s_sem = float(semantic_similarity(rk["embedding"], rs["embedding"]))
                        s_geo = geographic_similarity(rk, rs)
                        # Use rs timestamp as "uir time" proxy
                        rs_time = rs.get("timestamp", rs.get("receive_time"))
                        s_time = temporal_similarity(
                            rk, {"last_updated": rs_time, "created_at": rs_time}
                        )
                        pairs.append({"semantic": s_sem, "geographic": s_geo,
                                      "temporal": s_time, "label": 0})

            elif action == "merge":
                sim = entry.get("sim_scores", {})
                # Use logged scores if present (most accurate — captured at decision time)
                if sim.get("semantic") is not None:
                    pairs.append({
                        "semantic": float(sim["semantic"]),
                        "geographic": float(sim["geographic"]),
                        "temporal": float(sim["temporal"]),
                        "label": 1,
                    })
                else:
                    # Fall back to recomputing from current centroid embeddings
                    uir_a = uir_index.get(entry.get("uir_id_a"))
                    uir_b = uir_index.get(entry.get("uir_id_b"))
                    if (uir_a and uir_b
                            and uir_a.get("centroid_embedding") is not None
                            and uir_b.get("centroid_embedding") is not None):
                        s_sem = float(semantic_similarity(
                            uir_a["centroid_embedding"], uir_b["centroid_embedding"]))
                        s_geo = geographic_similarity(uir_a, uir_b)
                        s_time = temporal_similarity(uir_a, uir_b)
                        pairs.append({"semantic": s_sem, "geographic": s_geo,
                                      "temporal": s_time, "label": 1})

        return pairs
