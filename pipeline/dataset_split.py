"""
Dataset split utility — splits reports into train/test sets BY INCIDENT to
prevent data leakage in ML evaluation.

Splitting randomly by report leaks information: if FLOOD_093 has 5 reports and
the random split sends 4 to train and 1 to test, the model has effectively
"seen" that incident during training. At test time it gets unfair credit on a
known-incident pair.

Splitting by incident_id guarantees every report of a given incident ends up
entirely in train OR entirely in test — never both.
"""

import random
from typing import Tuple


def split_by_incident(reports: list,
                      ratio: float = 0.8,
                      seed: int = 42) -> Tuple[list, list]:
    """
    Group reports by `_ground_truth.incident_id`, shuffle the incident IDs with
    a fixed seed, then assign the first `ratio` fraction of incidents to train
    and the remainder to test.

    Args:
        reports: list of normalized report dicts (each with _ground_truth.incident_id)
        ratio:   fraction of *incidents* (not reports) for the train side
        seed:    RNG seed for the incident shuffle (reproducible)

    Returns:
        (train_reports, test_reports) — preserves original report order within
        each side.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")

    by_incident: dict = {}
    for r in reports:
        iid = r.get('_ground_truth', {}).get('incident_id', 'UNKNOWN')
        by_incident.setdefault(iid, []).append(r)

    # Sort first for determinism across runs, then shuffle with seeded RNG
    incident_ids = sorted(by_incident.keys())
    rng = random.Random(seed)
    rng.shuffle(incident_ids)

    n_train = int(round(len(incident_ids) * ratio))
    train_iids = set(incident_ids[:n_train])

    train_reports, test_reports = [], []
    for r in reports:
        iid = r.get('_ground_truth', {}).get('incident_id', 'UNKNOWN')
        if iid in train_iids:
            train_reports.append(r)
        else:
            test_reports.append(r)

    return train_reports, test_reports


def split_summary(train_reports: list, test_reports: list) -> dict:
    """Diagnostic counts for the split — useful for printing a banner."""
    def _n_incidents(reps):
        return len({r.get('_ground_truth', {}).get('incident_id', 'UNKNOWN')
                    for r in reps})

    return {
        'train_reports':   len(train_reports),
        'train_incidents': _n_incidents(train_reports),
        'test_reports':    len(test_reports),
        'test_incidents':  _n_incidents(test_reports),
    }
