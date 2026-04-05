"""
C4 Pipeline — Stage 4: Cross-Lingual Embedding
================================================
Primary: LaBSE (768-dim, 109 languages, cross-lingual)
Fallback: Multilingual character n-gram hashing (works offline)

The fallback uses character n-grams (3,4,5-gram) with a keyword bridge
for cross-language emergency terms. For production, deploy with LaBSE.
"""

import numpy as np
import hashlib
import time

EMBEDDING_DIM = 768
_model = None
_use_labse = False
_init_done = False


def _try_load_labse():
    global _model, _use_labse, _init_done
    _init_done = True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('LaBSE')
        _use_labse = True
        print("  [LaBSE] Model loaded (768-dim, 109 languages)")
        return True
    except Exception as e:
        print(f"  [LaBSE] Unavailable: {type(e).__name__}")
        print("  [FALLBACK] Using multilingual character n-gram embeddings")
        _use_labse = False
        return False


# Cross-language keyword bridge
KEYWORD_BRIDGE = {
    "ජලය": "water", "වතුර": "water", "ගංවතුර": "flood",
    "පපුව": "chest", "බෙල්ල": "neck", "කකුල්": "legs",
    "වහළ": "roof", "ගෙදර": "house", "මාර්ගය": "road",
    "උදව්": "help", "බේරගන්න": "rescue", "ළමයි": "children",
    "විදුලිය": "electricity", "පාලම": "bridge",
    "මට්ටමට": "level", "ඉණ": "waist", "ඉහළ": "rising",
    "මහ": "big", "කැපිලා": "cut", "යටවෙලා": "submerged",
    "ගස්": "trees", "කඩිලා": "fallen", "සතුන්": "animals",
    "ආහාර": "food", "හදිසි": "emergency", "අවදානම්": "dangerous",
    "மார்பு": "chest", "கழுத்து": "neck", "தண்ணீர்": "water",
    "வெள்ளம்": "flood", "கூரை": "roof", "வீடு": "house",
    "சாலை": "road", "உதவி": "help", "காப்பாற்று": "rescue",
    "குழந்தை": "children", "மின்சாரம்": "electricity",
    "பாலம்": "bridge", "மூழ்கி": "submerged",
    "உயர்ந்து": "rising", "அளவு": "level",
    "wathura": "water", "jalaya": "water", "gangwathur": "flood",
    "wahale": "roof", "ge": "house", "udaw": "help",
    "beraganna": "rescue", "lamayi": "children",
    "paalam": "bridge", "viduliya": "electricity",
    "mattamata": "level", "aawa": "came",
}


def _char_ngrams(text, ns=(3, 4, 5)):
    text = text.lower().strip()
    ngrams = []
    for n in ns:
        for i in range(max(0, len(text) - n + 1)):
            ngrams.append(text[i:i+n])
    return ngrams


def _bridge_text(text):
    words = text.split()
    bridged = []
    for word in words:
        clean = word.strip('.,!?;:()[]{}')
        if clean in KEYWORD_BRIDGE:
            bridged.append(KEYWORD_BRIDGE[clean])
    return text + ' ' + ' '.join(bridged) if bridged else text


def _hash_embed(text, dim=EMBEDDING_DIM):
    if not text or not text.strip():
        return np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    ngrams = _char_ngrams(text)
    if not ngrams:
        return vec
    for ng in ngrams:
        h = hashlib.md5(ng.encode('utf-8')).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.astype(np.float32)


def build_input_text(report):
    parts = [
        report.get('incident_type', ''),
        report.get('location_raw', ''),
        ' '.join(report.get('key_phrases', []))
    ]
    return ' '.join(x for x in parts if x).strip()


def get_embedding(report):
    text = build_input_text(report)
    if _use_labse and _model is not None:
        if not text:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        return _model.encode(text, normalize_embeddings=True)
    else:
        return _hash_embed(_bridge_text(text) if text else "")


def embed_batch(reports, batch_size=32):
    global _init_done
    if not _init_done:
        _try_load_labse()

    texts = [build_input_text(r) for r in reports]
    start = time.time()

    if _use_labse and _model is not None:
        non_empty = [(i, t) for i, t in enumerate(texts) if t]
        embed_map = {}
        if non_empty:
            indices, ne_texts = zip(*non_empty)
            embs = _model.encode(list(ne_texts), normalize_embeddings=True,
                                 batch_size=batch_size, show_progress_bar=False)
            embed_map = dict(zip(indices, embs))
        for i, r in enumerate(reports):
            r['embedding'] = embed_map.get(i, np.zeros(EMBEDDING_DIM, dtype=np.float32))
    else:
        for i, r in enumerate(reports):
            bridged = _bridge_text(texts[i]) if texts[i] else ""
            r['embedding'] = _hash_embed(bridged)

    elapsed = time.time() - start
    method = "LaBSE" if _use_labse else "CharNGram+Bridge"
    print(f"  [{method}] Encoded {len(reports)} reports in {elapsed:.2f}s "
          f"({len(reports)/max(elapsed,0.001):.0f} reports/sec)")
    return reports


def evaluate_cross_lingual_pairs(reports: list) -> dict:
    """
    Test whether LaBSE brings same-incident cross-lingual pairs (Sinhala ↔ Tamil)
    closer together than different-incident pairs.

    Returns a dict with:
      same_lingual_sim    — avg cosine sim for pairs where both reports same language
      cross_lingual_sim   — avg cosine sim for Sinhala ↔ Tamil pairs, same incident
      cross_incident_sim  — avg cosine sim for different incidents (baseline noise)
      separation_gap      — cross_lingual_sim - cross_incident_sim  (want > 0.10)
      labse_active        — True if real LaBSE was used (not fallback)

    This is the key number: a gap > 0.10 means LaBSE is giving you genuine
    cross-lingual merging ability that fixed char-ngram embeddings cannot.
    """
    import numpy as np

    same_sims, cross_lang_sims, cross_inc_sims = [], [], []

    # Group reports by ground-truth incident_id
    by_incident: dict = {}
    for r in reports:
        inc_id = r.get("_ground_truth", {}).get("incident_id", "UNKNOWN")
        if inc_id != "UNKNOWN":
            by_incident.setdefault(inc_id, []).append(r)

    incident_ids = list(by_incident.keys())

    # Same-incident pairs — split by language
    for inc_id, reps in by_incident.items():
        sinhala_reps = [r for r in reps
                        if r.get("_ground_truth", {}).get("language") in ("sinhala", "romanized_sinhala")
                        and r.get("embedding") is not None]
        tamil_reps = [r for r in reps
                      if r.get("_ground_truth", {}).get("language") == "tamil"
                      and r.get("embedding") is not None]

        # Same-lingual (Sinhala-Sinhala)
        for i in range(min(3, len(sinhala_reps))):
            for j in range(i + 1, min(4, len(sinhala_reps))):
                same_sims.append(float(np.dot(sinhala_reps[i]["embedding"],
                                              sinhala_reps[j]["embedding"])))

        # Cross-lingual (Sinhala-Tamil, same incident)
        for si in sinhala_reps[:2]:
            for ti in tamil_reps[:2]:
                cross_lang_sims.append(float(np.dot(si["embedding"], ti["embedding"])))

    # Cross-incident pairs (different incidents, baseline)
    for i in range(min(8, len(incident_ids))):
        for j in range(i + 1, min(9, len(incident_ids))):
            r1 = by_incident[incident_ids[i]][0]
            r2 = by_incident[incident_ids[j]][0]
            if r1.get("embedding") is not None and r2.get("embedding") is not None:
                cross_inc_sims.append(float(np.dot(r1["embedding"], r2["embedding"])))

    def _avg(lst): return round(float(np.mean(lst)), 4) if lst else None

    same_avg = _avg(same_sims)
    cross_lang_avg = _avg(cross_lang_sims)
    cross_inc_avg = _avg(cross_inc_sims)
    gap = round(cross_lang_avg - cross_inc_avg, 4) if (cross_lang_avg and cross_inc_avg) else None

    return {
        "same_lingual_sim": same_avg,
        "cross_lingual_sim": cross_lang_avg,
        "cross_incident_sim": cross_inc_avg,
        "separation_gap": gap,
        "labse_active": _use_labse,
        "n_same_lingual_pairs": len(same_sims),
        "n_cross_lingual_pairs": len(cross_lang_sims),
        "n_cross_incident_pairs": len(cross_inc_sims),
    }


if __name__ == "__main__":
    import json, sys
    
    from pipeline.normalizer import normalize_batch, reset_counters
    from pipeline.gazetteer import resolve_batch

    with open("/home/claude/data/disaster_dataset_600.json") as f:
        dataset = json.load(f)

    reset_counters()
    normalized = normalize_batch(dataset["reports"])
    resolved = resolve_batch(normalized)
    embedded = embed_batch(resolved)

    print(f"\n{'='*60}")
    print(f"  EMBEDDING RESULTS ({'LaBSE' if _use_labse else 'CharNGram+Bridge'})")
    print(f"{'='*60}")
    print(f"  Reports: {len(embedded)}, Dim: {embedded[0]['embedding'].shape}")

    print(f"\n  SAME-INCIDENT SIMILARITY (INCIDENT_001 Kelaniya):")
    kel = [r for r in embedded if r.get('_ground_truth',{}).get('incident_id')=='INCIDENT_001']
    same_sims = []
    for i in range(min(5, len(kel))):
        for j in range(i+1, min(6, len(kel))):
            sim = float(np.dot(kel[i]['embedding'], kel[j]['embedding']))
            same_sims.append(sim)
            l1, l2 = kel[i]['_ground_truth']['language'], kel[j]['_ground_truth']['language']
            print(f"    {l1:20s} <-> {l2:20s}: {sim:.4f}")
    if same_sims:
        print(f"    Avg same-incident: {np.mean(same_sims):.4f}")

    print(f"\n  DIFFERENT-INCIDENT SIMILARITY:")
    diff = {}
    for r in embedded:
        inc = r.get('_ground_truth',{}).get('incident_id','')
        if inc not in diff and len(diff) < 5:
            diff[inc] = r
    items = list(diff.items())
    cross = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            sim = float(np.dot(items[i][1]['embedding'], items[j][1]['embedding']))
            cross.append(sim)
            print(f"    {items[i][0]}({items[i][1]['_ground_truth']['actual_location']:12s}) <-> "
                  f"{items[j][0]}({items[j][1]['_ground_truth']['actual_location']:12s}): {sim:.4f}")
    if cross:
        print(f"    Avg cross-incident: {np.mean(cross):.4f}")
    if same_sims and cross:
        gap = np.mean(same_sims) - np.mean(cross)
        print(f"\n  SEPARATION GAP: {gap:.4f} {'GOOD' if gap > 0.05 else 'WEAK'}")

    print(f"\n{'='*60}")
    print(f"  CROSS-LINGUAL EVALUATION (Sinhala ↔ Tamil)")
    print(f"{'='*60}")
    cl = evaluate_cross_lingual_pairs(embedded)
    print(f"  LaBSE active:            {cl['labse_active']}")
    print(f"  Same-lingual pairs:      {cl['n_same_lingual_pairs']}")
    print(f"  Cross-lingual pairs:     {cl['n_cross_lingual_pairs']}")
    print(f"  Same-lingual avg sim:    {cl['same_lingual_sim']}")
    print(f"  Cross-lingual avg sim:   {cl['cross_lingual_sim']}"
          f"  (Sinhala ↔ Tamil, same incident)")
    print(f"  Cross-incident avg sim:  {cl['cross_incident_sim']}"
          f"  (baseline noise)")
    gap = cl['separation_gap']
    if gap is not None:
        quality = 'EXCELLENT' if gap > 0.15 else ('GOOD' if gap > 0.10 else
                  ('MARGINAL — LaBSE not loaded?' if gap > 0.03 else 'POOR — use real LaBSE'))
        print(f"  Separation gap:          {gap:.4f}  [{quality}]")
    else:
        print(f"  Separation gap:          N/A (need Tamil reports in dataset)")
