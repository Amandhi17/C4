# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# C4 — AI-Based Multi-Source Incident Fusion & Real-Time Dashboard

Component 4 of R26-DS-011: AI-Based Emergency Triage System for Sri Lankan Disasters.

## What This Does

Fuses 800 simulated disaster reports (flood + fire, Sinhala + Tamil) into deduplicated
Unified Incident Records (UIRs) using ML-based deduplication, GN-division-level geocoding,
and a real-time operator dashboard.

## Architecture

```
C1 (Voice) --> JSON --+
                      +--> Normalize --> Gazetteer --> Embed --> ML Dedup --> UIR --> Dashboard
C3 (SMS)   --> JSON --+                    |                      |
                                     4,258 GN divs          XGBoost classifier
                                     + Nominatim             trained on 7k pairs
```

## Development Commands

```bash
# Full setup (first time)
python -m venv venv
source venv/Scripts/activate   # bash on Windows
# venv\Scripts\activate        # Windows CMD / PowerShell
pip install -r requirements.txt
python pipeline/gn_extractor.py  # extract GN centroids (if data/gn_divisions.json missing)
python generate_dataset.py       # create data/disaster_dataset_1000.json
python server.py                 # starts at http://localhost:8000

# There is no test suite. Verify behaviour by hitting the API endpoints above.

# Re-run individual pipeline stages (import and call directly — no standalone runners)
python -c "from pipeline.normalizer import normalize_batch; ..."

# API testing
curl http://localhost:8000/api/uirs
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/ml/status
curl -X POST http://localhost:8000/api/ml/train

# Retrain with LaBSE fine-tuning (requires LaBSE ~1.8 GB download)
curl -X POST http://localhost:8000/api/ml/train \
  -H "Content-Type: application/json" \
  -d '{"finetune": true, "epochs": 3}'

# Geographic radius query (PostGIS required)
curl "http://localhost:8000/api/db/nearby?lat=6.9271&lng=79.8612&radius_km=5"

# Windows UTF-8 fix (if console shows garbled Sinhala/Tamil)
chcp 65001
set PYTHONIOENCODING=utf-8
```

## Pipeline Stages

| Stage | File | What It Does |
|-------|------|-------------|
| 1-2 | `pipeline/normalizer.py` | Field aliasing, validation, timestamp parsing |
| 3 | `pipeline/gazetteer.py` | 6-stage location resolution (exact/fuzzy/landmark/hierarchical/Nominatim/unresolved). 4,258 GN divisions from `lka_admin4.geojson` |
| 4 | `pipeline/embedder.py` | LaBSE 768-dim cross-lingual embeddings (Sinhala+Tamil). Fallback: char n-gram + keyword bridge |
| 5-6 | `pipeline/clustering.py` | Urgency-sensitive DBSCAN, 5-condition safety gate, conflict detection, UIR management |
| ML | `pipeline/ml_deduplication.py` | XGBoost pairwise classifier (F1=0.94, AUC=0.996). Optional LaBSE fine-tuning |
| DB | `pipeline/database.py` | PostgreSQL + PostGIS (optional). ST_DWithin geo queries, JSONB UIR storage |
| Server | `server.py` | FastAPI + WebSocket dashboard. Operator split/merge/correct actions |

## Cross-File Architecture Details

**Startup sequence** (`server.py`): On launch, the server synchronously runs the full pipeline — normalize → resolve → embed → cluster — before accepting requests. This is intentional: the dashboard always shows a fully processed dataset, not a partial one. Startup takes ~5 seconds (no LaBSE) to ~60 seconds (LaBSE). If no `data/ml_deduplicator.pkl` exists, the XGBoost model auto-trains on startup (adds ~30 s). If `data/disaster_dataset_800.json` is missing, the server falls back to `disaster_dataset_600.json`; if neither exists it exits with instructions to run `generate_dataset.py`.

**Similarity function** (`clustering.py` + `similarity_model.py`): The default combined similarity is `0.5*semantic + 0.3*geo + 0.2*temporal`. Once ≥20 operator split/merge corrections accumulate, `LearnedSimilarityModel` replaces these fixed weights with logistic-regression-learned weights. `set_similarity_model()` in `clustering.py` is the injection point.

**DBSCAN epsilon is urgency-sensitive** (`clustering.py`): CRITICAL reports use ε=0.15 (must be 85%+ similar to merge), LOW uses ε=0.40. Two safety gates are defined: `CRITICAL_GATE` (min_semantic=0.88, max_distance=0.8 km, max_time_gap=15 min, ≥2 corroborating sources, type match required) and `NON_CRITICAL_GATE` (min_semantic=0.70, max_distance=3.0 km, max_time_gap=45 min, ≥1 source, no type match required).

**ML deduplicator vs similarity model** (`ml_deduplication.py` vs `similarity_model.py`): These are two separate learners. `MLDeduplicator` (XGBoost) is trained on ground-truth pairs generated from the dataset's scenario structure. `LearnedSimilarityModel` (LogisticRegression) is trained on operator corrections at runtime. The ML deduplicator's `ml_similarity()` is a drop-in replacement for `combined_similarity()`.

**Feedback loop**: Operator split/merge actions → `FeedbackLogger` writes to `data/operator_corrections.jsonl` and `data/similarity_examples.jsonl` → triggers `LearnedSimilarityModel.retrain()`. This is the online learning loop. Gold-label pairs are also written to `data/training_pairs` for the XGBoost retraining.

**LaBSE vs fallback**: `embedder.py` tries to load `SentenceTransformer('LaBSE')` at import time. On failure, it falls back to multilingual char n-grams (3,4,5-gram) plus a hardcoded Sinhala/Tamil→English keyword bridge. The fallback is offline-capable and produces 768-dim vectors matching the same interface.

**Database is optional**: `pipeline/database.py` wraps all DB calls. `init_db()` returns `None` if `DATABASE_URL` env var is not set — all callers in `server.py` guard with `if DB and DB.connected`.

## Key Files

```
C4-main/
  generate_dataset.py          # 800 reports (Sinhala+Tamil, GN-precision coords)
  server.py                    # FastAPI server + all endpoints
  dashboard.html               # React + Leaflet real-time UI
  lka_admin4.geojson           # 14,043 GN division boundaries (312 MB, from HDX)
  requirements.txt             # Python dependencies
  pipeline/
    normalizer.py              # Stage 1-2
    gazetteer.py               # Stage 3 — 6-stage resolution chain
    embedder.py                # Stage 4 — LaBSE / n-gram fallback
    clustering.py              # Stage 5-6 — DBSCAN + conflict detection
    ml_deduplication.py        # ML deduplication (XGBoost + optional fine-tuning)
    similarity_model.py        # Lightweight LR model trained on operator corrections
    feedback_logger.py         # Operator correction logging (gold-label training data)
    database.py                # PostgreSQL + PostGIS integration
    gn_extractor.py            # Extract GN centroids from GeoJSON
  data/
    disaster_dataset_800.json  # Generated dataset
    gn_divisions.json          # 4,258 GN centroids (extracted from GeoJSON)
    nominatim_cache.json       # Cached geocoding results
    ml_deduplicator.pkl        # Trained ML model
    similarity_model.pkl       # Trained operator-correction model
    similarity_examples.jsonl  # Training pairs from corrections
    operator_corrections.jsonl # Operator feedback log
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard HTML |
| GET | `/api/uirs` | All UIRs + stats |
| GET | `/api/stats` | Summary statistics |
| POST | `/api/uirs/{id}/action` | Operator actions (confirm/dismiss/split/merge/correct) |
| GET | `/api/ml/status` | ML model + DB status |
| POST | `/api/ml/train` | Re-train ML deduplicator |
| GET | `/api/ml/cross_lingual` | Cross-lingual AUC breakdown |
| GET | `/api/similarity_model` | Operator-correction model status |
| GET | `/api/db/stats` | Database statistics |
| GET | `/api/db/nearby?lat=X&lng=Y&radius_km=Z` | Geographic radius query |
| WS | `/ws/dashboard` | Real-time UIR updates |

## Operator Actions (POST /api/uirs/{id}/action)

```json
{"action_type": "confirm", "operator_id": "op1"}
{"action_type": "dismiss", "operator_id": "op1"}
{"action_type": "split", "split_source_ids": ["call_0001","sms_0002"], "operator_id": "op1"}
{"action_type": "merge", "merge_with_uir_id": "UIR-20250517-0005", "operator_id": "op1"}
{"action_type": "correct", "field": "urgency", "value": "CRITICAL", "operator_id": "op1"}
```

Split/merge actions automatically log gold-label training pairs and retrain the similarity model.

## Conventions

- Python 3.10+
- All times UTC (timezone-aware datetimes)
- Embeddings: 768-dim float32, L2-normalized
- Urgency levels: CRITICAL > HIGH > MEDIUM > LOW
- UIR IDs: `UIR-YYYYMMDD-NNNN`
- Source IDs: `call_NNNN` (voice) or `sms_NNNN`

## Optional: PostgreSQL + PostGIS

Set `DATABASE_URL=postgresql://user:pass@localhost:5432/c4_triage` before starting the server.
The DB is auto-detected at startup — the system runs fully in-memory without it.

```bash
# Create DB (psql)
CREATE DATABASE c4_triage;
\c c4_triage
CREATE EXTENSION postgis;
```

Tables created automatically: `uirs`, `source_reports`, `timeline_entries`, `operator_actions`, `nominatim_cache`, `training_pairs`.
