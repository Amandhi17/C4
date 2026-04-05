# C4 Setup & Configuration Guide

Step-by-step instructions to configure and run the C4 AI-Based Emergency Triage System.

---

## Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **Git** (to clone the repository)
- **PostgreSQL 14+ with PostGIS 3.3+** (optional — system works without it)
- **~500 MB disk space** (312 MB for GeoJSON + dependencies + data)

---

## Step 1: Clone the Repository

```bash
git clone <your-repo-url> C4-main
cd C4-main
```

Or if you already have the folder, just navigate to it:

```bash
cd C4-main
```

---

## Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

- **Windows (CMD):**
  ```cmd
  venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

---

## Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | LaBSE embeddings (Sinhala + Tamil cross-lingual) |
| `rapidfuzz` | Fuzzy string matching for location resolution |
| `fastapi` | REST API + WebSocket server |
| `uvicorn` | ASGI server to run FastAPI |
| `websockets` | WebSocket protocol support |
| `numpy` | Numerical operations, embedding math |
| `scikit-learn` | DBSCAN clustering, LogisticRegression |
| `requests` | Nominatim API calls for geocoding fallback |
| `psycopg2-binary` | PostgreSQL driver (only needed if using DB) |
| `xgboost` | ML-based deduplication classifier |

**If you encounter errors:**

```bash
# If scikit-learn fails to install
pip install scikit-learn --no-cache-dir

# If psycopg2-binary fails (and you don't need PostgreSQL)
# You can skip it — the system works without a database
pip install sentence-transformers rapidfuzz fastapi uvicorn websockets numpy scikit-learn requests xgboost
```

---

## Step 4: Verify the GN Division Data

The gazetteer needs GN (Grama Niladhari) division centroids for sub-city location precision.

**Check if `data/gn_divisions.json` already exists:**

```bash
# Windows
dir data\gn_divisions.json

# Linux/macOS
ls -la data/gn_divisions.json
```

If it exists (~1.2 MB, ~4,258 divisions), skip to Step 5.

**If it does NOT exist**, extract it from the GeoJSON:

```bash
python pipeline/gn_extractor.py
```

This reads `lka_admin4.geojson` (312 MB, 14,043 GN divisions from HDX) and extracts
centroids for the 5 target districts: Colombo, Gampaha, Kalutara, Ratnapura, Kegalle.

Expected output:
```
Extracting GN divisions from lka_admin4.geojson...
  Target districts: Colombo, Gampaha, Kalutara, Ratnapura, Kegalle
  Extracted 4,258 GN divisions
  Saved to data/gn_divisions.json (1,223 KB)
```

**If `lka_admin4.geojson` is missing:**

Download it from the Humanitarian Data Exchange (HDX):
- URL: https://data.humdata.org/dataset/cod-ab-lka (search for "Sri Lanka Admin Level 4")
- Download the GeoJSON file and place it in the project root as `lka_admin4.geojson`

---

## Step 5: Generate the Dataset

```bash
python generate_dataset.py
```

This creates `data/disaster_dataset_800.json` with 800 reports:
- 400 flood + 400 fire
- ~55% Sinhala, ~30% Romanized Sinhala, ~15% Tamil
- GN-level coordinate precision
- Scenario types: mass_report, adjacent_critical, geographically_near, third_party, conflicting_counts, long_gap, standard, noise

Expected output:
```
Generating C4 disaster dataset (800 reports)...
  Loaded 4,258 GN divisions for coordinate precision
  Flood scenarios: 20 incidents -> 400 reports
  Fire scenarios: 20 incidents -> 400 reports
  Saved: data/disaster_dataset_800.json
  Languages: {'sinhala': ~440, 'romanized': ~240, 'tamil': ~120}
```

**Note:** If `data/gn_divisions.json` is not present, the generator falls back to
town-level coordinates (less precise but still functional).

---

## Step 6: Run the Server

```bash
python server.py
```

On first run, the pipeline will:
1. Load 800 reports
2. Normalize fields and parse timestamps
3. Resolve locations through the 6-stage gazetteer
4. Generate embeddings (LaBSE or char n-gram fallback)
5. Train the XGBoost ML deduplicator on ground-truth pairs
6. Cluster reports into UIRs using DBSCAN + safety gate

Expected startup output:
```
=======================================================
  C4 PIPELINE -- LOADING DATA
=======================================================
  Loaded 800 reports (['flood', 'fire'])
  Normalizer: 800 valid, 0 rejected
  Gazetteer: 97.1% resolved (777/800) | exact: 589 | fuzzy: 112 | landmark: 56 | hierarchical: 20 | unresolved: 23
  Embedder: 800 embedded (LaBSE: false, fallback: char_ngram_keyword)
  Similarity model: untrained (0 examples, using fixed weights)
  Training ML deduplicator on ground-truth pairs...
  ML Dedup: XGBoost | F1=0.942 | AUC=0.996 | 7172 pairs
  Done in 4.82s | UIRs: 38 | Precision: 0.9500 | CritFMR: 0.0000
=======================================================

  Dashboard: http://localhost:8000
```

**Open your browser to:** http://localhost:8000

---

## Step 7 (Optional): Enable LaBSE for Better Cross-Lingual Performance

By default, the system uses a char n-gram + keyword bridge fallback for embeddings.
For production-quality cross-lingual deduplication, download LaBSE:

```python
# Run this once — downloads ~1.8 GB
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/LaBSE")
model.save("./labse_model")
```

Or simply let the embedder auto-download on first use. The system checks for
`sentence-transformers/LaBSE` and downloads it if internet is available.

With LaBSE enabled:
- Sinhala-Tamil cross-lingual similarity improves significantly
- Semantic similarity becomes the dominant feature (instead of geographic)
- Optional fine-tuning available via `/api/ml/train` with `{"finetune": true}`

---

## Step 8 (Optional): Configure PostgreSQL + PostGIS

The system works fully in-memory without a database. To enable persistent storage
and geographic radius queries:

### 8a. Install PostgreSQL + PostGIS

**Windows:**
1. Download PostgreSQL from https://www.postgresql.org/download/windows/
2. Run the installer (include Stack Builder)
3. Open Stack Builder -> select your PostgreSQL installation -> Spatial Extensions -> PostGIS
4. Install PostGIS

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib postgis
```

**macOS (Homebrew):**
```bash
brew install postgresql@14 postgis
brew services start postgresql@14
```

### 8b. Create the Database

```bash
# Connect to PostgreSQL
psql -U postgres

# In the psql shell:
CREATE DATABASE c4_triage;
\c c4_triage
CREATE EXTENSION postgis;
\q
```

### 8c. Set the Environment Variable

**Windows (CMD):**
```cmd
set DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/c4_triage
```

**Windows (PowerShell):**
```powershell
$env:DATABASE_URL = "postgresql://postgres:yourpassword@localhost:5432/c4_triage"
```

**Linux/macOS:**
```bash
export DATABASE_URL="postgresql://postgres:yourpassword@localhost:5432/c4_triage"
```

### 8d. Run the Server

```bash
python server.py
```

The server will automatically:
- Connect to PostgreSQL
- Create all 6 tables (uirs, source_reports, timeline_entries, operator_actions, nominatim_cache, training_pairs)
- Create the PostGIS GIST index for geographic queries
- Store all UIRs after processing

You'll see in the output:
```
  Database: connected (PostgreSQL + PostGIS)
```

### 8e. Verify Database

```bash
# Check via API
curl http://localhost:8000/api/db/stats

# Geographic radius query (5 km around Colombo)
curl "http://localhost:8000/api/db/nearby?lat=6.9271&lng=79.8612&radius_km=5"
```

---

## Step 9 (Optional): Retrain ML Models

### Re-train XGBoost deduplicator:
```bash
curl -X POST http://localhost:8000/api/ml/train
```

### Re-train with LaBSE fine-tuning (requires LaBSE):
```bash
curl -X POST http://localhost:8000/api/ml/train \
  -H "Content-Type: application/json" \
  -d '{"finetune": true, "epochs": 3}'
```

### Check model status:
```bash
curl http://localhost:8000/api/ml/status
```

### View cross-lingual performance:
```bash
curl http://localhost:8000/api/ml/cross_lingual
```

---

## Operator Actions via API

After the dashboard is running, operators can perform corrections that
automatically generate gold-label training data:

### Confirm a UIR:
```bash
curl -X POST http://localhost:8000/api/uirs/UIR-20250517-0001/action \
  -H "Content-Type: application/json" \
  -d '{"action_type": "confirm", "operator_id": "op1"}'
```

### Split a UIR (separate wrongly merged reports):
```bash
curl -X POST http://localhost:8000/api/uirs/UIR-20250517-0001/action \
  -H "Content-Type: application/json" \
  -d '{"action_type": "split", "split_source_ids": ["call_0001", "sms_0002"], "operator_id": "op1"}'
```

### Merge two UIRs (combine duplicates):
```bash
curl -X POST http://localhost:8000/api/uirs/UIR-20250517-0001/action \
  -H "Content-Type: application/json" \
  -d '{"action_type": "merge", "merge_with_uir_id": "UIR-20250517-0005", "operator_id": "op1"}'
```

### Correct a field:
```bash
curl -X POST http://localhost:8000/api/uirs/UIR-20250517-0001/action \
  -H "Content-Type: application/json" \
  -d '{"action_type": "correct", "field": "urgency", "value": "CRITICAL", "operator_id": "op1"}'
```

After 20+ split/merge corrections, the similarity model automatically retrains
with learned weights replacing the fixed 0.5/0.3/0.2 formula.

---

## Troubleshooting

### "Dataset not found" error
```
Run: python generate_dataset.py
```

### "ModuleNotFoundError: No module named 'rapidfuzz'"
```bash
pip install rapidfuzz
```

### "ModuleNotFoundError: No module named 'xgboost'"
```bash
pip install xgboost
```

### psycopg2 installation fails
If you don't need PostgreSQL, just skip it. The system runs fully in-memory.
If you do need it:
```bash
# Try the binary version first
pip install psycopg2-binary

# On Linux, if that fails:
sudo apt install libpq-dev python3-dev
pip install psycopg2
```

### UnicodeEncodeError on Windows
Set your console to UTF-8:
```cmd
chcp 65001
```
Or set the environment variable:
```cmd
set PYTHONIOENCODING=utf-8
```

### Server starts but dashboard is blank
Make sure `dashboard.html` exists in the project root directory.

### Low location resolution rate
Ensure `data/gn_divisions.json` exists and was extracted properly.
Re-run: `python pipeline/gn_extractor.py`

### ML Deduplicator shows "LogisticRegression fallback"
This means neither XGBoost nor GradientBoosting could be imported.
Install XGBoost: `pip install xgboost`

---

## Quick Start (TL;DR)

```bash
cd C4-main
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
python generate_dataset.py     # Creates 800 reports
python server.py               # Starts at http://localhost:8000
```

That's it. Open http://localhost:8000 in your browser.
