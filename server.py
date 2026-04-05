"""
C4 Dashboard Server — Run: python server.py — Open: http://localhost:8000
"""
import json, sys, os, time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pipeline.normalizer import normalize_batch, reset_counters
from pipeline.gazetteer import resolve_batch
from pipeline.embedder import embed_batch
from pipeline.clustering import (
    IncidentClusterEngine, evaluate_clustering,
    combined_similarity, split_uir, merge_uirs, set_similarity_model,
)
from pipeline.feedback_logger import FeedbackLogger
from pipeline.similarity_model import LearnedSimilarityModel
from pipeline.ml_deduplication import MLDeduplicator, init_deduplicator
from pipeline.database import Database, init_db

# ── LOAD & PROCESS DATA BEFORE SERVER STARTS ──
print("=" * 55)
print("  C4 PIPELINE — LOADING DATA")
print("=" * 55)

DPATH = ROOT / "data" / "disaster_dataset_1000.json"
if not DPATH.exists():
    DPATH = ROOT / "data" / "disaster_dataset_800.json"  # fallback
if not DPATH.exists():
    print(f"  Dataset not found: {DPATH}")
    print(f"  Run: python generate_dataset.py")
    sys.exit(1)

with open(DPATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f"  Loaded {len(dataset['reports'])} reports ({dataset['metadata'].get('incident_types',[])})")

reset_counters()
t0 = time.time()
normalized = normalize_batch(dataset["reports"])
resolved = resolve_batch(normalized)
embedded = embed_batch(resolved)

FEEDBACK_LOGGER = FeedbackLogger(ROOT / "data")
SIMILARITY_MODEL = LearnedSimilarityModel(ROOT / "data")
print(f"  {SIMILARITY_MODEL.status_line()}")

# ML Deduplicator — train XGBoost on ground-truth pairs
ML_DEDUP = init_deduplicator(ROOT / "data")
if not ML_DEDUP.is_trained:
    print("  Training ML deduplicator on ground-truth pairs...")
    ML_DEDUP.train(embedded, method='auto')
else:
    print(f"  {ML_DEDUP.status_line()}")

# Inject XGBoost as the deduplication model
set_similarity_model(ML_DEDUP)
print("  [Dedup] XGBoost active for deduplication")

# Keep reference for retraining
EMBEDDED_REPORTS = embedded

# Database (optional — connect if DATABASE_URL is set)
DB = init_db()

ENGINE = IncidentClusterEngine()
for r in embedded:
    ENGINE.process_report(r)

active = [u for u in ENGINE.active_uirs if u["status"] == "active"]
ev = evaluate_clustering(ENGINE)
print(f"  Done in {time.time()-t0:.2f}s | UIRs: {len(active)} | Precision: {ev['precision']:.4f} | CritFMR: {ev['critical_false_merge_rate']:.4f}")

# Store to DB if connected
if DB and DB.connected:
    DB.store_all_uirs(ENGINE)

print("=" * 55)

# ── SERIALIZE ──
def safe_iso(v):
    return v.isoformat() if hasattr(v, "isoformat") else str(v) if v else None

def serialize_uir(uir):
    return {
        "uir_id": uir["uir_id"], "incident_type": uir["incident_type"],
        "location": {"display_name": uir["location"]["display_name"], "lat": uir.get("lat"),
                     "lng": uir.get("lng"), "resolved": uir["location"]["resolved"],
                     "source_strings": uir["location"]["source_strings"][:5]},
        "people_involved": uir["people_involved"], "urgency": uir["urgency"],
        "confidence": uir["confidence"], "source_count": uir["source_count"],
        "sources": [{"id":s["id"],"channel":s["channel"],"time":safe_iso(s["time"]),"confidence":s["confidence"]} for s in uir["sources"][:10]],
        "timeline": uir["timeline"][:10], "flags": uir["flags"],
        "linked_uirs": uir["linked_uirs"], "status": uir["status"],
        "created_at": safe_iso(uir["created_at"]), "last_updated": safe_iso(uir["last_updated"]),
        "operator_actions": uir.get("operator_actions", []),
    }

uo = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}
active.sort(key=lambda u: uo.get(u["urgency"],4))
ALL_UIRS = [serialize_uir(u) for u in active]

STATS = {"total_reports": len(dataset["reports"]), "active_uirs": len(active),
         "flagged_uirs": sum(1 for u in active if u["flags"]),
         "precision": ev["precision"], "recall": ev["recall"], "critical_fmr": ev["critical_false_merge_rate"],
         "flood_uirs": sum(1 for u in active if u["incident_type"]=="flood"),
         "fire_uirs": sum(1 for u in active if u["incident_type"]=="fire")}

# ── APP ──
app = FastAPI(title="C4 Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
clients = []

async def broadcast(msg):
    dead = []
    m = json.dumps(msg, default=str)
    for c in clients:
        try: await c.send_text(m)
        except: dead.append(c)
    for c in dead: clients.remove(c)

@app.websocket("/ws/dashboard")
async def ws_endpoint(ws: WebSocket):
    await ws.accept(); clients.append(ws)
    await ws.send_json({"type":"init","uirs":ALL_UIRS,"stats":STATS})
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        if ws in clients: clients.remove(ws)

@app.get("/api/uirs")
async def get_uirs(): return {"uirs": ALL_UIRS, "stats": STATS}

@app.get("/api/stats")
async def get_stats(): return STATS

def _retrain_if_ready():
    """After each operator correction, rebuild training pairs and retrain."""
    pairs = FEEDBACK_LOGGER.extract_training_pairs(ENGINE)
    # Rewrite examples file from all corrections (idempotent)
    ex_path = ROOT / "data" / "similarity_examples.jsonl"
    ex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ex_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    SIMILARITY_MODEL._n_examples = len(pairs)
    SIMILARITY_MODEL.retrain()


@app.post("/api/uirs/{uir_id}/action")
async def action(uir_id: str, act: dict):
    target = next((u for u in ENGINE.active_uirs if u["uir_id"] == uir_id), None)
    if not target:
        raise HTTPException(404)

    at = act.get("action_type", "")
    op_id = act.get("operator_id", "op1")
    note = act.get("note", "")
    log = {"action": at, "operator_id": op_id,
           "timestamp": datetime.now(timezone.utc).isoformat(), "note": note}

    broadcast_extra = []  # additional UIRs whose serialized form changed

    if at == "confirm":
        target["status"] = "confirmed"
        if "operator_review_required" in target["flags"]:
            target["flags"].remove("operator_review_required")
        target["confidence"] = min(1.0, target["confidence"] * 1.1)

    elif at == "dismiss":
        target["status"] = "dismissed"

    elif at == "split":
        # act must include "split_source_ids": ["call_0001", ...]
        split_ids = act.get("split_source_ids", [])
        if not split_ids:
            raise HTTPException(400, "split requires split_source_ids list")
        kept_ids = [s["id"] for s in target["sources"] if s["id"] not in split_ids]
        new_uir = split_uir(target, split_ids)
        if new_uir is None:
            raise HTTPException(400, "split would empty one side — aborted")
        ENGINE.active_uirs.append(new_uir)
        ALL_UIRS.append(serialize_uir(new_uir))
        broadcast_extra.append(new_uir)
        # Log correction → gold-label training pair
        FEEDBACK_LOGGER.log_split(uir_id, kept_ids, split_ids, op_id, note)
        _retrain_if_ready()

    elif at == "merge":
        # act must include "merge_with_uir_id": "UIR-..."
        other_id = act.get("merge_with_uir_id", "")
        other = next((u for u in ENGINE.active_uirs if u["uir_id"] == other_id), None)
        if not other:
            raise HTTPException(404, f"merge_with_uir_id {other_id!r} not found")
        # Capture similarity scores before merging (needed for training)
        sim = combined_similarity(
            {"embedding": other["centroid_embedding"],
             "lat": other.get("lat"), "lng": other.get("lng"),
             "timestamp": other.get("last_updated"), "receive_time": other.get("created_at")},
            target,
        )
        FEEDBACK_LOGGER.log_merge(uir_id, other_id,
                                  {"semantic": sim["semantic"],
                                   "geographic": sim["geographic"],
                                   "temporal": sim["temporal"]},
                                  op_id, note)
        merge_uirs(target, other)
        other["status"] = "dismissed"
        broadcast_extra.append(other)
        _retrain_if_ready()

    elif at == "correct":
        # act must include "field" and "value"
        field = act.get("field")
        new_value = act.get("value")
        if not field:
            raise HTTPException(400, "correct requires 'field' and 'value'")
        old_value = None
        if field == "urgency":
            old_value = target.get("urgency")
            target["urgency"] = new_value
        elif field == "incident_type":
            old_value = target.get("incident_type")
            target["incident_type"] = new_value
        elif field == "people_involved":
            old_value = target.get("people_involved", {}).get("value")
            target["people_involved"]["value"] = new_value
            target["people_involved"]["resolved"] = True
            if "people_count_conflict" in target["flags"]:
                target["flags"].remove("people_count_conflict")
        elif field == "location":
            old_value = target["location"].get("display_name")
            target["location"]["display_name"] = new_value
        FEEDBACK_LOGGER.log_correct(uir_id, field, old_value, new_value, op_id, note)

    target.setdefault("operator_actions", []).append(log)

    # Refresh serialized form
    s = serialize_uir(target)
    for i, u in enumerate(ALL_UIRS):
        if u["uir_id"] == uir_id:
            ALL_UIRS[i] = s
            break

    await broadcast({"type": "uir_update", "uir": s, "stats": STATS})
    for extra in broadcast_extra:
        es = serialize_uir(extra)
        for i, u in enumerate(ALL_UIRS):
            if u["uir_id"] == extra["uir_id"]:
                ALL_UIRS[i] = es
                break
        await broadcast({"type": "uir_update", "uir": es, "stats": STATS})

    return {"status": "ok", "action": log}


@app.get("/api/similarity_model")
async def get_model_status():
    return {
        "trained": SIMILARITY_MODEL.is_trained,
        "n_examples": SIMILARITY_MODEL.n_examples,
        "weights": SIMILARITY_MODEL.weights,
        "corrections": len(FEEDBACK_LOGGER.load_corrections()),
    }

@app.get("/api/ml/status")
async def ml_status():
    return {
        "ml_dedup": {
            "trained": ML_DEDUP.is_trained,
            "finetuned": ML_DEDUP.is_finetuned,
            "metrics": ML_DEDUP.metrics,
        },
        "similarity_model": {
            "trained": SIMILARITY_MODEL.is_trained,
            "weights": SIMILARITY_MODEL.weights,
            "n_examples": SIMILARITY_MODEL.n_examples,
        },
        "database": {
            "connected": DB.connected if DB else False,
            "stats": DB.get_stats() if DB and DB.connected else {},
        },
        "corrections": len(FEEDBACK_LOGGER.load_corrections()),
    }

@app.post("/api/ml/train")
async def ml_train(body: dict = None):
    """Re-train ML deduplicator. Pass {"finetune": true} to also fine-tune LaBSE."""
    body = body or {}
    metrics = ML_DEDUP.train(EMBEDDED_REPORTS, method='auto')
    if body.get("finetune"):
        ML_DEDUP.fine_tune(EMBEDDED_REPORTS, epochs=body.get("epochs", 3))
    return {"status": "ok", "metrics": metrics}

@app.get("/api/ml/cross_lingual")
async def ml_cross_lingual():
    """Cross-lingual AUC breakdown showing per-language-pair performance."""
    if not ML_DEDUP.is_trained:
        return {"error": "model not trained"}
    return ML_DEDUP.evaluate_cross_lingual(EMBEDDED_REPORTS)

@app.get("/api/db/stats")
async def db_stats():
    if not DB or not DB.connected:
        return {"connected": False, "message": "Set DATABASE_URL to enable PostgreSQL"}
    return {"connected": True, **DB.get_stats()}

@app.get("/api/db/nearby")
async def db_nearby(lat: float, lng: float, radius_km: float = 5.0):
    if not DB or not DB.connected:
        return {"error": "database not connected"}
    rows = DB.find_nearby_uirs(lat, lng, radius_km)
    return {"results": [dict(r) for r in rows], "count": len(rows)}

@app.get("/")
async def dashboard():
    p = ROOT / "dashboard.html"
    return HTMLResponse(p.read_text("utf-8")) if p.exists() else HTMLResponse("<h1>dashboard.html not found</h1>")

if __name__ == "__main__":
    print(f"\n  Dashboard: http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
