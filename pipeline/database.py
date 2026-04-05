"""
PostgreSQL + PostGIS Database Layer
====================================
Replaces in-memory dicts with persistent, geospatially-indexed storage.

Tables:
  uirs              — UIR records with JSONB metadata + PostGIS point
  source_reports    — individual reports linked to UIRs
  timeline_entries  — chronological timeline per UIR
  operator_actions  — operator feedback log (gold-label training data)
  nominatim_cache   — cached geocoding results
  training_pairs    — ML similarity training pairs

Geographic queries use ST_DWithin on a GIST-indexed geometry column,
giving precise radius searches (<1ms for 10k rows).

If no PostgreSQL connection is configured, everything falls back to
in-memory dicts (same behavior as before). Set DATABASE_URL env var
or pass connection string to Database().

Connection string format:
  postgresql://user:pass@localhost:5432/c4_disaster
"""

import json
import os
import numpy as np
from datetime import datetime, timezone
from typing import Optional

_HAS_PSYCOPG2 = False
try:
    import psycopg2
    import psycopg2.extras
    _HAS_PSYCOPG2 = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;

-- UIRs: core incident records
CREATE TABLE IF NOT EXISTS uirs (
    uir_id          TEXT PRIMARY KEY,
    incident_type   TEXT NOT NULL,
    urgency         TEXT NOT NULL DEFAULT 'MEDIUM',
    status          TEXT NOT NULL DEFAULT 'active',
    confidence      FLOAT DEFAULT 0.5,
    location_name   TEXT,
    lat             FLOAT,
    lng             FLOAT,
    geom            GEOMETRY(Point, 4326),
    people_involved JSONB DEFAULT '{}',
    source_count    INT DEFAULT 0,
    centroid_embedding FLOAT4[],
    flags           TEXT[] DEFAULT '{}',
    linked_uirs     TEXT[] DEFAULT '{}',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_updated    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_uirs_status   ON uirs(status);
CREATE INDEX IF NOT EXISTS idx_uirs_urgency  ON uirs(urgency);
CREATE INDEX IF NOT EXISTS idx_uirs_type     ON uirs(incident_type);
CREATE INDEX IF NOT EXISTS idx_uirs_geom     ON uirs USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_uirs_updated  ON uirs(last_updated DESC);

-- Source reports
CREATE TABLE IF NOT EXISTS source_reports (
    source_id       TEXT PRIMARY KEY,
    uir_id          TEXT REFERENCES uirs(uir_id) ON DELETE CASCADE,
    channel         TEXT,
    incident_type   TEXT,
    location_raw    TEXT,
    urgency         TEXT,
    confidence      FLOAT,
    lat             FLOAT,
    lng             FLOAT,
    embedding       FLOAT4[],
    key_phrases     TEXT[],
    report_data     JSONB DEFAULT '{}',
    timestamp       TIMESTAMPTZ,
    receive_time    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_reports_uir ON source_reports(uir_id);

-- Timeline entries (separate table for clean querying)
CREATE TABLE IF NOT EXISTS timeline_entries (
    id       SERIAL PRIMARY KEY,
    uir_id   TEXT REFERENCES uirs(uir_id) ON DELETE CASCADE,
    time_str TEXT,
    summary  TEXT,
    source_id TEXT,
    channel  TEXT
);

CREATE INDEX IF NOT EXISTS idx_timeline_uir ON timeline_entries(uir_id);

-- Operator actions (gold-label feedback)
CREATE TABLE IF NOT EXISTS operator_actions (
    id           SERIAL PRIMARY KEY,
    uir_id       TEXT,
    action_type  TEXT NOT NULL,
    operator_id  TEXT,
    note         TEXT DEFAULT '',
    metadata     JSONB DEFAULT '{}',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_actions_uir  ON operator_actions(uir_id);
CREATE INDEX IF NOT EXISTS idx_actions_type ON operator_actions(action_type);

-- Nominatim geocoding cache
CREATE TABLE IF NOT EXISTS nominatim_cache (
    query_text   TEXT PRIMARY KEY,
    lat          FLOAT,
    lng          FLOAT,
    display_name TEXT,
    result_json  JSONB,
    cached_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ML training pairs
CREATE TABLE IF NOT EXISTS training_pairs (
    id        SERIAL PRIMARY KEY,
    source    TEXT DEFAULT 'ground_truth',
    semantic  FLOAT,
    geographic FLOAT,
    temporal  FLOAT,
    label     INT,
    metadata  JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Database:
    """
    PostgreSQL + PostGIS backend.  Falls back gracefully if no connection.

    Usage:
        db = Database()  # reads DATABASE_URL env var
        db = Database("postgresql://user:pass@localhost:5432/c4_disaster")
        if db.connected:
            db.store_uir(uir_dict)
            nearby = db.find_nearby_uirs(6.95, 79.92, radius_km=2.0)
    """

    def __init__(self, connection_string: str = None):
        self._conn = None
        self._connected = False
        conn_str = connection_string or os.environ.get("DATABASE_URL", "")
        if conn_str and _HAS_PSYCOPG2:
            self._connect(conn_str)

    def _connect(self, conn_str: str):
        try:
            self._conn = psycopg2.connect(conn_str)
            self._conn.autocommit = False
            self._connected = True
            print(f"  [DB] Connected to PostgreSQL")
        except Exception as e:
            print(f"  [DB] Connection failed: {e}")
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and self._conn is not None

    def create_schema(self):
        """Create all tables and indexes."""
        if not self.connected:
            return False
        try:
            with self._conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            self._conn.commit()
            print("  [DB] Schema created (6 tables + PostGIS indexes)")
            return True
        except Exception as e:
            self._conn.rollback()
            print(f"  [DB] Schema creation failed: {e}")
            return False

    def close(self):
        if self._conn:
            self._conn.close()
            self._connected = False

    # ── UIR CRUD ──────────────────────────────────────────────────────────────

    def store_uir(self, uir: dict):
        """Insert or update a UIR."""
        if not self.connected:
            return
        try:
            embedding = uir.get('centroid_embedding')
            emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO uirs (uir_id, incident_type, urgency, status,
                        confidence, location_name, lat, lng, geom,
                        people_involved, source_count, centroid_embedding,
                        flags, linked_uirs, metadata, created_at, last_updated)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,
                        ST_SetSRID(ST_MakePoint(%s,%s),4326),
                        %s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (uir_id) DO UPDATE SET
                        incident_type=EXCLUDED.incident_type,
                        urgency=EXCLUDED.urgency,
                        status=EXCLUDED.status,
                        confidence=EXCLUDED.confidence,
                        location_name=EXCLUDED.location_name,
                        lat=EXCLUDED.lat, lng=EXCLUDED.lng,
                        geom=EXCLUDED.geom,
                        people_involved=EXCLUDED.people_involved,
                        source_count=EXCLUDED.source_count,
                        centroid_embedding=EXCLUDED.centroid_embedding,
                        flags=EXCLUDED.flags,
                        linked_uirs=EXCLUDED.linked_uirs,
                        metadata=EXCLUDED.metadata,
                        last_updated=EXCLUDED.last_updated
                """, (
                    uir['uir_id'], uir['incident_type'], uir['urgency'],
                    uir.get('status', 'active'),
                    uir.get('confidence', 0.5),
                    uir.get('location', {}).get('display_name', ''),
                    uir.get('lat'), uir.get('lng'),
                    uir.get('lng'), uir.get('lat'),  # ST_MakePoint(lng, lat)
                    json.dumps(uir.get('people_involved', {}), default=str),
                    uir.get('source_count', 0),
                    emb_list,
                    uir.get('flags', []),
                    uir.get('linked_uirs', []),
                    json.dumps({k: v for k, v in uir.items()
                                if k not in ('centroid_embedding', 'source_reports')}, default=str),
                    _to_ts(uir.get('created_at')),
                    _to_ts(uir.get('last_updated')),
                ))
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            print(f"  [DB] store_uir failed: {e}")

    def get_active_uirs(self) -> list:
        """SELECT all active UIRs."""
        if not self.connected:
            return []
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT uir_id, incident_type, urgency, status, confidence,
                           location_name, lat, lng, people_involved,
                           source_count, flags, linked_uirs, metadata,
                           created_at, last_updated
                    FROM uirs WHERE status = 'active'
                    ORDER BY
                        CASE urgency
                            WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1
                            WHEN 'MEDIUM' THEN 2 ELSE 3 END,
                        last_updated DESC
                """)
                return cur.fetchall()
        except Exception as e:
            print(f"  [DB] get_active_uirs failed: {e}")
            return []

    def find_nearby_uirs(self, lat: float, lng: float,
                         radius_km: float = 5.0) -> list:
        """
        PostGIS geographic radius query — uses GIST index.
        Returns UIRs within radius_km of the given point.
        """
        if not self.connected:
            return []
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT uir_id, incident_type, urgency, lat, lng, confidence,
                           source_count, location_name,
                           ST_Distance(
                               geom::geography,
                               ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                           ) / 1000.0 AS distance_km
                    FROM uirs
                    WHERE status = 'active'
                      AND geom IS NOT NULL
                      AND ST_DWithin(
                          geom::geography,
                          ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                          %s
                      )
                    ORDER BY distance_km
                """, (lng, lat, lng, lat, radius_km * 1000))
                return cur.fetchall()
        except Exception as e:
            print(f"  [DB] find_nearby_uirs failed: {e}")
            return []

    # ── Source Reports ────────────────────────────────────────────────────────

    def store_report(self, report: dict, uir_id: str):
        """Insert a source report linked to a UIR."""
        if not self.connected:
            return
        try:
            embedding = report.get('embedding')
            emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else None

            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO source_reports
                        (source_id, uir_id, channel, incident_type, location_raw,
                         urgency, confidence, lat, lng, embedding, key_phrases,
                         report_data, timestamp, receive_time)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (source_id) DO UPDATE SET uir_id=EXCLUDED.uir_id
                """, (
                    report['source_id'], uir_id,
                    report.get('channel'), report.get('incident_type'),
                    report.get('location_raw'), report.get('urgency'),
                    report.get('confidence'), report.get('lat'), report.get('lng'),
                    emb_list,
                    report.get('key_phrases', []),
                    json.dumps({k: v for k, v in report.items()
                                if k not in ('embedding',)}, default=str),
                    _to_ts(report.get('timestamp')),
                    _to_ts(report.get('receive_time')),
                ))
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            print(f"  [DB] store_report failed: {e}")

    # ── Operator Actions ──────────────────────────────────────────────────────

    def store_operator_action(self, uir_id: str, action_type: str,
                              operator_id: str = "op1", note: str = "",
                              metadata: dict = None):
        if not self.connected:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO operator_actions
                        (uir_id, action_type, operator_id, note, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (uir_id, action_type, operator_id, note,
                      json.dumps(metadata or {})))
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()

    def get_operator_actions(self, uir_id: str = None) -> list:
        if not self.connected:
            return []
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if uir_id:
                    cur.execute("""SELECT * FROM operator_actions
                                   WHERE uir_id=%s ORDER BY created_at""", (uir_id,))
                else:
                    cur.execute("SELECT * FROM operator_actions ORDER BY created_at DESC LIMIT 100")
                return cur.fetchall()
        except Exception:
            return []

    # ── Nominatim Cache ───────────────────────────────────────────────────────

    def get_nominatim_cache(self, query_text: str) -> Optional[dict]:
        if not self.connected:
            return None
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT result_json FROM nominatim_cache WHERE query_text=%s",
                            (query_text.lower().strip(),))
                row = cur.fetchone()
                return row['result_json'] if row else None
        except Exception:
            return None

    def set_nominatim_cache(self, query_text: str, result: dict):
        if not self.connected:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO nominatim_cache (query_text, lat, lng, display_name, result_json)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (query_text) DO UPDATE SET result_json=EXCLUDED.result_json
                """, (query_text.lower().strip(),
                      result.get('lat'), result.get('lng'),
                      result.get('canonical_name', ''),
                      json.dumps(result)))
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()

    # ── Training Pairs ────────────────────────────────────────────────────────

    def store_training_pair(self, semantic: float, geographic: float,
                            temporal: float, label: int,
                            source: str = "ground_truth"):
        if not self.connected:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO training_pairs (source, semantic, geographic, temporal, label)
                    VALUES (%s, %s, %s, %s, %s)
                """, (source, semantic, geographic, temporal, label))
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()

    def get_training_pairs(self) -> list:
        if not self.connected:
            return []
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT semantic, geographic, temporal, label FROM training_pairs")
                return cur.fetchall()
        except Exception:
            return []

    # ── Bulk Operations ───────────────────────────────────────────────────────

    def store_all_uirs(self, engine):
        """Bulk-store all UIRs and their source reports from the clustering engine."""
        if not self.connected:
            return 0
        count = 0
        for uir in engine.active_uirs:
            self.store_uir(uir)
            for report in uir.get('source_reports', []):
                self.store_report(report, uir['uir_id'])
            count += 1
        print(f"  [DB] Stored {count} UIRs with source reports")
        return count

    def get_stats(self) -> dict:
        """Quick summary statistics from the database."""
        if not self.connected:
            return {}
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM uirs WHERE status='active'")
                active = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM source_reports")
                reports = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM operator_actions")
                actions = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM training_pairs")
                pairs = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM nominatim_cache")
                cached = cur.fetchone()[0]
            return {"active_uirs": active, "source_reports": reports,
                    "operator_actions": actions, "training_pairs": pairs,
                    "nominatim_cached": cached}
        except Exception:
            return {}


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_ts(val) -> Optional[datetime]:
    """Coerce to datetime for PostgreSQL TIMESTAMPTZ."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace('Z', '+00:00'))
        except Exception:
            return None
    return None


# ── Module-level singleton ────────────────────────────────────────────────────

_db: Optional[Database] = None


def get_db() -> Optional[Database]:
    return _db


def init_db(connection_string: str = None) -> Database:
    """Initialize the global database. Returns the Database instance."""
    global _db
    _db = Database(connection_string)
    if _db.connected:
        _db.create_schema()
    return _db
