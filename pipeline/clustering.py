"""
C4 Pipeline — Stage 5 & 6: Clustering + Conflict Detection + UIR Management
============================================================================
- Three-dimensional similarity: 0.5*semantic + 0.3*geographic + 0.2*temporal
  (weights replaced by LearnedSimilarityModel once operator corrections exist)
- Urgency-sensitive DBSCAN epsilon
- Five-condition safety gate for Critical reports
- Conflict detection and weighted resolution
- UIR creation, update, and timeline reconstruction
- Operator-driven split_uir / merge_uirs for gold-label feedback loop
"""

import math
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import Counter
from copy import deepcopy
from typing import Optional


# ============================================================
# EPSILON VALUES — Urgency-Sensitive
# ============================================================

EPSILON = {
    'CRITICAL': 0.15,   # min similarity 0.85 — near-refusal to merge
    'HIGH':     0.25,   # min similarity 0.75
    'MEDIUM':   0.30,   # min similarity 0.70
    'LOW':      0.40,   # min similarity 0.60
}

# Critical safety gate thresholds
CRITICAL_GATE = {
    'min_semantic': 0.88,
    'max_distance_km': 0.8,
    'max_time_gap_min': 15,
    'min_corroborating_sources': 2,
    'require_type_match': True,
}

NON_CRITICAL_GATE = {
    'min_semantic': 0.70,
    'max_distance_km': 3.0,
    'max_time_gap_min': 45,
    'min_corroborating_sources': 1,
    'require_type_match': False,
}


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================

def haversine_km(lat1, lng1, lat2, lng2) -> float:
    """Calculate distance in km between two GPS coordinates."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def semantic_similarity(embed_r: np.ndarray, embed_u: np.ndarray) -> float:
    """Cosine similarity via dot product (both L2-normalized)."""
    return float(np.dot(embed_r, embed_u))


def geographic_similarity(r: dict, uir: dict) -> float:
    """Geographic similarity: 1 - min(distance_km / 5.0, 1.0). 0 if unresolved."""
    r_lat, r_lng = r.get('lat'), r.get('lng')
    u_lat, u_lng = uir.get('lat'), uir.get('lng')
    if r_lat is None or u_lat is None:
        return 0.0
    d_km = haversine_km(r_lat, r_lng, u_lat, u_lng)
    return 1.0 - min(d_km / 5.0, 1.0)


def temporal_similarity(r: dict, uir: dict) -> float:
    """Temporal similarity: exp(-0.05 * delta_t_minutes). Half-life ~14 min."""
    r_time = r.get('timestamp', r.get('receive_time'))
    u_time = uir.get('last_updated', uir.get('created_at'))
    if r_time is None or u_time is None:
        return 0.5  # default moderate similarity
    dt_min = abs((r_time - u_time).total_seconds()) / 60
    return math.exp(-0.05 * dt_min)


# Optional learned similarity model (set via set_similarity_model() from server.py)
_similarity_model = None


def set_similarity_model(model) -> None:
    """Inject a LearnedSimilarityModel so combined_similarity uses learned weights."""
    global _similarity_model
    _similarity_model = model


def combined_similarity(report: dict, uir: dict) -> dict:
    """
    Compute three-dimensional similarity score.
    If an MLDeduplicator is injected (has ml_similarity), uses XGBoost directly.
    If a LearnedSimilarityModel is injected, uses its learned weights.
    Otherwise falls back to fixed weights: 0.5*sem + 0.3*geo + 0.2*time.
    Returns dict with component scores and combined score.
    """
    # XGBoost path — uses all 9 features including interaction terms
    if (_similarity_model is not None
            and hasattr(_similarity_model, 'ml_similarity')
            and _similarity_model.is_trained):
        return _similarity_model.ml_similarity(report, uir)

    s_sem = semantic_similarity(report['embedding'], uir['centroid_embedding'])
    s_geo = geographic_similarity(report, uir)
    s_time = temporal_similarity(report, uir)

    if _similarity_model is not None and _similarity_model.is_trained:
        combined = _similarity_model.predict_score(s_sem, s_geo, s_time)
    else:
        combined = 0.5 * s_sem + 0.3 * s_geo + 0.2 * s_time

    return {
        'semantic': s_sem,
        'geographic': s_geo,
        'temporal': s_time,
        'combined': combined,
        'distance': 1.0 - combined,
    }


# ============================================================
# FIVE-CONDITION SAFETY GATE
# ============================================================

def passes_safety_gate(report: dict, uir: dict, sim_scores: dict) -> tuple[bool, list[str]]:
    """
    Check if a merge passes the safety gate.
    For CRITICAL reports: all 5 conditions must pass.
    For non-CRITICAL: relaxed thresholds.
    Returns (passed, list_of_failed_conditions).
    """
    is_critical = report['urgency'] == 'CRITICAL'
    gate = CRITICAL_GATE if is_critical else NON_CRITICAL_GATE
    failures = []
    
    # Condition 1: Semantic similarity
    if sim_scores['semantic'] < gate['min_semantic']:
        failures.append(f"semantic={sim_scores['semantic']:.3f} < {gate['min_semantic']}")
    
    # Condition 2: Geographic distance
    if report.get('lat') and uir.get('lat'):
        dist_km = haversine_km(report['lat'], report['lng'], uir['lat'], uir['lng'])
        if dist_km > gate['max_distance_km']:
            failures.append(f"distance={dist_km:.2f}km > {gate['max_distance_km']}km")
    
    # Condition 3: Time gap
    r_time = report.get('timestamp', report.get('receive_time'))
    u_time = uir.get('last_updated')
    if r_time and u_time:
        dt_min = abs((r_time - u_time).total_seconds()) / 60
        if dt_min > gate['max_time_gap_min']:
            failures.append(f"time_gap={dt_min:.1f}min > {gate['max_time_gap_min']}min")
    
    # Condition 4: Corroborating sources
    if uir['source_count'] < gate['min_corroborating_sources']:
        failures.append(f"sources={uir['source_count']} < {gate['min_corroborating_sources']}")
    
    # Condition 5: Incident type match
    if gate['require_type_match']:
        if report['incident_type'] != uir['incident_type'] and report['incident_type'] != 'unknown':
            failures.append(f"type_mismatch: {report['incident_type']} != {uir['incident_type']}")
    
    passed = len(failures) == 0
    return passed, failures


# ============================================================
# CONFLICT DETECTION
# ============================================================

def detect_conflicts(uir: dict, new_report: dict) -> list[dict]:
    """Detect field-level contradictions between a UIR and a new report."""
    conflicts = []
    
    # People count: conflict if differ by > 30%
    uir_people = uir.get('people_involved', {}).get('value')
    new_people = new_report.get('people_involved')
    if uir_people is not None and new_people is not None:
        try:
            ep_i = int(str(uir_people).split('–')[0].split('-')[0])  # handle range strings
            np_i = int(new_people)
            if ep_i != np_i and abs(ep_i - np_i) / max(ep_i, np_i, 1) > 0.30:
                conflicts.append({
                    'field': 'people_involved',
                    'existing': ep_i,
                    'incoming': np_i,
                })
        except (ValueError, TypeError):
            pass
    
    # Incident type: exact mismatch
    if (new_report.get('incident_type', 'unknown') != 'unknown' and
            uir.get('incident_type') != new_report.get('incident_type')):
        conflicts.append({
            'field': 'incident_type',
            'existing': uir.get('incident_type'),
            'incoming': new_report.get('incident_type'),
        })
    
    # Urgency mismatch
    if new_report.get('urgency') != uir.get('urgency'):
        conflicts.append({
            'field': 'urgency',
            'existing': uir.get('urgency'),
            'incoming': new_report.get('urgency'),
        })
    
    return conflicts


def resolve_conflict(field: str, uir: dict, new_report: dict) -> dict:
    """
    Three-factor weighted resolution.
    Priority: 1. Source reliability (voice > SMS)
              2. Corroboration (2+ sources agreeing)
              3. Recency (newer wins)
    """
    if field == 'people_involved':
        existing_val = uir.get('people_involved', {}).get('value')
        new_val = new_report.get('people_involved')
    elif field == 'urgency':
        existing_val = uir.get('urgency')
        new_val = new_report.get('urgency')
        # Urgency always upgrades to highest
        urgency_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        if urgency_order.get(new_val, 0) > urgency_order.get(existing_val, 0):
            return {'value': new_val, 'resolved': True, 'method': 'urgency_escalation'}
        return {'value': existing_val, 'resolved': True, 'method': 'urgency_kept'}
    else:
        existing_val = uir.get(field)
        new_val = new_report.get(field)
    
    # Factor 1: Source reliability (voice > SMS)
    if (new_report.get('channel') == 'voice' and
            uir.get('dominant_channel') == 'sms'):
        return {'value': new_val, 'resolved': True, 'method': 'source_reliability'}
    
    # Factor 2: Corroboration
    source_values = uir.get('source_values', {}).get(field, [])
    value_counts = Counter(str(sv.get('value', '')) for sv in source_values)
    value_counts[str(new_val)] = value_counts.get(str(new_val), 0) + 1
    
    if value_counts:
        top_val, top_count = value_counts.most_common(1)[0]
        if top_count >= 2:
            return {'value': top_val, 'resolved': True, 'method': 'corroboration'}
    
    # Factor 3: Recency
    r_time = new_report.get('timestamp', new_report.get('receive_time'))
    u_time = uir.get('last_updated')
    if r_time and u_time and r_time > u_time:
        return {'value': new_val, 'resolved': True, 'method': 'recency'}
    
    # Cannot resolve — display range
    try:
        lo = min(int(str(existing_val).split('–')[0].split('-')[0]), int(new_val))
        hi = max(int(str(existing_val).split('–')[-1].split('-')[-1]), int(new_val))
        range_str = f'{lo}–{hi}'
    except (ValueError, TypeError):
        range_str = f'{existing_val} / {new_val}'
    
    return {'value': range_str, 'resolved': False, 'method': 'range_flagged'}


# ============================================================
# UIR MANAGEMENT
# ============================================================

_uir_counter = 0


def create_uir(report: dict, event_date: str = None) -> dict:
    """Create a new single-source UIR from a report."""
    global _uir_counter
    _uir_counter += 1
    
    if event_date is None:
        ts = report.get('timestamp', report.get('receive_time', datetime.now(timezone.utc)))
        event_date = ts.strftime('%Y%m%d')
    
    uir_id = f"UIR-{event_date}-{_uir_counter:04d}"
    
    uir = {
        'uir_id': uir_id,
        'incident_type': report['incident_type'],
        'location': {
            'display_name': report.get('location_resolved', {}).get('canonical_name') or report.get('location_raw', 'Unknown'),
            'lat': report.get('lat'),
            'lng': report.get('lng'),
            'resolved': report.get('lat') is not None,
            'source_strings': [report.get('location_raw', '')],
        },
        'lat': report.get('lat'),
        'lng': report.get('lng'),
        'people_involved': {
            'value': report.get('people_involved'),
            'resolved': True,
            'conflict': False,
            'method': 'initial',
        },
        'urgency': report['urgency'],
        'confidence': compute_confidence(report, source_count=1, unresolved_conflicts=0),
        'source_count': 1,
        'sources': [{
            'id': report['source_id'],
            'channel': report['channel'],
            'time': report.get('timestamp', report.get('receive_time')),
            'confidence': report['confidence'],
        }],
        'source_reports': [report],
        'source_values': {
            'people_involved': [{'value': report.get('people_involved'), 'source': report['source_id']}],
        },
        'dominant_channel': report['channel'],
        'timeline': [],
        'flags': list(report.get('flags', [])),
        'linked_uirs': [],
        'status': 'active',
        'created_at': report.get('timestamp', report.get('receive_time')),
        'last_updated': report.get('timestamp', report.get('receive_time')),
        'centroid_embedding': report['embedding'].copy(),
        'operator_actions': [],
        # Ground truth for evaluation
        '_ground_truth_incidents': [report.get('_ground_truth', {}).get('incident_id', 'UNKNOWN')],
    }
    
    uir['timeline'] = rebuild_timeline(uir['source_reports'])
    return uir


def merge_into_uir(uir: dict, report: dict, conflicts: list[dict]) -> dict:
    """Merge a new report into an existing UIR."""
    # Add source
    uir['sources'].append({
        'id': report['source_id'],
        'channel': report['channel'],
        'time': report.get('timestamp', report.get('receive_time')),
        'confidence': report['confidence'],
    })
    uir['source_reports'].append(report)
    uir['source_count'] = len(uir['sources'])
    
    # Track ground truth
    inc_id = report.get('_ground_truth', {}).get('incident_id', 'UNKNOWN')
    if inc_id not in uir['_ground_truth_incidents']:
        uir['_ground_truth_incidents'].append(inc_id)
    
    # Add location string
    loc_str = report.get('location_raw', '')
    if loc_str and loc_str not in uir['location']['source_strings']:
        uir['location']['source_strings'].append(loc_str)
    
    # Update location if newly resolved
    if report.get('lat') and not uir['location']['resolved']:
        uir['location']['lat'] = report['lat']
        uir['location']['lng'] = report['lng']
        uir['location']['resolved'] = True
        uir['lat'] = report['lat']
        uir['lng'] = report['lng']
    
    # Urgency: always escalate to highest
    urgency_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    if urgency_order.get(report['urgency'], 0) > urgency_order.get(uir['urgency'], 0):
        uir['urgency'] = report['urgency']
    
    # Dominant channel
    channels = [s['channel'] for s in uir['sources']]
    uir['dominant_channel'] = 'voice' if channels.count('voice') >= channels.count('sms') else 'sms'
    
    # Handle conflicts
    unresolved_count = 0
    for conflict in conflicts:
        field = conflict['field']
        resolution = resolve_conflict(field, uir, report)
        
        if field == 'people_involved':
            uir['people_involved'] = {
                'value': resolution['value'],
                'resolved': resolution['resolved'],
                'conflict': True,
                'method': resolution['method'],
            }
            if not resolution['resolved']:
                unresolved_count += 1
                if 'people_count_conflict' not in uir['flags']:
                    uir['flags'].append('people_count_conflict')
        elif field == 'urgency':
            uir['urgency'] = resolution['value']
        elif field == 'incident_type':
            if not resolution['resolved']:
                unresolved_count += 1
                if 'type_conflict' not in uir['flags']:
                    uir['flags'].append('type_conflict')
    
    # Track source values for corroboration
    if report.get('people_involved') is not None:
        uir['source_values'].setdefault('people_involved', []).append({
            'value': report['people_involved'],
            'source': report['source_id']
        })
    
    # Update timestamp
    r_time = report.get('timestamp', report.get('receive_time'))
    if r_time and (uir['last_updated'] is None or r_time > uir['last_updated']):
        uir['last_updated'] = r_time
    
    # Update centroid embedding (running average)
    n = uir['source_count']
    uir['centroid_embedding'] = (
        (uir['centroid_embedding'] * (n - 1) + report['embedding']) / n
    )
    # Re-normalize
    norm = np.linalg.norm(uir['centroid_embedding'])
    if norm > 0:
        uir['centroid_embedding'] /= norm
    
    # Recompute confidence
    uir['confidence'] = compute_confidence_uir(uir, unresolved_count)
    
    # Rebuild timeline
    uir['timeline'] = rebuild_timeline(uir['source_reports'])
    
    # Flag for operator review if conflicts unresolved
    if unresolved_count > 0 and 'operator_review_required' not in uir['flags']:
        uir['flags'].append('operator_review_required')
    
    return uir


def compute_confidence(report: dict, source_count: int, unresolved_conflicts: int) -> float:
    """C = C_base * P_source * P_conflict * P_location"""
    c_base = report['confidence']
    p_source = 0.80 if source_count == 1 else 1.0
    p_conflict = 0.85 ** unresolved_conflicts
    p_location = 0.75 if report.get('lat') is None else 1.0
    return round(c_base * p_source * p_conflict * p_location, 4)


def compute_confidence_uir(uir: dict, unresolved_conflicts: int = 0) -> float:
    """Compute confidence for a UIR based on all sources."""
    confidences = [s['confidence'] for s in uir['sources']]
    c_base = sum(confidences) / len(confidences) if confidences else 0.5
    p_source = 0.80 if uir['source_count'] == 1 else 1.0
    p_conflict = 0.85 ** unresolved_conflicts
    p_location = 0.75 if not uir['location']['resolved'] else 1.0
    return round(c_base * p_source * p_conflict * p_location, 4)


def rebuild_timeline(sources: list[dict]) -> list[dict]:
    """Build chronological timeline from all source reports."""
    sorted_src = sorted(sources, key=lambda s: s.get('timestamp', s.get('receive_time', datetime.min.replace(tzinfo=timezone.utc))))
    timeline = []
    for s in sorted_src:
        ts = s.get('timestamp', s.get('receive_time'))
        time_str = ts.strftime('%H:%M') if ts else '??:??'
        summary = s['key_phrases'][0] if s.get('key_phrases') else s.get('incident_type', 'flood')
        timeline.append({
            'time': time_str,
            'summary': summary,
            'source': s.get('source_id', 'unknown'),
            'channel': s.get('channel', 'unknown'),
        })
    return timeline


# ============================================================
# OPERATOR-DRIVEN SPLIT / MERGE  (gold-label feedback loop)
# ============================================================

def split_uir(uir: dict, split_source_ids: list) -> Optional[dict]:
    """
    Operator decided some sources in uir belong to a different incident.
    Removes sources whose source_id is in split_source_ids from uir,
    creates and returns a new UIR from those sources.
    Returns None if split_source_ids are not all found in uir.
    Caller is responsible for appending the new UIR to engine.active_uirs.
    """
    split_set = set(split_source_ids)
    kept_reports = [r for r in uir['source_reports'] if r.get('source_id') not in split_set]
    moved_reports = [r for r in uir['source_reports'] if r.get('source_id') in split_set]

    if not moved_reports or not kept_reports:
        # Cannot split: would empty one side
        return None

    # Rebuild the original UIR from kept reports
    uir['source_reports'] = kept_reports
    uir['sources'] = [s for s in uir['sources'] if s['id'] not in split_set]
    uir['source_count'] = len(uir['sources'])

    # Recompute centroid embedding from kept reports
    embeds = [r['embedding'] for r in kept_reports if r.get('embedding') is not None]
    if embeds:
        centroid = np.mean(embeds, axis=0)
        norm = np.linalg.norm(centroid)
        uir['centroid_embedding'] = centroid / norm if norm > 0 else centroid

    # Recompute urgency (max of remaining)
    urgency_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    uir['urgency'] = max(
        (r['urgency'] for r in kept_reports),
        key=lambda u: urgency_order.get(u, 0),
        default=uir['urgency']
    )
    uir['confidence'] = compute_confidence_uir(uir)
    uir['timeline'] = rebuild_timeline(kept_reports)
    uir['_ground_truth_incidents'] = list({
        r.get('_ground_truth', {}).get('incident_id', 'UNKNOWN') for r in kept_reports
    })

    # Build new UIR from moved reports (use first as seed, merge the rest)
    new_uir = create_uir(moved_reports[0])
    for r in moved_reports[1:]:
        merge_into_uir(new_uir, r, detect_conflicts(new_uir, r))

    return new_uir


def merge_uirs(uir_a: dict, uir_b: dict) -> dict:
    """
    Operator decided uir_b is the same incident as uir_a.
    Merges all sources from uir_b into uir_a (which is updated in-place).
    Caller should mark uir_b as 'dismissed' and update ALL_UIRS.
    Returns updated uir_a.
    """
    for r in uir_b['source_reports']:
        conflicts = detect_conflicts(uir_a, r)
        merge_into_uir(uir_a, r, conflicts)

    # Merge linked_uirs lists (dedup)
    combined_links = set(uir_a['linked_uirs']) | set(uir_b['linked_uirs'])
    combined_links.discard(uir_a['uir_id'])
    combined_links.discard(uir_b['uir_id'])
    uir_a['linked_uirs'] = list(combined_links)

    return uir_a


# ============================================================
# INCREMENTAL DBSCAN CLUSTERING ENGINE
# ============================================================

class IncidentClusterEngine:
    """
    Incremental DBSCAN clustering for incoming reports.
    Maintains a list of active UIRs and assigns incoming reports.
    """
    
    def __init__(self, epsilon_values: dict = None):
        self.active_uirs: list[dict] = []
        self.epsilon = epsilon_values or EPSILON
        self.stats = {
            'total_reports': 0,
            'new_uirs': 0,
            'merges': 0,
            'blocked_merges': 0,
            'possible_duplicates': 0,
        }
    
    def process_report(self, report: dict) -> dict:
        """
        Process a single report through the clustering engine.
        Returns the UIR it was assigned to (new or existing).
        """
        self.stats['total_reports'] += 1
        
        eps = self.epsilon[report['urgency']]
        best_uir = None
        best_score = 0.0
        best_sim = None
        
        # Find best matching UIR
        for uir in self.active_uirs:
            if uir['status'] != 'active':
                continue
            
            sim = combined_similarity(report, uir)
            distance = sim['distance']
            
            if distance <= eps and sim['combined'] > best_score:
                best_score = sim['combined']
                best_uir = uir
                best_sim = sim
        
        if best_uir is not None:
            # Check safety gate
            passed, failures = passes_safety_gate(report, best_uir, best_sim)
            
            if passed:
                # Merge
                conflicts = detect_conflicts(best_uir, report)
                merge_into_uir(best_uir, report, conflicts)
                self.stats['merges'] += 1
                return best_uir
            else:
                # Safety gate blocked merge — create new UIR, link as possible duplicate
                new_uir = create_uir(report)
                new_uir['linked_uirs'].append(best_uir['uir_id'])
                new_uir['flags'].append('operator_review_required')
                best_uir['linked_uirs'].append(new_uir['uir_id'])
                if 'possible_duplicate_nearby' not in best_uir['flags']:
                    best_uir['flags'].append('possible_duplicate_nearby')
                self.active_uirs.append(new_uir)
                self.stats['blocked_merges'] += 1
                self.stats['new_uirs'] += 1
                self.stats['possible_duplicates'] += 1
                return new_uir
        else:
            # No match — create new UIR
            new_uir = create_uir(report)
            self.active_uirs.append(new_uir)
            self.stats['new_uirs'] += 1
            return new_uir
    
    def process_batch(self, reports: list[dict]) -> list[dict]:
        """Process a batch of reports sequentially."""
        results = []
        for i, report in enumerate(reports):
            uir = self.process_report(report)
            results.append(uir)
            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(reports)} reports, "
                      f"active UIRs: {len([u for u in self.active_uirs if u['status'] == 'active'])}")
        return results
    
    def get_summary(self) -> dict:
        """Get clustering summary statistics."""
        active = [u for u in self.active_uirs if u['status'] == 'active']
        return {
            'total_reports': self.stats['total_reports'],
            'total_uirs': len(self.active_uirs),
            'active_uirs': len(active),
            'merges': self.stats['merges'],
            'blocked_merges': self.stats['blocked_merges'],
            'possible_duplicates': self.stats['possible_duplicates'],
            'uirs_by_urgency': Counter(u['urgency'] for u in active),
            'uirs_by_source_count': Counter(u['source_count'] for u in active),
            'avg_sources_per_uir': sum(u['source_count'] for u in active) / max(len(active), 1),
            'flagged_uirs': sum(1 for u in active if u['flags']),
            'conflict_uirs': sum(1 for u in active if 'people_count_conflict' in u.get('flags', [])),
        }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_clustering(engine: IncidentClusterEngine) -> dict:
    """
    Evaluate clustering quality against ground truth.
    Computes precision, recall, F1, and critical false-merge rate.
    """
    active = [u for u in engine.active_uirs if u['status'] == 'active']
    
    # Precision: % of UIRs where all reports belong to same ground truth incident
    pure_uirs = 0
    multi_source_uirs = 0
    
    for uir in active:
        gt_incidents = set(uir['_ground_truth_incidents'])
        gt_incidents.discard('UNKNOWN')
        if len(gt_incidents) <= 1:
            pure_uirs += 1
        if uir['source_count'] > 1:
            multi_source_uirs += 1
    
    precision = pure_uirs / max(len(active), 1)
    
    # Recall: % of same-incident report pairs that are in the same UIR
    # Build ground truth clusters
    gt_clusters = {}
    for uir in active:
        for report in uir.get('source_reports', []):
            gt_id = report.get('_ground_truth', {}).get('incident_id', 'UNKNOWN')
            if gt_id != 'UNKNOWN':
                gt_clusters.setdefault(gt_id, []).append(uir['uir_id'])
    
    total_same_pairs = 0
    correctly_merged_pairs = 0
    
    for gt_id, uir_ids in gt_clusters.items():
        n = len(uir_ids)
        if n < 2:
            continue
        total_same_pairs += n * (n - 1) // 2
        # Count pairs in the same UIR
        uir_counts = Counter(uir_ids)
        for uid, count in uir_counts.items():
            correctly_merged_pairs += count * (count - 1) // 2
    
    recall = correctly_merged_pairs / max(total_same_pairs, 1)
    
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    
    # Critical false-merge rate
    critical_uirs = [u for u in active if u['urgency'] == 'CRITICAL']
    critical_false_merges = 0
    for uir in critical_uirs:
        gt = set(uir['_ground_truth_incidents'])
        gt.discard('UNKNOWN')
        if len(gt) > 1:
            critical_false_merges += 1
    
    critical_fmr = critical_false_merges / max(len(critical_uirs), 1)
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'critical_false_merge_rate': round(critical_fmr, 4),
        'total_uirs': len(active),
        'pure_uirs': pure_uirs,
        'multi_source_uirs': multi_source_uirs,
        'total_same_pairs': total_same_pairs,
        'correctly_merged_pairs': correctly_merged_pairs,
        'critical_uirs': len(critical_uirs),
        'critical_false_merges': critical_false_merges,
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    import json
    import sys
    
    from pipeline.normalizer import normalize_batch, reset_counters
    from pipeline.gazetteer import resolve_batch
    from pipeline.embedder import embed_batch
    
    print("Loading dataset...")
    with open("/home/claude/data/disaster_dataset_600.json") as f:
        dataset = json.load(f)
    
    print("Stage 1-2: Normalizing...")
    reset_counters()
    normalized = normalize_batch(dataset["reports"])
    
    print("Stage 3: Resolving locations...")
    resolved = resolve_batch(normalized)
    
    print("Stage 4: Generating embeddings...")
    embedded = embed_batch(resolved)
    
    print("\nStage 5-6: Clustering + Conflict Detection...")
    engine = IncidentClusterEngine()
    engine.process_batch(embedded)
    
    # Summary
    summary = engine.get_summary()
    print(f"\n{'='*60}")
    print(f"  CLUSTERING RESULTS")
    print(f"{'='*60}")
    print(f"  Total reports processed:  {summary['total_reports']}")
    print(f"  Active UIRs created:      {summary['active_uirs']}")
    print(f"  Reports merged:           {summary['merges']}")
    print(f"  Blocked merges (safety):  {summary['blocked_merges']}")
    print(f"  Possible duplicates:      {summary['possible_duplicates']}")
    print(f"  Avg sources per UIR:      {summary['avg_sources_per_uir']:.1f}")
    print(f"  Flagged UIRs:             {summary['flagged_uirs']}")
    print(f"  Conflict UIRs:            {summary['conflict_uirs']}")
    print(f"{'='*60}")
    print(f"  UIRs BY URGENCY:")
    for u, c in summary['uirs_by_urgency'].most_common():
        print(f"    {u}: {c}")
    print(f"{'='*60}")
    print(f"  UIRs BY SOURCE COUNT:")
    for sc, c in sorted(summary['uirs_by_source_count'].items()):
        print(f"    {sc} sources: {c} UIRs")
    
    # Evaluation
    eval_results = evaluate_clustering(engine)
    print(f"\n{'='*60}")
    print(f"  EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"  Precision:                {eval_results['precision']:.4f}  (target > 0.90)")
    print(f"  Recall:                   {eval_results['recall']:.4f}  (target > 0.85)")
    print(f"  F1 Score:                 {eval_results['f1']:.4f}")
    print(f"  Critical FMR:             {eval_results['critical_false_merge_rate']:.4f}  (target < 0.02)")
    print(f"  Pure UIRs:                {eval_results['pure_uirs']}/{eval_results['total_uirs']}")
    print(f"  Multi-source UIRs:        {eval_results['multi_source_uirs']}")
    print(f"  Critical false merges:    {eval_results['critical_false_merges']}/{eval_results['critical_uirs']}")
    print(f"{'='*60}")
    
    # Show sample UIRs
    print(f"\n  SAMPLE UIRs:")
    active = [u for u in engine.active_uirs if u['status'] == 'active']
    active.sort(key=lambda u: {'CRITICAL':0,'HIGH':1,'MEDIUM':2,'LOW':3}.get(u['urgency'],4))
    for uir in active[:5]:
        print(f"\n  {uir['uir_id']} [{uir['urgency']}]")
        print(f"    Location:    {uir['location']['display_name']}")
        print(f"    Sources:     {uir['source_count']}")
        print(f"    People:      {uir['people_involved']['value']} "
              f"({'conflict' if uir['people_involved']['conflict'] else 'ok'})")
        print(f"    Confidence:  {uir['confidence']:.2f}")
        print(f"    Flags:       {uir['flags']}")
        print(f"    GT incidents: {uir['_ground_truth_incidents']}")
        print(f"    Timeline:    {len(uir['timeline'])} entries")
        if uir['timeline']:
            for t in uir['timeline'][:3]:
                print(f"      {t['time']} [{t['channel']}] {t['summary'][:50]}")
