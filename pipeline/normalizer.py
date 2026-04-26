"""Stage 1-2: Normalization"""
from datetime import datetime, timezone

_vc = 0
_sc = 0

def get_next_source_id(ch):
    global _vc, _sc
    if ch == "voice": _vc += 1; return f"call_{_vc:04d}"
    else: _sc += 1; return f"sms_{_sc:04d}"

def reset_counters():
    global _vc, _sc; _vc = 0; _sc = 0

def normalize(raw, channel, receive_time):
    UMAP = {'critical':'CRITICAL','high':'HIGH','medium':'MEDIUM','low':'LOW',
            '1':'CRITICAL','2':'HIGH','3':'MEDIUM','4':'LOW'}
    VALID = {'flood','fire','landslide','medical','accident'}
    out, flags = {}, []
    raw_t = str(raw.get('incident_type', raw.get('type',''))).lower().strip()
    out['incident_type'] = raw_t if raw_t in VALID else 'unknown'
    out['location_raw'] = str(raw.get('location_raw', raw.get('loc_text', raw.get('location','')))).strip()
    out['urgency'] = UMAP.get(str(raw.get('urgency','medium')).lower().strip(), 'MEDIUM')
    pv = raw.get('people_involved', raw.get('people', None))
    try: out['people_involved'] = max(0, min(int(pv), 9999)) if pv is not None else None
    except: out['people_involved'] = None
    ts = raw.get('timestamp', raw.get('ts', None))
    if ts:
        try:
            p = datetime.fromisoformat(str(ts))
            out['timestamp'] = p.replace(tzinfo=timezone.utc) if p.tzinfo is None else p
        except: out['timestamp'] = receive_time; flags.append('no_timestamp')
    else: out['timestamp'] = receive_time; flags.append('no_timestamp')
    try: c = float(raw.get('confidence', 0.5)); out['confidence'] = max(0.0, min(c, 1.0))
    except: out['confidence'] = 0.5
    if out['confidence'] < 0.5: flags.append('low_confidence')
    kp = raw.get('key_phrases', [])
    out['key_phrases'] = [str(x) for x in kp] if isinstance(kp, list) else []
    out['channel'] = 'voice' if channel == 'voice' else 'sms'
    out['receive_time'] = receive_time
    out['source_id'] = raw.get('source_id', get_next_source_id(channel))
    if not out['location_raw']: flags.append('no_location')
    out['flags'] = flags
    return out

def normalize_batch(reports):
    result = []
    for raw in reports:
        ch = raw.get('channel', 'sms')
        rts = raw.get('receive_time')
        if rts:
            try:
                rt = datetime.fromisoformat(str(rts))
                rt = rt.replace(tzinfo=timezone.utc) if rt.tzinfo is None else rt
            except: rt = datetime.now(timezone.utc)
        else: rt = datetime.now(timezone.utc)
        gt = raw.get('_ground_truth')
        nr = normalize(raw, ch, rt)
        if gt: nr['_ground_truth'] = gt
        result.append(nr)
    return result
