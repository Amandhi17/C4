"""
Stage 3: Multilingual Location Gazetteer (Sinhala + Tamil + Romanized)
======================================================================
Resolution chain — each stage catches what the previous missed:

  A:   Exact match        — instant, GN-division or town level
  A.5: Address fast-path  — detects "No 45/B, Kandy Road, Kelaniya" patterns
                            and routes directly to Google Maps / Nominatim for
                            precise street-level resolution.
  B:   Fuzzy match        — handles typos / transliteration variants
  C:   Landmark pattern   — "near X bridge", "X bridge ළඟ"
  D:   Hierarchical parse — splits "Kelaniya, Gonawala, Yakkala Road ළඟ"
  E:   Google Maps API    — street-level, full formatted address, Sinhala/Tamil
                            aware. Set GOOGLE_MAPS_API_KEY to enable.
  E.5: Nominatim fallback — OpenStreetMap geocoding when no Google key is set.
                            Rate-limited 1 req/sec, cached locally.
  F:   Unresolved         — geo_score = 0, flagged for operator review

Data sources:
  - 31 town/city entries (original hand-curated, backward-compatible)
  - 4,258 GN Division centroids extracted from lka_admin4.geojson
  - Google Maps Geocoding API (GOOGLE_MAPS_API_KEY env var, recommended)
  - OpenStreetMap Nominatim (free fallback, no key needed)

Coverage target: >97% of reports resolve to lat/lng.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional
from rapidfuzz import process, fuzz

# ── Address detection patterns ────────────────────────────────────────────────
# Matches: "No 45", "No. 12/B", "45/A", "45/3", "123B", leading digit+comma
_ADDRESS_PATTERN = re.compile(
    r'(?:'
    r'\bNo\.?\s*\d+'            # "No 45" / "No. 45"
    r'|\b\d+\s*/\s*[A-Z0-9]'   # "45/B" / "12/3"
    r'|\b\d+[A-Z]\b'            # "45B"
    r'|^\d+\s*,'                # "45, Kelaniya Road" — leading number+comma
    r')',
    re.IGNORECASE | re.MULTILINE,
)

# Street-type keywords — used to strip road names when extracting city portion
_STREET_KEYWORDS = re.compile(
    r'\b(?:road|lane|street|mawatha|place|avenue|drive|rd|st|ln|ave)\b'
    r'|(?:පාර|මාවත|ලේන්|රෝඩ්|வீதி|தெரு|மாவத்தை|நெடுஞ்சாலை)',
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# STATIC GAZETTEER — 31 original town entries (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

GAZETTEER = {
    "Kelaniya": {"lat":6.9541,"lng":79.9196,"district":"Colombo","province":"Western",
        "variants":["කැලණිය","කැලණි","කැලණිය පාලම ළඟ","Kelaniya","Kelaniye","Kelani","Kelaniya pala",
                    "කැලණිය පන්සල ළඟ","කැලණිය පාලම පසු කරලා","කැලණිය හන්දිය",
                    "களனி","களணி நகரம்"]},
    "Kaduwela": {"lat":6.9307,"lng":79.9831,"district":"Colombo","province":"Western",
        "variants":["කඩුවෙල","කඩුවෙල නගරය","Kaduwela","Kaduwala","කඩුවෙල පාලම ළඟ","කඩුවෙල හන්දිය",
                    "கடுவெல"]},
    "Kolonnawa": {"lat":6.9297,"lng":79.8853,"district":"Colombo","province":"Western",
        "variants":["කොළොන්නාව","කොලොන්නාව","Kolonnawa","Kolonnawe","කොළොන්නාව ඇළ ළඟ",
                    "කොළොන්නාව තෙල් පිරිපහදුව පසුපස","கொலொன்னாவ"]},
    "Wellampitiya": {"lat":6.9388,"lng":79.8881,"district":"Colombo","province":"Western",
        "variants":["වැල්ලම්පිටිය","වැල්ලම්පිටි","Wellampitiya","Wellampitiye",
                    "වැල්ලම්පිටිය පොළ ළඟ","වැල්ලම්පිටිය පාර","வெல்லம்பிட்டிய"]},
    "Mattakkuliya": {"lat":6.9536,"lng":79.8756,"district":"Colombo","province":"Western",
        "variants":["මට්ටක්කුලිය","Mattakkuliya","Mattakuliya","මට්ටක්කුලිය නිවාස ළඟ",
                    "මට්ටක්කුලිය ඇළ අස","மட்டக்குளிய"]},
    "Grandpass": {"lat":6.9458,"lng":79.8697,"district":"Colombo","province":"Western",
        "variants":["ග්‍රෑන්ඩ්පාස්","ග්‍රෑන්ඩ් පාස්","Grandpass","Grand Pass",
                    "ග්‍රෑන්ඩ්පාස් පාර ළඟ","ග්‍රෑන්ඩ්පාස් නිවාස","கிராண்ட்பாஸ்"]},
    "Dematagoda": {"lat":6.9326,"lng":79.8741,"district":"Colombo","province":"Western",
        "variants":["දෙමටගොඩ","Dematagoda","දෙමටගොඩ දුම්රිය ළඟ","දෙමටගොඩ ඇළ","தெமடகொட"]},
    "Battaramulla": {"lat":6.8990,"lng":79.9183,"district":"Colombo","province":"Western",
        "variants":["බත්තරමුල්ල","Battaramulla","පාර්ලිමේන්තු පාර ළඟ","බත්තරමුල්ල හන්දිය",
                    "பத்தரமுல்ல"]},
    "Hanwella": {"lat":6.9012,"lng":80.0851,"district":"Colombo","province":"Western",
        "variants":["හංවැල්ල","Hanwella","Hanwelle","හංවැල්ල පාලම ළඟ","හංවැල්ල නගරය","ஹன்வெல்ல"]},
    "Avissawella": {"lat":6.9533,"lng":80.2100,"district":"Colombo","province":"Western",
        "variants":["අවිස්සාවේල්ල","Avissawella","Avissawelle","අවිස්සාවේල්ල නගරය ළඟ","அவிஸாவெல்ல"]},
    "Ja-Ela": {"lat":7.0757,"lng":79.8916,"district":"Gampaha","province":"Western",
        "variants":["ජා-ඇල","ජාඇල","Ja-Ela","Ja Ela","Jaela","ජා-ඇල නගරය ළඟ","ජා-ඇල හන්දිය","ஜா-ஏல"]},
    "Kadana": {"lat":7.0009,"lng":80.0251,"district":"Gampaha","province":"Western",
        "variants":["කදාන","කදාන නගරය","Kadana","Kadane","කදාන පාලම ළඟ","கடான"]},
    "Biyagama": {"lat":6.9618,"lng":79.9834,"district":"Gampaha","province":"Western",
        "variants":["බියගම","බියගම කලාපය","Biyagama","Biyagame","බියගම කර්මාන්ත කලාපය ළඟ","பியகம"]},
    "Wattala": {"lat":6.9896,"lng":79.8918,"district":"Gampaha","province":"Western",
        "variants":["වත්තල","වත්තල නගරය","Wattala","Wattale","Watthala","වත්තල හන්දිය ළඟ",
                    "වත්තල පාලම","வத்தல"]},
    "Negombo": {"lat":7.2083,"lng":79.8358,"district":"Gampaha","province":"Western",
        "variants":["මීගමුව","මීගමු","Negombo","Meegamuwa","Migamuwa","මීගමුව කලපුව ළඟ",
                    "මීගමුව මාළු වෙළඳපොළ","நீர்கொழும்பு","நீர்கொழும்பு நகரம்"]},
    "Gampaha": {"lat":7.0917,"lng":80.0003,"district":"Gampaha","province":"Western",
        "variants":["ගම්පහ","ගම්පහ නගරය","Gampaha","Gamapaha","ගම්පහ රෝහල ළඟ",
                    "ගම්පහ නගර මධ්‍යයේ","கம்பஹா"]},
    "Kalutara": {"lat":6.5854,"lng":79.9607,"district":"Kalutara","province":"Western",
        "variants":["කළුතර","කළුතර නගරය","Kalutara","Kaluthara","කළුතර පාලම ළඟ",
                    "කළුතර බෝධිය ළඟ","களுத்துறை","களுத்துறை நகரம்"]},
    "Beruwala": {"lat":6.4789,"lng":79.9826,"district":"Kalutara","province":"Western",
        "variants":["බේරුවල","බේරුවළ","Beruwala","Beruwale","බේරුවල වරාය ළඟ","බේරුවල මාළු ගම",
                    "பேருவல"]},
    "Horana": {"lat":6.7153,"lng":80.0622,"district":"Kalutara","province":"Western",
        "variants":["හොරණ","හොරණ නගරය","Horana","Horane","හොරණ හන්දිය ළඟ","හොරණ රෝහල ළඟ","ஹொரண"]},
    "Panadura": {"lat":6.7136,"lng":79.9044,"district":"Kalutara","province":"Western",
        "variants":["පානදුර","පානදුර නගරය","Panadura","Panadure","පානදුර පාලම ළඟ","පානදුර ගඟ අස",
                    "பாணந்துறை"]},
    "Bandaragama": {"lat":6.6977,"lng":80.0388,"district":"Kalutara","province":"Western",
        "variants":["බණ්ඩාරගම","Bandaragama","Bandaregama","බණ්ඩාරගම නගරය ළඟ","බණ්ඩාරගම කුඹුරු",
                    "பண்டாரகம"]},
    "Kandy": {"lat":7.2906,"lng":80.6337,"district":"Kandy","province":"Central",
        "variants":["මහනුවර","මහනුවර නගරය","නුවර","Kandy","Mahanuwara","Nuwara",
                    "මහනුවර වැව ළඟ","දළදා මාළිගාව ළඟ","මහනුවර නගර මධ්‍යයේ",
                    "கண்டி","கண்டி நகரம்","மாநகரம்"]},
    "Peradeniya": {"lat":7.2590,"lng":80.5964,"district":"Kandy","province":"Central",
        "variants":["පේරාදෙණිය","Peradeniya","Peradeniye","පේරාදෙණිය විශ්ව විද්‍යාලය ළඟ",
                    "පේරාදෙණිය උද්‍යානය","பேராதெனிய"]},
    "Katugastota": {"lat":7.3131,"lng":80.6247,"district":"Kandy","province":"Central",
        "variants":["කටුගස්තොට","Katugastota","Katugasthota","කටුගස්තොට පාලම ළඟ",
                    "කටුගස්තොට නගරය","கட்டுகஸ்தொட"]},
    "Gampola": {"lat":7.1642,"lng":80.5770,"district":"Kandy","province":"Central",
        "variants":["ගම්පොල","ගම්පොළ","Gampola","Gampole","ගම්පොල පාලම ළඟ",
                    "ගම්පොල නගර මධ්‍යයේ","கம்போல"]},
    "Kadugannawa": {"lat":7.2542,"lng":80.5239,"district":"Kandy","province":"Central",
        "variants":["කඩුගන්නාව","Kadugannawa","Kadugannave","කඩුගන්නාව පාස් එක ළඟ",
                    "කඩුගන්නාව දුම්රිය","கடுகண்ணாவ"]},
    "Ratnapura": {"lat":6.6828,"lng":80.3992,"district":"Ratnapura","province":"Central",
        "variants":["රත්නපුර","රත්නපුර නගරය","Ratnapura","Rathnapura","රත්නපුර නගරය ළඟ",
                    "රත්නපුර මැණික් වෙළඳපොළ","இரத்தினபுரி","இரத்தினபுர நகரம்"]},
    "Eheliyagoda": {"lat":6.8438,"lng":80.2708,"district":"Ratnapura","province":"Central",
        "variants":["ඇහැලියගොඩ","Eheliyagoda","Eheliyagode","ඇහැලියගොඩ නගරය ළඟ",
                    "ඇහැලියගොඩ රෝහල ළඟ","எஹெலியகொட"]},
    "Kuruwita": {"lat":6.7746,"lng":80.3662,"district":"Ratnapura","province":"Central",
        "variants":["කුරුවිට","Kuruwita","Kuruwite","කුරුවිට පාලම ළඟ","කුරුවිට නගරය","குருவிட"]},
    "Pelmadulla": {"lat":6.6197,"lng":80.4633,"district":"Ratnapura","province":"Central",
        "variants":["පැල්මඩුල්ල","Pelmadulla","Pelmadulle","පැල්මඩුල්ල හන්දිය ළඟ","பெல்மடுல்ல"]},
    "Balangoda": {"lat":6.6467,"lng":80.7003,"district":"Ratnapura","province":"Central",
        "variants":["බලංගොඩ","Balangoda","Balangode","බලංගොඩ බස් නැවතුම ළඟ","බලංගොඩ රෝහල","பலங்கொட"]},
}


# ═══════════════════════════════════════════════════════════════════════════════
# GN DIVISION LAYER — 4,258 entries from lka_admin4.geojson
# ═══════════════════════════════════════════════════════════════════════════════

_GN_DATA_PATH = Path(__file__).parent.parent / "data" / "gn_divisions.json"
_gn_divisions: list = []


def _load_gn_divisions():
    """Load extracted GN division centroids into the gazetteer index."""
    global _gn_divisions
    if not _GN_DATA_PATH.exists():
        return 0
    with open(_GN_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    _gn_divisions = data.get("gn_divisions", [])

    # Merge into EXACT_INDEX — English and Sinhala names
    added = 0
    for gn in _gn_divisions:
        canon = gn["name_en"]
        if not canon:
            continue
        # Register English name
        k = canon.lower().strip()
        if k not in EXACT_INDEX:
            EXACT_INDEX[k] = canon
            added += 1
        # Register Sinhala name
        si = gn.get("name_si", "")
        if si:
            k_si = si.lower().strip()
            if k_si not in EXACT_INDEX:
                EXACT_INDEX[k_si] = canon
                added += 1
        # Register Tamil name
        ta = gn.get("name_ta", "")
        if ta:
            k_ta = ta.lower().strip()
            if k_ta not in EXACT_INDEX:
                EXACT_INDEX[k_ta] = canon
                added += 1

        # Build GAZETTEER entry if not already a town
        if canon not in GAZETTEER:
            variants = [canon]
            if si:
                variants.append(si)
            if ta:
                variants.append(ta)
            GAZETTEER[canon] = {
                "lat": gn["lat"], "lng": gn["lng"],
                "district": gn["district"], "province": gn["province"],
                "ds_division": gn.get("ds_division", ""),
                "gn_code": gn.get("pcode", ""),
                "level": "gn_division",
                "variants": variants,
            }

    return added


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

EXACT_INDEX: dict = {}

def _build_index():
    """Build combined index from town entries + GN divisions."""
    global ALL_VARIANTS
    # Town entries first (higher priority)
    for canon, entry in GAZETTEER.items():
        for v in entry.get("variants", []):
            k = v.lower().strip()
            if k not in EXACT_INDEX:
                EXACT_INDEX[k] = canon

    # GN divisions
    gn_count = _load_gn_divisions()

    ALL_VARIANTS.clear()
    ALL_VARIANTS.extend(EXACT_INDEX.keys())
    return gn_count

ALL_VARIANTS: list = []


# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

LANDMARK_PATTERNS = [
    # Sinhala postpositions
    re.compile(r'(.+?)\s+(?:ළඟ|පසුපස|අස|පසු කරලා)$'),
    # Tamil postpositions
    re.compile(r'(.+?)\s+(?:அருகில்|பக்கத்தில்|அருகே)$'),
    # English prepositions
    re.compile(r'(?:near|past|behind|beside|at|opposite)\s+(?:the\s+)?(.+?)(?:\s+(?:bridge|hospital|temple|junction|kovil|church|mosque))?$', re.I),
    # Sinhala/English suffix stripping
    re.compile(r'(.+?)\s+(?:area|junction|road|town|city|zone|side|හන්දිය|නගරය|පාර|பகுதி|சந்தி)$', re.I),
]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE A: EXACT MATCH
# ═══════════════════════════════════════════════════════════════════════════════

def lookup_exact(text: str) -> Optional[str]:
    return EXACT_INDEX.get(text.lower().strip())


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE B: FUZZY MATCH
# ═══════════════════════════════════════════════════════════════════════════════

def lookup_fuzzy(text: str, threshold: int = 80) -> Optional[tuple]:
    r = process.extractOne(text.lower().strip(), ALL_VARIANTS,
                           scorer=fuzz.WRatio, score_cutoff=threshold)
    return (EXACT_INDEX[r[0]], r[1]) if r else None


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE C: LANDMARK PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

def lookup_landmark(text: str) -> Optional[tuple]:
    for p in LANDMARK_PATTERNS:
        m = p.search(text.lower().strip())
        if m:
            core = m.group(1).strip()
            if len(core) < 2:
                continue
            ex = lookup_exact(core)
            if ex:
                return ex, 95.0
            fz = lookup_fuzzy(core, 75)
            if fz:
                return fz[0], fz[1] * 0.9
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE D: HIERARCHICAL PARSE  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

_HIER_SPLIT = re.compile(r'[,;/|෴]+')


def hierarchical_resolve(text: str) -> Optional[dict]:
    """
    Split compound location strings and resolve from most specific (rightmost)
    to broadest. Handles callers who say "Kelaniya, Gonawala, Yakkala Road".

    Returns a result dict if any segment resolves, else None.
    """
    parts = _HIER_SPLIT.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return None  # Nothing to split

    # Try from most specific (right) to broadest (left)
    for part in reversed(parts):
        # Try exact
        ex = lookup_exact(part)
        if ex:
            e = GAZETTEER[ex]
            return _make_result(ex, e, "hierarchical_exact", 92,
                                "hierarchical_location", parsed_from=text, matched_part=part)
        # Try fuzzy
        fz = lookup_fuzzy(part, 78)
        if fz:
            e = GAZETTEER[fz[0]]
            return _make_result(fz[0], e, "hierarchical_fuzzy", fz[1] * 0.9,
                                "hierarchical_location", parsed_from=text, matched_part=part)
        # Try landmark on this segment
        lm = lookup_landmark(part)
        if lm:
            e = GAZETTEER[lm[0]]
            return _make_result(lm[0], e, "hierarchical_landmark", lm[1] * 0.85,
                                "hierarchical_location", parsed_from=text, matched_part=part)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED GEOCODING CACHE
# ═══════════════════════════════════════════════════════════════════════════════

_GEO_CACHE: dict = {}
_GEO_CACHE_PATH = Path(__file__).parent.parent / "data" / "geocoding_cache.json"


def _load_geocoding_cache():
    """Load unified geocoding cache (Google Maps + Nominatim results) from disk."""
    # Also load old nominatim_cache.json for backward compatibility
    for old_path in [
        _GEO_CACHE_PATH,
        Path(__file__).parent.parent / "data" / "nominatim_cache.json",
    ]:
        if old_path.exists():
            try:
                with open(old_path, encoding="utf-8") as f:
                    cached = json.load(f)
                _GEO_CACHE.update(cached)
            except Exception:
                pass


def _save_geocoding_cache():
    """Persist unified geocoding cache to disk."""
    try:
        _GEO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_GEO_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_GEO_CACHE, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE E: GOOGLE MAPS GEOCODING API  (primary — set GOOGLE_MAPS_API_KEY)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Why Google Maps over Nominatim:
#   • Returns the EXACT address, not just a place-name centroid
#   • Full formatted address:  "No 45/B, Kandy Road, Kelaniya, Western Province, Sri Lanka"
#   • Street-level lat/lng (precision ~10 m vs ~500 m for town centroid)
#   • Native Sinhala + Tamil Unicode understanding
#   • Structured address components: street, locality, district, province separately
#   • Much higher accuracy for rural/suburban Sri Lanka than OpenStreetMap
#
# Set GOOGLE_MAPS_API_KEY environment variable to enable.
# Free tier: 40,000 requests/month.  For 800 reports, cost = $0.
# ═══════════════════════════════════════════════════════════════════════════════

_GMAPS_API_KEY: Optional[str] = os.environ.get("GOOGLE_MAPS_API_KEY")
_GMAPS_BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def _parse_gmaps_components(components: list) -> dict:
    """
    Extract district and province from Google Maps address_components list.
    Google returns typed components — we pick the right administrative levels.

    Sri Lanka structure:
      administrative_area_level_1 → Province  (e.g. "Western Province")
      administrative_area_level_2 → District  (e.g. "Colombo District")
      locality / sublocality      → Town/City (e.g. "Kelaniya")
    """
    mapping = {}
    for comp in components:
        types = comp.get("types", [])
        name = comp.get("long_name", "")
        if "administrative_area_level_1" in types:
            mapping["province"] = name.replace(" Province", "").strip()
        elif "administrative_area_level_2" in types:
            mapping["district"] = name.replace(" District", "").strip()
        elif "locality" in types and "locality" not in mapping:
            mapping["locality"] = name
        elif "sublocality_level_1" in types and "locality" not in mapping:
            mapping["locality"] = name
        elif "route" in types:
            mapping["route"] = name
        elif "street_number" in types:
            mapping["street_number"] = name
    return mapping


def google_maps_resolve(text: str, db=None) -> Optional[dict]:
    """
    Geocode using Google Maps Geocoding API.

    Returns a result dict with:
      - Precise street-level lat/lng
      - Full formatted address from Google
      - Structured district + province
      - Confidence score = 95 (higher than Nominatim's 85)

    Results are cached in data/geocoding_cache.json so each unique
    location string is only looked up once.

    Requires GOOGLE_MAPS_API_KEY environment variable.
    Falls back silently if key is not set or request fails.
    """
    if not _GMAPS_API_KEY:
        return None
    if not text or not text.strip():
        return None

    cache_key = f"gmaps:{text.lower().strip()}"

    # Check DB cache
    if db is not None:
        try:
            cached = db.get_nominatim_cache(cache_key)
            if cached:
                return cached
        except Exception:
            pass

    # Check memory/disk cache
    if cache_key in _GEO_CACHE:
        return _GEO_CACHE[cache_key]

    try:
        import requests
        params = {
            "address": f"{text}, Sri Lanka",
            "key": _GMAPS_API_KEY,
            "region": "lk",          # bias results toward Sri Lanka
            "language": "en",         # return address in English for consistency
            "components": "country:LK",
        }
        resp = requests.get(_GMAPS_BASE_URL, params=params, timeout=5)
        data = resp.json()

        if data.get("status") != "OK" or not data.get("results"):
            _GEO_CACHE[cache_key] = None
            _save_geocoding_cache()
            return None

        top = data["results"][0]
        loc = top["geometry"]["location"]
        components = _parse_gmaps_components(top.get("address_components", []))
        formatted = top.get("formatted_address", text)

        # Precision score: ROOFTOP > RANGE_INTERPOLATED > GEOMETRIC_CENTER > APPROXIMATE
        precision_scores = {
            "ROOFTOP": 97,
            "RANGE_INTERPOLATED": 93,
            "GEOMETRIC_CENTER": 88,
            "APPROXIMATE": 80,
        }
        location_type = top.get("geometry", {}).get("location_type", "APPROXIMATE")
        score = precision_scores.get(location_type, 85)

        result = _make_result(
            canonical_name=formatted,
            entry={
                "lat": loc["lat"],
                "lng": loc["lng"],
                "district": components.get("district", ""),
                "province": components.get("province", ""),
            },
            method="google_maps",
            score=score,
            flag="google_maps_location",
            formatted_address=formatted,
            location_type=location_type,
            locality=components.get("locality", ""),
            route=components.get("route", ""),
        )

        _GEO_CACHE[cache_key] = result
        _save_geocoding_cache()

        if db is not None:
            try:
                db.set_nominatim_cache(cache_key, result)
            except Exception:
                pass

        return result

    except ImportError:
        pass
    except Exception:
        pass

    _GEO_CACHE[cache_key] = None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE E.5: NOMINATIM FALLBACK  (used when no Google Maps API key is set)
# ═══════════════════════════════════════════════════════════════════════════════

_NOM_LAST_REQUEST: float = 0.0
_NOM_ENABLED = True


def nominatim_resolve(text: str, db=None) -> Optional[dict]:
    """
    Query OpenStreetMap Nominatim for street-level resolution.
    Rate-limited to 1 req/sec per ToS.  Results cached in memory + disk.
    Used as fallback when GOOGLE_MAPS_API_KEY is not set.
    """
    global _NOM_LAST_REQUEST
    if not _NOM_ENABLED or not text or not text.strip():
        return None

    cache_key = f"nom:{text.lower().strip()}"

    # Check DB cache first
    if db is not None:
        try:
            cached = db.get_nominatim_cache(cache_key)
            if cached:
                return cached
        except Exception:
            pass

    if cache_key in _GEO_CACHE:
        return _GEO_CACHE[cache_key]

    # Rate limit: 1 request per second
    now = time.time()
    wait = max(0, 1.05 - (now - _NOM_LAST_REQUEST))
    if wait > 0:
        time.sleep(wait)

    try:
        import requests
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{text}, Sri Lanka",
            "format": "json",
            "limit": 1,
            "countrycodes": "lk",
            "addressdetails": 1,
        }
        headers = {"User-Agent": "C4-DisasterTriage/1.0 (academic-research)"}
        _NOM_LAST_REQUEST = time.time()
        resp = requests.get(url, params=params, timeout=5, headers=headers)
        data = resp.json()

        if data:
            result = _make_result(
                canonical_name=data[0].get("display_name", text)[:80],
                entry={"lat": float(data[0]["lat"]), "lng": float(data[0]["lon"]),
                       "district": data[0].get("address", {}).get("county", ""),
                       "province": data[0].get("address", {}).get("state", "")},
                method="nominatim", score=85,
                flag="nominatim_location",
            )
            _GEO_CACHE[cache_key] = result
            _save_geocoding_cache()

            if db is not None:
                try:
                    db.set_nominatim_cache(cache_key, result)
                except Exception:
                    pass

            return result
    except ImportError:
        pass
    except Exception:
        pass

    _GEO_CACHE[cache_key] = None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _make_result(canonical_name: str, entry: dict, method: str, score: float,
                 flag: Optional[str] = None, **extra) -> dict:
    result = {
        "canonical_name": canonical_name,
        "lat": entry.get("lat"),
        "lng": entry.get("lng"),
        "district": entry.get("district", ""),
        "province": entry.get("province", ""),
        "method": method,
        "score": round(score, 1),
        "flag": flag,
    }
    # Optional GN-level metadata
    if "ds_division" in entry:
        result["ds_division"] = entry["ds_division"]
    if "gn_code" in entry:
        result["gn_code"] = entry["gn_code"]
    if "level" in entry:
        result["level"] = entry["level"]
    result.update(extra)
    return result


_UNRESOLVED = {
    "canonical_name": None, "lat": None, "lng": None,
    "district": None, "province": None,
    "method": "unresolved", "score": 0, "flag": "location_unresolved",
}
_NO_LOCATION = {
    "canonical_name": None, "lat": None, "lng": None,
    "district": None, "province": None,
    "method": "none", "score": 0, "flag": "no_location",
}


# ═══════════════════════════════════════════════════════════════════════════════
# ADDRESS FAST-PATH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _is_address_string(text: str) -> bool:
    """Return True if text looks like a house/street address rather than a place name."""
    return bool(_ADDRESS_PATTERN.search(text))


def _extract_city_from_address(text: str) -> Optional[str]:
    """
    Extract the city/area token from an address string.

    Examples:
      "No 45/B, Kandy Road, Kelaniya"  →  "Kelaniya"
      "45, කැලණිය මාවත, කැලණිය"       →  "කැලණිය"
      "No 12, Kelaniya Road"           →  "Kelaniya"  (stripped from street name)
      "54, வீதி, களனி"                 →  "களனி"

    Returns None if no city candidate can be extracted.
    """
    parts = [p.strip() for p in re.split(r'[,;]', text) if p.strip()]

    candidates = []
    for part in parts:
        # Skip house-number tokens ("No 45", "45/B", bare digits)
        if re.match(r'^\s*(?:No\.?\s*)?\d+[/\w]*\s*$', part, re.IGNORECASE):
            continue
        if re.match(r'^\s*(?:No\.?\s*)?\d+', part, re.IGNORECASE) and len(part) < 10:
            continue

        # If part contains a street keyword, strip it and use the remainder
        stripped = _STREET_KEYWORDS.sub('', part).strip(' ,')
        if stripped and len(stripped) >= 3 and stripped != part:
            candidates.append(stripped)
        elif not _STREET_KEYWORDS.search(part) and len(part) >= 3:
            candidates.append(part)

    # Prefer the last candidate (rightmost = broadest geographic scope)
    return candidates[-1] if candidates else None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RESOLUTION FUNCTION — 6-stage chain + address fast-path
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_location(text: str, use_nominatim: bool = False, db=None) -> dict:
    """
    Resolution chain:
      A:   Exact match
      A.5: Address fast-path (house/street → Google Maps or Nominatim directly)
      B:   Fuzzy match
      C:   Landmark pattern
      D:   Hierarchical parse
      E:   Google Maps Geocoding API  (if GOOGLE_MAPS_API_KEY is set)
      E.5: Nominatim fallback         (if no Google key, or Google failed)
      F:   Unresolved
    """
    if not text or not text.strip():
        return dict(_NO_LOCATION)

    # Stage A: Exact match
    ex = lookup_exact(text)
    if ex:
        e = GAZETTEER[ex]
        return _make_result(ex, e, "exact", 100)

    # Stage A.5: Address fast-path
    # Strings like "No 45/B, Kandy Road, Kelaniya" will never match fuzzy or
    # landmark stages. Route them to Google Maps (or Nominatim) for precise
    # street-level resolution. Results are cached — no slowdown after first run.
    if _is_address_string(text):
        geo = google_maps_resolve(text, db=db) or nominatim_resolve(text, db=db)
        if geo:
            return geo
        # Both APIs offline/failed — extract city portion and resolve that
        city = _extract_city_from_address(text)
        if city:
            ex2 = lookup_exact(city)
            if ex2:
                e = GAZETTEER[ex2]
                return _make_result(ex2, e, "address_city_exact", 82,
                                    "address_city_fallback", original_address=text)
            fz2 = lookup_fuzzy(city, 78)
            if fz2:
                e = GAZETTEER[fz2[0]]
                return _make_result(fz2[0], e, "address_city_fuzzy", fz2[1] * 0.85,
                                    "address_city_fallback", original_address=text)

    # Detect compound strings early — if separators present, try hierarchical
    # BEFORE fuzzy to get the most specific segment (GN division > town)
    has_separator = bool(_HIER_SPLIT.search(text))
    if has_separator:
        hp = hierarchical_resolve(text)
        if hp:
            return hp

    # Stage B: Fuzzy match
    fz = lookup_fuzzy(text)
    if fz:
        e = GAZETTEER[fz[0]]
        return _make_result(fz[0], e, "fuzzy", fz[1], "fuzzy_location")

    # Stage C: Landmark pattern
    lm = lookup_landmark(text)
    if lm:
        e = GAZETTEER[lm[0]]
        return _make_result(lm[0], e, "landmark", lm[1], "landmark_location")

    # Stage D: Hierarchical parse (non-separator cases, e.g. long location strings)
    if not has_separator:
        hp = hierarchical_resolve(text)
        if hp:
            return hp

    # Stage E: Google Maps (for place names that passed all local stages)
    if _GMAPS_API_KEY:
        geo = google_maps_resolve(text, db=db)
        if geo:
            return geo

    # Stage E.5: Nominatim fallback (opt-in, used when no Google key or Google failed)
    if use_nominatim or not _GMAPS_API_KEY:
        nom = nominatim_resolve(text, db=db)
        if nom:
            return nom

    # Stage F: Unresolved
    return dict(_UNRESOLVED)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_batch(reports: list, use_nominatim: bool = False, db=None) -> list:
    """Resolve all reports. Returns same list with lat/lng/location_resolved added."""
    for r in reports:
        res = resolve_location(r.get("location_raw", ""),
                               use_nominatim=use_nominatim, db=db)
        r["location_resolved"] = res
        r["lat"] = res["lat"]
        r["lng"] = res["lng"]
        if res["flag"] and res["flag"] not in r.get("flags", []):
            r.setdefault("flags", []).append(res["flag"])
    return reports


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Build index on import
_gn_count = _build_index()
_load_geocoding_cache()

_geo_source = "Google Maps" if _GMAPS_API_KEY else "Nominatim (OSM)"
if _gn_count > 0:
    print(f"  [Gazetteer] {len(GAZETTEER)} entries ({31} towns + {_gn_count} GN variants) | geocoder: {_geo_source}")
else:
    print(f"  [Gazetteer] {len(GAZETTEER)} town entries (run gn_extractor for GN divisions) | geocoder: {_geo_source}")
