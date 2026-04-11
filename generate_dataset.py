"""
C4 — Disaster Report Dataset Generator
========================================
Generates 1000 simulated reports:
  - 500 FLOOD reports (250 voice + 250 SMS)
  - 500 FIRE  reports (250 voice + 250 SMS)
Languages: Sinhala Unicode (~55%) + Romanized Sinhala (~30%) + Tamil (~15%)
Provinces: Western (Colombo, Gampaha, Kalutara) + Central (Kandy, Ratnapura)

Each report simulates structured JSON from:
  - Component 1 (C1): Voice call pipeline
  - Component 3 (C3): SMS pipeline

Reports are grouped into real-world INCIDENTS with duplicates,
conflicting data, cross-variant location names, and edge cases.

Scenario types:
  mass_report            — 10-12 reports about same incident
  adjacent_critical      — two events 0.5-2 km apart (MUST NOT merge)
  geographically_near    — adjacent GN divisions, same type, different incident
  third_party            — friend/family member reporting on someone else's behalf
  conflicting_counts     — same incident, wildly different people counts
  long_gap               — same event reported 2-4 hours apart
  standard               — normal incidents, various urgency levels
  noise                  — isolated single reports
  cross_lingual_duplicate — same incident, same time, guaranteed Sinhala+Tamil pair
  victim_address         — victim gives exact house address instead of town name
  location_unknown       — caller cannot state their location at all
  minimal_message        — 1-2 word message, no location, no details (e.g. "help")
  same_street_neighbors  — two houses 10-50 m apart, identical geocoded coords, MUST NOT merge

GN-level precision: uses actual GN division centroids from lka_admin4.geojson
for sub-city coordinate variation (reports for same town get different lat/lng
based on their GN division, typically 0.3-2 km apart).
"""

import json
import random
import os
from datetime import datetime, timedelta, timezone
from collections import Counter
from pathlib import Path

random.seed(42)

# ── Load GN divisions for sub-city coordinate precision ──
_GN_BY_DS: dict = {}  # ds_division -> list of {name_en, lat, lng, ...}
_GN_PATH = Path(__file__).parent / "data" / "gn_divisions.json"

def _load_gn_for_reports():
    """Load GN divisions grouped by DS division name for coordinate jitter."""
    if not _GN_PATH.exists():
        return
    with open(_GN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for gn in data.get("gn_divisions", []):
        ds = gn.get("ds_division", "")
        if ds:
            _GN_BY_DS.setdefault(ds, []).append(gn)

_load_gn_for_reports()


def _get_nearby_gn(loc_key, loc_data):
    """Get a random GN division near this location for coordinate precision."""
    # Try matching DS division by location key name
    for ds_name in [loc_key, loc_key.replace("_City", "").replace("_Town", ""),
                    loc_data.get("district", "")]:
        gns = _GN_BY_DS.get(ds_name, [])
        if gns:
            return random.choice(gns)
    # Fallback: return None (use town center)
    return None

# ============================================================
# 1. LOCATIONS — Sinhala Unicode + Romanized Sinhala variants
# ============================================================
LOCATIONS = {
    # ---- COLOMBO DISTRICT ----
    "Kelaniya": {
        "lat": 6.9541, "lng": 79.9196, "district": "Colombo", "province": "Western",
        "sinhala": ["කැලණිය", "කැලණි", "කැලණිය පාලම ළඟ"],
        "romanized": ["Kelaniya", "Kelaniye", "Kelani", "Kelaniya pala"],
        "landmark": ["කැලණිය පන්සල ළඟ", "කැලණිය පාලම පසු කරලා", "කැලණිය හන්දිය"]
    },
    "Kaduwela": {
        "lat": 6.9307, "lng": 79.9831, "district": "Colombo", "province": "Western",
        "sinhala": ["කඩුවෙල", "කඩුවෙල නගරය"],
        "romanized": ["Kaduwela", "Kaduwala"],
        "landmark": ["කඩුවෙල පාලම ළඟ", "කඩුවෙල හන්දිය"]
    },
    "Kolonnawa": {
        "lat": 6.9297, "lng": 79.8853, "district": "Colombo", "province": "Western",
        "sinhala": ["කොළොන්නාව", "කොලොන්නාව"],
        "romanized": ["Kolonnawa", "Kolonnawe"],
        "landmark": ["කොළොන්නාව ඇළ ළඟ", "කොළොන්නාව තෙල් පිරිපහදුව පසුපස"]
    },
    "Wellampitiya": {
        "lat": 6.9388, "lng": 79.8881, "district": "Colombo", "province": "Western",
        "sinhala": ["වැල්ලම්පිටිය", "වැල්ලම්පිටි"],
        "romanized": ["Wellampitiya", "Wellampitiye"],
        "landmark": ["වැල්ලම්පිටිය පොළ ළඟ", "වැල්ලම්පිටිය පාර"]
    },
    "Mattakkuliya": {
        "lat": 6.9536, "lng": 79.8756, "district": "Colombo", "province": "Western",
        "sinhala": ["මට්ටක්කුලිය"],
        "romanized": ["Mattakkuliya", "Mattakuliya"],
        "landmark": ["මට්ටක්කුලිය නිවාස ළඟ", "මට්ටක්කුලිය ඇළ අස"]
    },
    "Grandpass": {
        "lat": 6.9458, "lng": 79.8697, "district": "Colombo", "province": "Western",
        "sinhala": ["ග්‍රෑන්ඩ්පාස්", "ග්‍රෑන්ඩ් පාස්"],
        "romanized": ["Grandpass", "Grand Pass"],
        "landmark": ["ග්‍රෑන්ඩ්පාස් පාර ළඟ", "ග්‍රෑන්ඩ්පාස් නිවාස"]
    },
    "Dematagoda": {
        "lat": 6.9326, "lng": 79.8741, "district": "Colombo", "province": "Western",
        "sinhala": ["දෙමටගොඩ"],
        "romanized": ["Dematagoda"],
        "landmark": ["දෙමටගොඩ දුම්රිය ළඟ", "දෙමටගොඩ ඇළ"]
    },
    "Battaramulla": {
        "lat": 6.8990, "lng": 79.9183, "district": "Colombo", "province": "Western",
        "sinhala": ["බත්තරමුල්ල"],
        "romanized": ["Battaramulla"],
        "landmark": ["පාර්ලිමේන්තු පාර ළඟ", "බත්තරමුල්ල හන්දිය"]
    },
    "Hanwella": {
        "lat": 6.9012, "lng": 80.0851, "district": "Colombo", "province": "Western",
        "sinhala": ["හංවැල්ල"],
        "romanized": ["Hanwella", "Hanwelle"],
        "landmark": ["හංවැල්ල පාලම ළඟ", "හංවැල්ල නගරය"]
    },
    "Avissawella": {
        "lat": 6.9533, "lng": 80.2100, "district": "Colombo", "province": "Western",
        "sinhala": ["අවිස්සාවේල්ල"],
        "romanized": ["Avissawella", "Avissawelle"],
        "landmark": ["අවිස්සාවේල්ල නගරය ළඟ"]
    },
    # ---- GAMPAHA DISTRICT ----
    "Ja-Ela": {
        "lat": 7.0757, "lng": 79.8916, "district": "Gampaha", "province": "Western",
        "sinhala": ["ජා-ඇල", "ජාඇල"],
        "romanized": ["Ja-Ela", "Ja Ela", "Jaela"],
        "landmark": ["ජා-ඇල නගරය ළඟ", "ජා-ඇල හන්දිය"]
    },
    "Kadana": {
        "lat": 7.0009, "lng": 80.0251, "district": "Gampaha", "province": "Western",
        "sinhala": ["කදාන", "කදාන නගරය"],
        "romanized": ["Kadana", "Kadane"],
        "landmark": ["කදාන පාලම ළඟ"]
    },
    "Biyagama": {
        "lat": 6.9618, "lng": 79.9834, "district": "Gampaha", "province": "Western",
        "sinhala": ["බියගම", "බියගම කලාපය"],
        "romanized": ["Biyagama", "Biyagame"],
        "landmark": ["බියගම කර්මාන්ත කලාපය ළඟ"]
    },
    "Wattala": {
        "lat": 6.9896, "lng": 79.8918, "district": "Gampaha", "province": "Western",
        "sinhala": ["වත්තල", "වත්තල නගරය"],
        "romanized": ["Wattala", "Wattale", "Watthala"],
        "landmark": ["වත්තල හන්දිය ළඟ", "වත්තල පාලම"]
    },
    "Negombo": {
        "lat": 7.2083, "lng": 79.8358, "district": "Gampaha", "province": "Western",
        "sinhala": ["මීගමුව", "මීගමු"],
        "romanized": ["Negombo", "Meegamuwa", "Migamuwa"],
        "landmark": ["මීගමුව කලපුව ළඟ", "මීගමුව මාළු වෙළඳපොළ"]
    },
    "Gampaha_Town": {
        "lat": 7.0917, "lng": 80.0003, "district": "Gampaha", "province": "Western",
        "sinhala": ["ගම්පහ", "ගම්පහ නගරය"],
        "romanized": ["Gampaha", "Gamapaha"],
        "landmark": ["ගම්පහ රෝහල ළඟ", "ගම්පහ නගර මධ්‍යයේ"]
    },
    # ---- KALUTARA DISTRICT ----
    "Kalutara": {
        "lat": 6.5854, "lng": 79.9607, "district": "Kalutara", "province": "Western",
        "sinhala": ["කළුතර", "කළුතර නගරය"],
        "romanized": ["Kalutara", "Kaluthara"],
        "landmark": ["කළුතර පාලම ළඟ", "කළුතර බෝධිය ළඟ"]
    },
    "Beruwala": {
        "lat": 6.4789, "lng": 79.9826, "district": "Kalutara", "province": "Western",
        "sinhala": ["බේරුවල", "බේරුවළ"],
        "romanized": ["Beruwala", "Beruwale"],
        "landmark": ["බේරුවල වරාය ළඟ", "බේරුවල මාළු ගම"]
    },
    "Horana": {
        "lat": 6.7153, "lng": 80.0622, "district": "Kalutara", "province": "Western",
        "sinhala": ["හොරණ", "හොරණ නගරය"],
        "romanized": ["Horana", "Horane"],
        "landmark": ["හොරණ හන්දිය ළඟ", "හොරණ රෝහල ළඟ"]
    },
    "Panadura": {
        "lat": 6.7136, "lng": 79.9044, "district": "Kalutara", "province": "Western",
        "sinhala": ["පානදුර", "පානදුර නගරය"],
        "romanized": ["Panadura", "Panadure"],
        "landmark": ["පානදුර පාලම ළඟ", "පානදුර ගඟ අස"]
    },
    "Bandaragama": {
        "lat": 6.6977, "lng": 80.0388, "district": "Kalutara", "province": "Western",
        "sinhala": ["බණ්ඩාරගම"],
        "romanized": ["Bandaragama", "Bandaregama"],
        "landmark": ["බණ්ඩාරගම නගරය ළඟ", "බණ්ඩාරගම කුඹුරු"]
    },
    # ---- KANDY DISTRICT ----
    "Kandy_City": {
        "lat": 7.2906, "lng": 80.6337, "district": "Kandy", "province": "Central",
        "sinhala": ["මහනුවර", "මහනුවර නගරය", "නුවර"],
        "romanized": ["Kandy", "Mahanuwara", "Nuwara"],
        "landmark": ["මහනුවර වැව ළඟ", "දළදා මාළිගාව ළඟ", "මහනුවර නගර මධ්‍යයේ"]
    },
    "Peradeniya": {
        "lat": 7.2590, "lng": 80.5964, "district": "Kandy", "province": "Central",
        "sinhala": ["පේරාදෙණිය"],
        "romanized": ["Peradeniya", "Peradeniye"],
        "landmark": ["පේරාදෙණිය විශ්ව විද්‍යාලය ළඟ", "පේරාදෙණිය උද්‍යානය"]
    },
    "Katugastota": {
        "lat": 7.3131, "lng": 80.6247, "district": "Kandy", "province": "Central",
        "sinhala": ["කටුගස්තොට"],
        "romanized": ["Katugastota", "Katugasthota"],
        "landmark": ["කටුගස්තොට පාලම ළඟ", "කටුගස්තොට නගරය"]
    },
    "Gampola": {
        "lat": 7.1642, "lng": 80.5770, "district": "Kandy", "province": "Central",
        "sinhala": ["ගම්පොල", "ගම්පොළ"],
        "romanized": ["Gampola", "Gampole"],
        "landmark": ["ගම්පොල පාලම ළඟ", "ගම්පොල නගර මධ්‍යයේ"]
    },
    "Kadugannawa": {
        "lat": 7.2542, "lng": 80.5239, "district": "Kandy", "province": "Central",
        "sinhala": ["කඩුගන්නාව"],
        "romanized": ["Kadugannawa", "Kadugannave"],
        "landmark": ["කඩුගන්නාව පාස් එක ළඟ", "කඩුගන්නාව දුම්රිය"]
    },
    # ---- RATNAPURA DISTRICT ----
    "Ratnapura_City": {
        "lat": 6.6828, "lng": 80.3992, "district": "Ratnapura", "province": "Central",
        "sinhala": ["රත්නපුර", "රත්නපුර නගරය"],
        "romanized": ["Ratnapura", "Rathnapura"],
        "landmark": ["රත්නපුර නගරය ළඟ", "රත්නපුර මැණික් වෙළඳපොළ"]
    },
    "Eheliyagoda": {
        "lat": 6.8438, "lng": 80.2708, "district": "Ratnapura", "province": "Central",
        "sinhala": ["ඇහැලියගොඩ"],
        "romanized": ["Eheliyagoda", "Eheliyagode"],
        "landmark": ["ඇහැලියගොඩ නගරය ළඟ", "ඇහැලියගොඩ රෝහල ළඟ"]
    },
    "Kuruwita": {
        "lat": 6.7746, "lng": 80.3662, "district": "Ratnapura", "province": "Central",
        "sinhala": ["කුරුවිට"],
        "romanized": ["Kuruwita", "Kuruwite"],
        "landmark": ["කුරුවිට පාලම ළඟ", "කුරුවිට නගරය"]
    },
    "Pelmadulla": {
        "lat": 6.6197, "lng": 80.4633, "district": "Ratnapura", "province": "Central",
        "sinhala": ["පැල්මඩුල්ල"],
        "romanized": ["Pelmadulla", "Pelmadulle"],
        "landmark": ["පැල්මඩුල්ල හන්දිය ළඟ"]
    },
    "Balangoda": {
        "lat": 6.6467, "lng": 80.7003, "district": "Ratnapura", "province": "Central",
        "sinhala": ["බලංගොඩ"],
        "romanized": ["Balangoda", "Balangode"],
        "landmark": ["බලංගොඩ බස් නැවතුම ළඟ", "බලංගොඩ රෝහල"]
    },
}

# ============================================================
# 2. SINHALA PHRASES BY DISASTER TYPE
# ============================================================

FLOOD_PHRASES = {
    "water_level": [
        "ජලය පපුව දක්වා ඉහළ ගිහිං",
        "වතුර බෙල්ල මට්ටමට",
        "ජලය වහළ මට්ටමට ළඟා වෙනවා",
        "වතුර ගේට මට්ටමට ආවා",
        "ජලය කකුල් මට්ටමට",
        "වතුර ඉණ මට්ටමට",
        "ගෙට වතුර ආවා",
        "පහළ මහල සම්පූර්ණයෙන්ම වතුරෙන් පිරිලා",
        "මහ ගංවතුරක් ආවා",
        "වතුර ගහපු ගේ ඇතුලට ආවා",
        "වතුර ඉක්මනට නැගෙනවා",
        "ගංවතුර නිසා මාර්ගය යටවෙලා",
    ],
    "help_needed": [
        "උදව් අවශ්‍යයි කරුණාකර ඉක්මනට එන්න",
        "අපිට එළියට යන්න බැහැ උදව් කරන්න",
        "බේරගන්න ඕන මිනිස්සු ඉන්නවා ගෙදර",
        "කරුණාකර බෝට්ටුවක් එවන්න බේරගන්න",
        "ළමයි ඉන්නවා කුඩා ළමයි බේරගන්න ඕන",
        "හදිසි ආධාර අවශ්‍යයි",
        "අපට ආහාර සහ ජලය අවශ්‍යයි",
        "වැඩිහිටි අය ඉන්නවා ගෙදර ගෙනියන්න බැහැ",
        "කරුණාකර යාත්‍රාවක් එවන්න",
    ],
    "situation": [
        "ගෙදර වහළේ උඩ ඉන්නවා",
        "වහළේ උඩ නැග්ගා",
        "ගෙවල් ගිලුණා",
        "මාර්ගය ජලයෙන් යටවෙලා",
        "පාලම උඩින් වතුර යනවා",
        "විදුලිය කැපිලා",
        "ගංවතුර නිසා ගස් කඩිලා",
        "වතුර ඇතුලට ආවා ගෙදරට",
        "මුළු ප්‍රදේශයම යටවෙලා",
    ]
}

FIRE_PHRASES = {
    "fire_status": [
        "ගිනි ගෙන තියෙනවා",
        "ගින්නක් ඇතිවෙලා",
        "ගින්දර ලොකුයි වැඩි වෙනවා",
        "ගෙදර ගිනි අරන් තියෙනවා",
        "ගින්නෙන් ගෙවල් ගිනිගන්නවා",
        "ගිනි දැල් පැතිරෙනවා",
        "ගින්න පාලනය කරන්න බැහැ",
        "ගෙදර සම්පූර්ණයෙන්ම ගිනිගත්තා",
        "කර්මාන්ත ශාලාවක ගිනි ඇතිවෙලා",
        "දුම නැගෙනවා ගොඩක්",
        "ගිනි පුළුල් වෙනවා අසල ගෙවල්වලට",
        "ගෑස් ලීක් වෙලා ගින්නක් ඇතිවෙලා",
    ],
    "help_needed": [
        "ගිනි නිවන බල්ලන් එවන්න ඉක්මනට",
        "ගිනි නිවීම් රථයක් එවන්න කරුණාකර",
        "මිනිස්සු හිරවෙලා ඉන්නවා ඇතුලේ",
        "ගින්නෙන් බේරගන්න ඕන ළමයි ඉන්නවා",
        "ගිනි නිවීම් ආධාර අවශ්‍යයි වහාම",
        "ඉක්මනට එන්න ගෙවල් ගිනිගන්නවා",
        "ගින්නට අසුවෙලා ඉන්නවා මිනිස්සු",
        "ගිනි නිවන්න වතුර නැහැ",
        "ගිනි නිවීම් සේවය කැඳවන්න",
    ],
    "situation": [
        "ගෙදර දැනට ගිනි ගෙන තියෙනවා",
        "පළමු මහල ගිනිගත්තා",
        "මුළු ගොඩනැගිල්ලම ගිනිගෙන",
        "ගිනි පාලනය කරන්න බැරි තත්ත්වයක්",
        "ගෙවල් තුනක් ගිනිගත්තා",
        "වහළේ ගිනිගෙන තියෙනවා",
        "විදුලි කාන්දුවක් නිසා ගින්නක්",
        "ගිනි දුම නිසා හුස්ම ගන්න බැහැ",
        "අසල්වාසීන් ඉවත් කරනවා",
    ]
}

ROMANIZED_FLOOD = {
    "water_level": [
        "wathura papuwa dakwa", "jalaya bella mattamata",
        "wathure geta aawa", "pahala mahala wathuren pirilaa",
        "maha gangwathurk aawa", "wathura kakul mattamata",
        "wathura ina mattamata", "wathura ikmanata nagenawa",
    ],
    "help_needed": [
        "udaw onaa karunakara enna", "apita eliyata yanna baha",
        "beraganna ona minissu innawa", "karunakara boat ekak ewanna",
        "lamayi innawa kudaa lamayi beraganna", "hadisi adhar awashyai",
    ],
    "situation": [
        "wahale uda innawa", "ge giluuna wathuren",
        "maargaya jalyen yata wela", "viduliya kaepila",
        "paalam udin wathura yanawa",
    ]
}

# ============================================================
# TAMIL PHRASES — Sri Lankan Tamil (cross-lingual test pairs)
# These describe the same disaster events as the Sinhala phrases above.
# LaBSE embeds these in the same semantic space — the ML model then
# learns the interaction pattern, which fixed weights cannot.
# ============================================================

TAMIL_FLOOD = {
    "water_level": [
        "தண்ணீர் மார்பு அளவிற்கு வந்துவிட்டது",
        "வெள்ளம் கழுத்து வரை உயர்ந்துவிட்டது",
        "வீட்டிற்கு தண்ணீர் வந்துவிட்டது",
        "கீழ் மாடி முழுவதும் தண்ணீரில் மூழ்கிவிட்டது",
        "பெரும் வெள்ளம் வந்துவிட்டது",
        "தண்ணீர் இடுப்பு வரை உள்ளது",
        "வீட்டின் கூரை வரை தண்ணீர் வருகிறது",
    ],
    "help_needed": [
        "உதவி தேவை இப்போதே வாருங்கள்",
        "எங்களால் வெளியே போக முடியவில்லை உதவுங்கள்",
        "காப்பாற்ற வேண்டிய மக்கள் வீட்டில் உள்ளனர்",
        "படகு அனுப்புங்கள் தயவுசெய்து",
        "சிறு குழந்தைகள் உள்ளனர் காப்பாற்றுங்கள்",
        "அவசர உதவி தேவை",
    ],
    "situation": [
        "வீட்டின் கூரை மேல் இருக்கிறோம்",
        "வீடுகள் மூழ்கிவிட்டன",
        "சாலை தண்ணீரில் மூழ்கிவிட்டது",
        "மின்சாரம் துண்டிக்கப்பட்டது",
        "பாலத்தின் மேல் தண்ணீர் ஓடுகிறது",
    ]
}

TAMIL_FIRE = {
    "fire_status": [
        "தீ பற்றிக்கொண்டுள்ளது",
        "தீ விபத்து ஏற்பட்டுள்ளது",
        "தீ பெரிதாக பரவுகிறது",
        "வீட்டில் தீ பிடித்துவிட்டது",
        "தீ கட்டுப்படுத்த முடியவில்லை",
        "புகை மிகவும் அதிகமாக உள்ளது",
        "நெருப்பு தீவிரமாக எரிகிறது",
    ],
    "help_needed": [
        "தீயணைப்பு வாகனம் அனுப்புங்கள் உடனடியாக",
        "மக்கள் உள்ளே சிக்கிக்கொண்டுள்ளனர்",
        "தீயிலிருந்து காப்பாற்ற வேண்டிய குழந்தைகள் உள்ளனர்",
        "உடனடி தீயணைப்பு உதவி தேவை",
        "விரைவாக வாருங்கள் வீடுகள் எரிகின்றன",
    ],
    "situation": [
        "முதல் மாடி தீ பிடித்துவிட்டது",
        "கட்டிடம் முழுவதும் தீ பரவிவிட்டது",
        "மூன்று வீடுகளில் தீ பிடித்துவிட்டது",
        "புகையினால் சுவாசிக்க கஷ்டமாக உள்ளது",
        "அடுத்த வீட்டிற்கும் தீ பரவிவிட்டது",
    ]
}

# Tamil location names for key Sri Lankan cities
TAMIL_LOCATIONS = {
    "Kelaniya":       ["களனி", "களணி நகரம்"],
    "Kaduwela":       ["கடுவெல"],
    "Kolonnawa":      ["கொலொன்னாவ"],
    "Wellampitiya":   ["வெல்லம்பிட்டிய"],
    "Negombo":        ["நீர்கொழும்பு", "நீர்கொழும்பு நகரம்"],
    "Kalutara":       ["களுத்துறை", "களுத்துறை நகரம்"],
    "Panadura":       ["பாணந்துறை"],
    "Kandy_City":     ["கண்டி", "கண்டி நகரம்", "மாநகரம்"],
    "Ratnapura_City": ["இரத்தினபுரி", "இரத்தினபுர நகரம்"],
    "Gampaha":        ["கம்பஹா"],
    "Ja-Ela":         ["ஜா-ஏல"],
    "Horana":         ["ஹொரண"],
    "Peradeniya":     ["பேராதெனிய"],
    "Katugastota":    ["கட்டுகஸ்தொட"],
    "Gampola":        ["கம்போல"],
    "Avissawella":    ["அவிஸாவெல்ல"],
    "Hanwella":       ["ஹன்வெல்ல"],
    "Battaramulla":   ["பத்தரமுல்ல"],
    "Dematagoda":     ["தெமடகொட"],
    "Grandpass":      ["கிராண்ட்பாஸ்"],
    "Wattala":        ["வத்தல"],
    "Biyagama":       ["பியகம"],
    "Beruwala":       ["பேருவல"],
    "Bandaragama":    ["பண்டாரகம"],
    "Eheliyagoda":    ["எஹெலியகொட"],
    "Kuruwita":       ["குருவிட"],
    "Pelmadulla":     ["பெல்மடுல்ல"],
    "Balangoda":      ["பலங்கொட"],
    "Kadana":         ["கடான"],
    "Mattakkuliya":   ["மட்டக்குளிய"],
    "Kadugannawa":    ["கடுகண்ணாவ"],
}

# ============================================================
# ROMANIZED TAMIL LOCATIONS — Tamil place names in Latin script
# Very common in text messages from young Tamil speakers in SL
# ============================================================
ROMANIZED_TAMIL_LOCATIONS = {
    # Western Province
    "Kelaniya":       ["Kelaniya", "Kelani"],
    "Kaduwela":       ["Kaduvela", "Kaduwela"],
    "Kolonnawa":      ["Kolonnawa", "Kolonnawa area"],
    "Wellampitiya":   ["Wellampitiya", "Wellampitiyaa"],
    "Negombo":        ["Neerkozhumbu", "Negombo"],
    "Kalutara":       ["Kaluthurai", "Kalutara"],
    "Panadura":       ["Paananturai", "Panadura"],
    "Wattala":        ["Vatthalai", "Wattala"],
    "Battaramulla":   ["Battaramulla", "Battaramulle"],
    "Ja-Ela":         ["Saa-Eelam", "Ja-Ela"],
    "Horana":         ["Horanai", "Horana"],
    "Beruwala":       ["Peruwala", "Beruwala"],
    "Bandaragama":    ["Pantaaragama", "Bandaragama"],
    # Central Province
    "Kandy_City":     ["Kanndi", "Maahanuvara"],
    "Peradeniya":     ["Peraatheniyaa", "Peradeniya"],
    "Katugastota":    ["Katukasthottai", "Katugastota"],
    "Gampola":        ["Kampoolaa", "Gampola"],
    "Ratnapura_City": ["Iraththinapuri", "Ratnapura"],
    "Kuruwita":       ["Kuruvitai", "Kuruwita"],
    "Eheliyagoda":    ["Eheliyagoda", "Eheliyakooda"],
}

# ============================================================
# ROMANIZED TAMIL PHRASES — Tamil written in Latin script
# Very common in Sri Lankan Tamil SMS and informal communication
# ============================================================
ROMANIZED_TAMIL_FLOOD = {
    "water_level": [
        "veedu varaikkum thanneer vandhuchu",
        "vellam kaluthuvare vandhuchu engal veedu",
        "keezh maadi ellaam thanneerla moozhkiduchu",
        "thanneer maarvaa varaikkum irukku veetukkulle",
        "peria vellam vandhuchu thanneer naala naala erudhu",
        "iduppu varaikkum thanneer vandhuchu tholaikala mudiyala",
        "veettu koorai varaikkum thanneer erungudhu",
    ],
    "help_needed": [
        "udavi thevai ungalukku kaathaama irunga please",
        "engaloda tholaikala mudiyala vellathula",
        "kaapaatranum ingae makkal irukaanga",
        "paadagu anupungal please engalai kaapaatrungal",
        "pillaigal irukaanga siriya pillaigal kaapaatrungal",
        "avasara udavi thevai ungalukku unarka veendum",
        "naanga vazhikku pooga mudiyala please varungal",
    ],
    "situation": [
        "veettu koorai mela irukkoam naanga",
        "veedukal moozhkipochu ellaam",
        "vazhiya thanneerla moozhkiduchu",
        "meensaaraangu thundikappattuchu",
        "palathoda mela thanneer oadugiradhu",
        "poova keezhae kochi maram vizhunthuchu vellathula",
    ],
}

ROMANIZED_TAMIL_FIRE = {
    "fire_status": [
        "veedu eriyudhu peria thee pidichirukkum",
        "thee pidichirukku kattuppaduthala mudiyala",
        "periya thee parariyudhu asamuthu veedukalukku",
        "veettu muzhuvathum thee paravuduchu",
        "puthiru migavum adhigama irukku swaasikka mudiyala",
        "gas likkaa thee moooti erinduchu",
        "mudhal maadi muzhuvathum eriyudhu",
    ],
    "help_needed": [
        "thee aanaikka vaandam anupungal uadanae",
        "makkal ullae maattikirukkaanga kaapaatrungal",
        "pillaikku thee pidichirukku vegam varungal",
        "aasai udanae thee aanaikka team varanum",
        "vegama varungal veedukal eriyudhu",
        "thanneer illai thee aanaikka please help",
    ],
    "situation": [
        "mudhal maadi thee pidichirukku",
        "mooonu veedukalula thee paravuduchu",
        "kattidaththai muzhuvathum thee pidichuduchu",
        "puthiru karanamaaga swaasikka kastama irukku",
        "aaandavane thee vera veedukullavum pararindhu",
        "viduthalaip paathai thee paravuthal thadukkappadavilla",
    ],
}

ROMANIZED_TAMIL_THIRD_PARTY = [
    "en nanban veedu thanneerla moozhkiduchu",
    "en atthai veedu thee pidichirukku uadanae varungal",
    "en thambi kudumbam maattikirukkaanga vellathula",
    "en akka veedu adutha vella vandhuchu",
    "en nanban sonna udavi venum nu",
    "engal uruvinar aasaiyaana nila irukkaanga",
]


# ============================================================
# VICTIM PANIC MESSAGES — raw 1-5 word cries for help
# The victim is too panicked / has no time to give details.
# Often just a location + disaster word, or just "help".
# These are the HARDEST cases for the system: no context,
# minimal signal — the operator must infer everything.
#
# Sinhala, Romanized Sinhala, Tamil, Romanized Tamil variants
# covering every location in Western + Central provinces.
# ============================================================

def _make_panic_messages(loc_key, loc_data, incident_type):
    """
    Generate realistic panic message variants for a given location.
    Returns a pool of short panic strings across all 4 languages.
    """
    # Location name tokens per language
    si_names  = loc_data.get("sinhala", [loc_key])
    ro_names  = loc_data.get("romanized", [loc_key])
    ta_names  = TAMIL_LOCATIONS.get(loc_key, [loc_key.replace("_City","").replace("_Town","")])
    rt_names  = ROMANIZED_TAMIL_LOCATIONS.get(loc_key, [loc_key.replace("_City","").replace("_Town","")])

    si  = random.choice(si_names)
    ro  = random.choice(ro_names)
    ta  = random.choice(ta_names)
    rt  = random.choice(rt_names)

    if incident_type == "flood":
        si_disaster  = random.choice(["ගංවතුර", "ජලය", "ගිලෙනවා", "වතුර"])
        ro_disaster  = random.choice(["gangwatura", "watura", "flood", "jalaya"])
        ta_disaster  = random.choice(["வெள்ளம்", "தண்ணீர்", "வெள்ளப்பெருக்கு"])
        rt_disaster  = random.choice(["vellam", "thanneer", "flood"])
    elif incident_type == "fire":
        si_disaster  = random.choice(["ගිනි", "ගින්නෙ", "ගිනිගෙන"])
        ro_disaster  = random.choice(["gini", "ginnk", "fire"])
        ta_disaster  = random.choice(["தீ", "நெருப்பு", "தீவிபத்து"])
        rt_disaster  = random.choice(["thee", "nerruppu", "fire"])
    else:
        si_disaster  = random.choice(["ගිනි", "ගංවතුර", "උදව්"])
        ro_disaster  = random.choice(["gini", "gangwatura", "help"])
        ta_disaster  = random.choice(["தீ", "வெள்ளம்"])
        rt_disaster  = random.choice(["thee", "vellam"])

    pool = [
        # just "help" / "SOS" — no location at all
        "help", "SOS", "help!", "help me", "please help",
        "උදව්", "හදිසි", "බේරගන්න", "ගොඩ ගන්න",
        "udaw", "hadisi", "beraganna",
        "உதவி", "காப்பாற்றுங்கள்", "வாருங்கள்",
        "udavi", "kaapaatrungal", "varungal",
        # "help [location]"
        f"help {ro}",
        f"help! {ro}",
        f"help {ro} please",
        f"udaw {ro}",
        f"udaw {si}",
        f"udavi {rt}",
        f"udavi {ta}",
        f"உதவி {ta}",
        f"உதவி தேவை {ta}",
        # "[disaster] [location]"
        f"{si_disaster} {si}",
        f"{si_disaster}! {si}",
        f"{ro_disaster} {ro}",
        f"{ro_disaster}! {ro}",
        f"{ta_disaster} {ta}",
        f"{ta_disaster}! {ta}",
        f"{rt_disaster} {rt}",
        f"{rt_disaster}! {rt}",
        # "[location] [disaster]"
        f"{ro} {ro_disaster}",
        f"{si} {si_disaster}",
        f"{ta} {ta_disaster}",
        f"{rt} {rt_disaster}",
        # "[location] help"
        f"{ro} help",
        f"{si} උදව්",
        f"{ta} உதவி",
        f"{rt} udavi",
        # ultra short location-only (caller can barely speak)
        f"{ro}!!",
        f"{si}!!",
    ]
    return pool

# ============================================================
# FULL CONVERSATION TEMPLATES — simulates actual voice call transcripts
# Stored in report["transcript"] for voice calls.
# Much richer than individual key_phrases.
# ============================================================
VOICE_CONVERSATIONS_FLOOD = {
    "sinhala": [
        "සාර් මමු කැළණිය ළඟ ඉන්නේ. ගෙදරට ගොඩක් වතුර ආවා. පපුව තරම් ජලය ඇතුලේ. අපිට ළමයි හතර දෙනෙකුත් ඉන්නවා. ගෙදරින් එළියට යන්න බැහැ. ඉක්මනින් බෝට්ටුවක් ගෙනියන්න කරුණාකර.",
        "අයියෝ උදව් කරන්න. හංවැල්ල ළඟ ගෙදර වතුරෙ. ගෙදරේ ඉඩ මහල සම්පූර්ණ ගිලිලා. ළමයි දෙන්නෙකුයි ආච්චිත් ඉන්නෙ. ආච්චිට් ඇවිද්ද ගන්න බැහැ. ඉක්මනට ගෙනිය ගන්නනෝ.",
        "කොළොන්නාවේ ඉන්නෙ මමු. ඇළ පිටාරිලා ගෙදරට ජලය ආවා. ඉණ මට්ටමට ජලය. මිනිස්සු ගාණ ගොඩයි ගෙදර. කෑම් නැහැ. ඉස්කෝල් ළමයිත් ඉන්නෙ. ජලය ඉක්මනට යන්නේ නැහැ. ආධාර ඕනේ.",
        "ජා-ඇල ළඟ ගංවතුරෙ. ජලය පපුව ඉහළට ඇවිත්. ගෙවල් ගාණකට වතුර ඇවිත්. අම්මා රෝගිනිය ගෙදරේ ඉන්නේ. ගෙනිය ගන්නනෝ. ගිනි නිවීම් ආධාර නැහැ. ජලය ඉහළ යනවා.",
    ],
    "romanized": [
        "sir api Kelaniya langa innawa. gedara watura wadi aawa. papuwa thrama jalaya aethule. lamayi hathara denekuth innawa. gedaren eliyata yanna baha. ikmanata boat ekak ewaanna.",
        "ayyo udaw karanna. Hanwella langa ge watrn. bim mahala sampurnayenma gilila. lamayi dennakyui aachchi innawa. ikmanata geniya gannno.",
        "Kolonnawa watura awa ela pitaarila. ina mattamata. minissu godai gedara. kama naha. jalaya ikmanata yenne naha. adhar oene karunakara.",
        "Ja-Ela gangwatura. jalaya papuwa ihalata. gewal gaankata watura. amma roginiyin gedara. geniya gannno please. jalaya ihala yanawa.",
    ],
    "tamil": [
        "ஐயா நான் கல்முனையில் இருக்கிறேன். எங்கள் வீட்டிற்கு தண்ணீர் வந்துவிட்டது. மார்பு அளவிற்கு தண்ணீர் இருக்கிறது. என் குடும்பத்தில் ஆறு பேர் இருக்கிறோம். வயதான பெற்றோரும் சிறு குழந்தைகளும் உள்ளனர். தயவுசெய்து படகு அனுப்புங்கள் மிகவும் அவசரம்.",
        "ஐயா திருகோணமலையில் வெள்ளம் வந்துவிட்டது. தண்ணீர் கழுத்து வரை உயர்ந்துவிட்டது. வீட்டில் நாங்கள் ஐந்து பேர் இருக்கிறோம். குழந்தைகள் கூரையில் ஏறியிருக்கிறோம். உடனடியாக உதவி அனுப்புங்கள்.",
        "சார் மட்டக்களப்பில் இருந்து பேசுகிறேன். ஆற்று வெள்ளம் வீட்டுக்கு வந்துவிட்டது. இடுப்பு அளவு தண்ணீர். வீட்டில் முதியோர் இருக்கிறார்கள் நகர முடியவில்லை. தயவுசெய்து படகு அனுப்புங்கள்.",
    ],
    "romanized_tamil": [
        "aiyaa naan Kalmunai irukkaen. engal veetukku thanneer vandhuchu. maarvaa varaikkum thanneer irukku. engal kudumbathula aaru pear irukkoam. vayatha pettorum siru pillaikalum irukaanga. padagu anupungal please mikavum avasaram.",
        "aiyaa Trinco vellam vandhuchu. thanneer kaluthu varaikkum eruduchu. veetula naanga ainthu pear. pillaigal koorai mela erindhoam. uadanae udavi anupungal.",
        "sir Batticaloa irundhu pesuginaen. aatruvellam veetukku vandhuchu. iduppu alavu thanneer. veetula muthiyoor irukaanga naga mudiyala. padagu anupungal please.",
    ],
}

VOICE_CONVERSATIONS_FIRE = {
    "sinhala": [
        "ගිනිය ගිනිය! කොළොන්නාවේ ගෙදරකට ගිනි ගෙනලා. ළමයි දෙන්නෙකුයි ඇතුලේ. ගිනි නිවීමේ රථය ඉක්මනට ගෙනත් ටිකේ ගිනි පැතිරෙනවා.",
        "සාර් දෙමටගොඩ ළඟ ගිනිණක්. ගේ සම්පූර්ණ ගිනිගෙනලා. කීප ගෙයක් ඇවිළෙලා. ළමා ශාලාවත් ළඟ ඉන්නෙ. ඉස්කෝලේ ළමයිත් ළඟ ඉන්නෙ. කරුණාකර ඉක්මනට.",
        "බත්තරමුල්ල ළඟ ගෑස් ලීකිං. ගිනිගෙනලා ගෙදර. ගෙදර ඇතුලේ අය ගොඩ ගන්නෙ කොහොමද. ගිනිකෙළිය ලොකු ගලා. ඉවත් යන්නේ කොහෙද.",
    ],
    "romanized": [
        "giniyi giniyi! Kolonnawa gedarak gini ganala. lamayi dennakyui aethule. gini niwima rathaya ikmanata genenna tikae gini paethirenawa.",
        "sir Dematagoda langa ginnk. ge sampurna ginigena. kipa geyyak aweelela. lamaasala langa innae. ikmanata karunakara.",
        "Battaramulla gas leeking. ginigena gedara. aethulae ay goda gannae kohomada. ginikalaya loku. iwaath yanna koheda.",
    ],
    "tamil": [
        "ஐயா கொலொன்னாவில் வீட்டில் தீ பிடித்துவிட்டது. குழந்தைகள் உள்ளே சிக்கிக்கொண்டுள்ளனர். தீயணைப்பு வாகனம் உடனே அனுப்புங்கள் தீ பரவுகிறது.",
        "சார் மட்டக்குளியில் பெரிய தீ விபத்து. மூன்று வீடுகளில் தீ பரவிவிட்டது. மக்கள் வெளியே வர முடியவில்லை. உடனடியாக வாருங்கள்.",
        "திருகோணமலையில் கேஸ் கசிந்து தீ பிடித்துவிட்டது. வீட்டில் உள்ளவர்களை எப்படி காப்பாற்றுவது. தீ மிகவும் பெரிதாக உள்ளது.",
    ],
    "romanized_tamil": [
        "aiyaa Kolonnawala veetula thee pidichirukku. pillaigal ullae maattikirukkaanga. theeanaikkappu vaandam uadanae anupungal thee parariyudhu.",
        "sir Mattakkuliyala peria thee vipaththu. moonu veedukalula thee paravuduchu. makkal veliyae vara mudiyala. uadanae varungal.",
        "Trinco gaas kassinthu thee pidichirukku. veetula irupavar eppadi kaapaatruvadhu. thee migavum perithaka irukku.",
    ],
}


# ============================================================
# THIRD-PARTY REPORTING PHRASES — someone calling about another person
# ============================================================
THIRD_PARTY_SINHALA = [
    "මගේ යාළුවාගේ ගෙදරට වතුර ආවා",
    "මගේ නැන්දාගේ ගෙදර යටවෙලා",
    "මගේ මල්ලිගේ පවුල හිරවෙලා",
    "අපේ ඥාතියෙකුගේ ගෙදර ගිනි ඇතිවෙලා",
    "මගේ අක්කාගේ ගෙදර ළඟ ගංවතුර",
    "මගේ මිතුරා කිව්වා උදව් ඕන කියලා",
    "අපේ ගමේ මිනිස්සු අසරණයි",
    "අපේ පවුලේ අයට උදව් කරන්න",
]

THIRD_PARTY_TAMIL = [
    "என் நண்பரின் வீட்டிற்கு தண்ணீர் வந்துவிட்டது",
    "என் அத்தையின் வீடு மூழ்கிவிட்டது",
    "என் தம்பியின் குடும்பம் சிக்கிக்கொண்டுள்ளது",
    "எங்கள் உறவினரின் வீட்டில் தீ பிடித்துவிட்டது",
    "என் அக்காவின் வீட்டருகில் வெள்ளம்",
    "என் நண்பர் உதவி வேண்டும் என்று கூறினார்",
]

THIRD_PARTY_ROMANIZED = [
    "mage yaluwage gedara wathura aawa",
    "mage naendage ge yatawela",
    "mage mallige pavula hirawela",
    "ape gnathiyekuge ge gini aethiwela",
    "mage akkage ge langa gangwathura",
]

# ============================================================
# EXACT ADDRESS — street/lane templates for victim_address scenario
# ============================================================
_STREET_SI = ["මාවත", "පාර", "වීදිය", "ලේන්", "රෝඩ්"]
_STREET_EN = ["Road", "Lane", "Street", "Place", "Mawatha"]
_STREET_TA = ["வீதி", "தெரு", "மாவத்தை", "நெடுஞ்சாலை"]


def _make_address(loc_key, loc_data, use_tamil, use_romanized):
    """Build a plausible exact house address string for a location."""
    num = random.randint(1, 250)
    sub = random.choice(["", f"/{random.randint(1, 9)}", "/A", "/B", "/C"])
    if use_tamil:
        town = random.choice(TAMIL_LOCATIONS.get(loc_key, [loc_key]))
        street = random.choice(_STREET_TA)
        return f"{num}{sub}, {random.randint(1, 50)} {street}, {town}"
    elif use_romanized:
        town = loc_key.replace("_City", "").replace("_Town", "")
        street = random.choice(_STREET_EN)
        return f"No {num}{sub}, {town} {street}"
    else:
        town = random.choice(loc_data.get("sinhala", [loc_key]))
        street = random.choice(_STREET_SI)
        return f"{num}{sub}, {town} {street}"


# ============================================================
# UNKNOWN LOCATION — caller cannot state where they are
# ============================================================
UNKNOWN_LOC_SINHALA = [
    "ස්ථානය හරියටම කියන්න බැහැ",
    "ලිපිනය දන්නේ නැහැ",
    "හරිය කොතැනද දන්නේ නැහැ",
    "ගෙදර ලිපිනය කියන්න බැරිවෙලා",
    "නිශ්චිත ස්ථානය නොදනිමි",
    "මම ආගන්තුකයෙක් මෙතන",
]
UNKNOWN_LOC_TAMIL = [
    "இடம் சரியாக தெரியவில்லை",
    "முகவரி தெரியவில்லை",
    "எங்கு என்று சொல்ல முடியவில்லை",
    "சரியான இடம் தெரியவில்லை",
    "நான் இங்கு புதியவன்",
]
UNKNOWN_LOC_ROMANIZED = [
    "sthaaanaya hariyatama kiyanna baha",
    "lipinaaya dannae naha",
    "hariya kothanada dannae naha",
    "ge lipinaya kiyanna bariwela",
]

# ============================================================
# MINIMAL MESSAGE — 1-2 word distress signals
# ============================================================
MINIMAL_FLOOD = [
    "ගංවතුර", "ජලය", "උදව්", "හදිසි", "ගිලෙනවා", "බේරගන්න",
    "gangwathura", "udaw", "wathura", "help", "flood", "SOS",
    "வெள்ளம்", "உதவி", "தண்ணீர்",
]
MINIMAL_FIRE = [
    "ගිනි", "ගින්නෙ", "ගිනිගෙන", "උදව්", "හදිසි", "බේරගන්න",
    "gini", "udaw", "fire", "help", "SOS",
    "தீ", "உதவி", "நெருப்பு",
]

# ============================================================
# HOUSE-SPECIFIC DAMAGE PHRASES — same_street_neighbors scenario
# house_idx=0: structural damage (roof / walls / entry)
# house_idx=1: water depth / spread through the house
# These differ enough semantically that XGBoost can learn they
# are separate incidents despite identical geocoded coordinates.
# ============================================================
HOUSE_DAMAGE_FLOOD = [
    {   # house 0 — structural / roof damage
        "sinhala":   ["වහළ කඩාවැටිලා", "ගෙදර බිත්ති බිඳිලා", "ඉදිරිපස දොර ජලය නිසා ඇරෙන්නේ නැහැ"],
        "tamil":     ["கூரை இடிந்துவிட்டது", "சுவர் விரிசல் விட்டுவிட்டது", "கதவு திறக்க முடியவில்லை"],
        "romanized": ["wahala kadawaetila", "ge bitti bindila", "dora arenney naha wathuren"],
    },
    {   # house 1 — water depth / inundation
        "sinhala":   ["ගෙදර බිම් මහල සම්පූර්ණ ගිලිලා", "ගෙදර ඇතුලේ ජලය පපුව දක්වා", "ගෙදරින් එළියට යන්න බැහැ"],
        "tamil":     ["தரை மாடி முழுவதும் மூழ்கிவிட்டது", "வீட்டினுள் நீர் மார்பு வரை உயர்ந்துள்ளது", "வெளியே வர முடியவில்லை"],
        "romanized": ["bim mahala gilila", "aethule jalaya papuwa dakwa", "eliyata yanna baha"],
    },
]

HOUSE_DAMAGE_FIRE = [
    {   # house 0 — fire origin (kitchen / gas)
        "sinhala":   ["කුස්සිය ගිනිගෙන", "ගෑස් ලීක් වෙලා ගිනිගත්තා", "ගෙදර ඉදිරිපස කොටස ගිනිගෙන"],
        "tamil":     ["சமையலறை தீப்பிடித்தது", "கேஸ் கசிந்து தீப்பிடித்தது", "வீட்டின் முன்பகுதி எரிகிறது"],
        "romanized": ["kussiya ginigena", "gas leak wela ginigaththa", "ideripse kotas ginigena"],
    },
    {   # house 1 — fire spread (roof / rear)
        "sinhala":   ["වහළේ ගිනිගෙන", "ගෙදර පිටිපස ගිනිගත්තා", "අසල ගෙවල්වලටත් ගිනි ගිහිං"],
        "tamil":     ["கூரை தீப்பிடித்துள்ளது", "வீட்டின் பின்பகுதி எரிகிறது", "அடுத்த வீட்டிற்கும் தீ பரவிவிட்டது"],
        "romanized": ["wahale ginigena", "pitipas ginigaththa", "asal gewalwalatamath gini gihin"],
    },
]

ROMANIZED_FIRE = {
    "fire_status": [
        "gini gena tiyenawa", "ginnk aethiwela",
        "gindara lokuyi wadi wenawa", "gedara gini aran tiyenawa",
        "ginnen gewal ginigannawa", "gini dael paethirenawa",
        "ginna paalanaya karanna baha", "duma nagenawa godak",
    ],
    "help_needed": [
        "gini niwana ballan ewanna ikmanata",
        "gini niweem rathayak ewanna karunakara",
        "minissu hirawela innawa aethule",
        "ginnen beraganna ona lamayi innawa",
        "gini niweem adhar awashyai wahama",
    ],
    "situation": [
        "gedara danata gini gena tiyenawa", "palamu mahala ginigaththa",
        "mulu godanaegillamama ginigena", "gewal thunaka ginigaththa",
        "viduli kaanduwak nisa ginnak", "gini duma nisa husma ganna baha",
    ]
}

# ============================================================
# 3. URGENCY PROFILES
# ============================================================
URGENCY_PROFILES = {
    "CRITICAL": {"people_range": (3, 25), "confidence_range": (0.75, 0.98),
                 "extra": ["මිනිස්සු මැරෙනවා", "ආධාර නැතිනම් මැරෙනවා", "ගොඩක් අවදානම්"]},
    "HIGH":     {"people_range": (2, 15), "confidence_range": (0.65, 0.95),
                 "extra": ["ඉක්මනට එන්න", "තත්ත්වය නරක අතට"]},
    "MEDIUM":   {"people_range": (1, 8),  "confidence_range": (0.50, 0.85),
                 "extra": ["මාර්ගය යටවෙලා", "තත්ත්වය බලන්න"]},
    "LOW":      {"people_range": (0, 3),  "confidence_range": (0.35, 0.70),
                 "extra": ["තත්ත්වය හොඳයි", "අඩු වෙනවා"]},
}

# ============================================================
# 4. SCENARIO BUILDER
# ============================================================
def build_scenarios(incident_type):
    scenarios = []
    loc_keys = list(LOCATIONS.keys())

    if incident_type == "flood":
        critical_locs = ["Kelaniya", "Kaduwela", "Ratnapura_City", "Kandy_City", "Kalutara", "Panadura"]
    elif incident_type == "fire":
        critical_locs = ["Kolonnawa", "Wellampitiya", "Dematagoda", "Grandpass", "Mattakkuliya", "Battaramulla"]
    else:
        critical_locs = []

    # Critical mass-report (6 × 10-12 reports)
    for loc in critical_locs:
        scenarios.append({"id": f"{incident_type.upper()}_{len(scenarios)+1:03d}", "location": loc,
            "urgency": "CRITICAL", "report_count": random.randint(10, 12), "type": "mass_report",
            "base_offset": random.randint(0, 120), "people_base": random.randint(5, 20), "escalating": True})

    # Adjacent critical pairs (3 pairs × 5-7)
    if incident_type == "flood":
        pairs = [("Kolonnawa", "Wellampitiya"), ("Grandpass", "Dematagoda"), ("Katugastota", "Kandy_City")]
    elif incident_type == "fire":
        pairs = [("Kelaniya", "Kaduwela"), ("Panadura", "Horana"), ("Gampola", "Kadugannawa")]
    else:
        pairs = []
    for l1, l2 in pairs:
        for loc in [l1, l2]:
            scenarios.append({"id": f"{incident_type.upper()}_{len(scenarios)+1:03d}", "location": loc,
                "urgency": "CRITICAL", "report_count": random.randint(5, 7), "type": "adjacent_critical",
                "base_offset": random.randint(0, 60), "people_base": random.randint(3, 10), "escalating": False})

    # High (8 × 5-8)
    if incident_type == "flood":
        high_locs = ["Biyagama", "Wattala", "Negombo", "Gampaha_Town", "Beruwala", "Horana", "Peradeniya", "Gampola"]
    elif incident_type == "fire":
        high_locs = ["Ja-Ela", "Kadana", "Hanwella", "Avissawella", "Eheliyagoda", "Kuruwita", "Pelmadulla", "Balangoda"]
    else:
        high_locs = []
    for loc in high_locs:
        scenarios.append({"id": f"{incident_type.upper()}_{len(scenarios)+1:03d}", "location": loc,
            "urgency": "HIGH", "report_count": random.randint(5, 8), "type": "standard",
            "base_offset": random.randint(30, 180), "people_base": random.randint(2, 10), "escalating": random.choice([True, False])})

    # Medium (8 × 4-6)
    med_locs = random.sample(loc_keys, min(8, len(loc_keys)))
    for loc in med_locs:
        scenarios.append({"id": f"{incident_type.upper()}_{len(scenarios)+1:03d}", "location": loc,
            "urgency": "MEDIUM", "report_count": random.randint(4, 6), "type": "standard",
            "base_offset": random.randint(60, 300), "people_base": random.randint(1, 5), "escalating": False})

    # Low (5 × 3-4)
    for loc in random.sample(loc_keys, 5):
        scenarios.append({"id": f"{incident_type.upper()}_{len(scenarios)+1:03d}", "location": loc,
            "urgency": "LOW", "report_count": random.randint(3, 4), "type": "standard",
            "base_offset": random.randint(120, 480), "people_base": random.randint(0, 3), "escalating": False})

    # Noise (single reports)
    for loc in random.sample(loc_keys, 8):
        scenarios.append({"id": f"{incident_type.upper()}_{len(scenarios)+1:03d}", "location": loc,
            "urgency": random.choice(["LOW", "MEDIUM"]), "report_count": 1, "type": "noise",
            "base_offset": random.randint(0, 600), "people_base": random.randint(0, 2), "escalating": False})

    # Conflicting counts (3 × 5-7)
    for loc in random.sample(loc_keys, 3):
        scenarios.append({"id": f"{incident_type.upper()}_{len(scenarios)+1:03d}", "location": loc,
            "urgency": "HIGH", "report_count": random.randint(5, 7), "type": "conflicting_counts",
            "base_offset": random.randint(30, 120), "people_base": random.randint(3, 8), "escalating": False})

    # Long time gap (2)
    for loc in random.sample(loc_keys, 2):
        scenarios.append({"id": f"{incident_type.upper()}_{len(scenarios)+1:03d}", "location": loc,
            "urgency": "HIGH", "report_count": random.randint(4, 6), "type": "long_gap",
            "base_offset": random.randint(0, 60), "people_base": random.randint(3, 8), "escalating": True})

    # Geographically near (4 pairs) — same town, adjacent GN divisions
    # These are DIFFERENT incidents 0.3-2 km apart; system MUST NOT merge them
    geo_near_locs = random.sample(loc_keys, min(4, len(loc_keys)))
    for loc in geo_near_locs:
        for sub_idx in range(2):  # 2 incidents per location
            scenarios.append({
                "id": f"{incident_type.upper()}_{len(scenarios)+1:03d}",
                "location": loc,
                "urgency": random.choice(["HIGH", "CRITICAL"]),
                "report_count": random.randint(4, 6),
                "type": "geographically_near",
                "base_offset": random.randint(0, 90),
                "people_base": random.randint(2, 12),
                "escalating": False,
                "gn_sub_index": sub_idx,  # forces different GN divisions
            })

    # Third-party reporting (5) — friend/family member calling about someone
    for loc in random.sample(loc_keys, 5):
        scenarios.append({
            "id": f"{incident_type.upper()}_{len(scenarios)+1:03d}",
            "location": loc,
            "urgency": random.choice(["HIGH", "CRITICAL", "MEDIUM"]),
            "report_count": random.randint(4, 7),
            "type": "third_party",
            "base_offset": random.randint(15, 180),
            "people_base": random.randint(2, 8),
            "escalating": False,
        })

    # Cross-lingual duplicate (6) — same incident, same time, guaranteed Sinhala+Tamil
    # report_idx 0 → Sinhala Unicode, report_idx 1 → Tamil, rest → random mix
    # Provides guaranteed positive training pairs across language boundaries.
    for loc in random.sample(loc_keys, 6):
        scenarios.append({
            "id": f"{incident_type.upper()}_{len(scenarios)+1:03d}",
            "location": loc,
            "urgency": random.choice(["CRITICAL", "HIGH", "MEDIUM"]),
            "report_count": random.randint(4, 6),
            "type": "cross_lingual_duplicate",
            "base_offset": random.randint(0, 120),
            "people_base": random.randint(2, 10),
            "escalating": False,
        })

    # Victim gives exact address (5) — "No 45/B, Kandy Road" instead of town name
    # Tests gazetteer's ability to parse street addresses + ML dedup on address strings
    for loc in random.sample(loc_keys, 5):
        scenarios.append({
            "id": f"{incident_type.upper()}_{len(scenarios)+1:03d}",
            "location": loc,
            "urgency": random.choice(["CRITICAL", "HIGH", "MEDIUM"]),
            "report_count": random.randint(3, 5),
            "type": "victim_address",
            "base_offset": random.randint(0, 180),
            "people_base": random.randint(1, 8),
            "escalating": False,
        })

    # Location unknown (5) — caller cannot state where they are
    # Teaches XGBoost: high semantic similarity can still mean same incident
    # even when geo_sim=0 (unresolved location)
    for loc in random.sample(loc_keys, 5):
        scenarios.append({
            "id": f"{incident_type.upper()}_{len(scenarios)+1:03d}",
            "location": loc,
            "urgency": random.choice(["HIGH", "CRITICAL"]),
            "report_count": random.randint(3, 5),
            "type": "location_unknown",
            "base_offset": random.randint(0, 120),
            "people_base": random.randint(2, 10),
            "escalating": False,
        })

    # Minimal message (5) — 1-2 words only, no location, no people count
    # Teaches XGBoost: very sparse reports are hard negatives / edge cases
    # System should create new UIR rather than force-merge on weak signal
    for loc in random.sample(loc_keys, 5):
        scenarios.append({
            "id": f"{incident_type.upper()}_{len(scenarios)+1:03d}",
            "location": loc,
            "urgency": random.choice(["HIGH", "MEDIUM", "LOW"]),
            "report_count": random.randint(2, 4),
            "type": "minimal_message",
            "base_offset": random.randint(0, 300),
            "people_base": random.randint(0, 3),
            "escalating": False,
        })

    # Victim panic (8) — 1-5 word messages: "help gampola", "flood!", "SOS kelaniya"
    # The victim is panicked / injured / has no time — bare minimum information.
    # Different from minimal_message: panic messages ALWAYS contain either a
    # location token or a disaster word (or both), and confidence is near-zero.
    for loc in random.sample(loc_keys, 8):
        scenarios.append({
            "id": f"{incident_type.upper()}_{len(scenarios)+1:03d}",
            "location": loc,
            "urgency": random.choice(["CRITICAL", "HIGH"]),
            "report_count": random.randint(1, 3),
            "type": "victim_panic",
            "base_offset": random.randint(0, 60),
            "people_base": random.randint(1, 5),
            "escalating": False,
        })

    # Same-street neighbors (5 pairs) — two houses 10-50 m apart on the same street
    # Both resolve to the SAME GN centroid (identical lat/lng after geocoding)
    # MUST NOT merge — only distinguishing signals are address string + damage phrases
    # Teaches XGBoost: geo_sim=1.0 alone does NOT guarantee same incident
    for loc in random.sample(loc_keys, 5):
        for house_idx in range(2):
            scenarios.append({
                "id": f"{incident_type.upper()}_{len(scenarios)+1:03d}",
                "location": loc,
                "urgency": random.choice(["CRITICAL", "HIGH"]),
                "report_count": random.randint(3, 5),
                "type": "same_street_neighbors",
                "base_offset": random.randint(0, 60),
                "people_base": random.randint(2, 8),
                "escalating": False,
                "house_idx": house_idx,   # 0 = lower house numbers, 1 = higher house numbers
            })

    return scenarios


# ============================================================
# 5. REPORT GENERATOR
# ============================================================
def get_phrases(incident_type, category, use_romanized, use_tamil=False, use_romanized_tamil=False):
    if use_romanized_tamil:
        src = ROMANIZED_TAMIL_FLOOD if incident_type == "flood" else ROMANIZED_TAMIL_FIRE
    elif use_tamil:
        src = TAMIL_FLOOD if incident_type == "flood" else TAMIL_FIRE
    elif use_romanized:
        src = ROMANIZED_FLOOD if incident_type == "flood" else ROMANIZED_FIRE
    else:
        src = FLOOD_PHRASES if incident_type == "flood" else FIRE_PHRASES
    return random.choice(src.get(category, src[list(src.keys())[0]]))


def generate_report(scenario, report_idx, base_time, channel, incident_type, voice_c, sms_c):
    loc_key = scenario["location"]
    loc = LOCATIONS[loc_key]
    urgency = scenario["urgency"]
    profile = URGENCY_PROFILES[urgency]

    # Language assignment
    # cross_lingual_duplicate: report 0 → Sinhala, report 1 → Tamil,
    #   report 2 → Romanized Tamil, rest → random
    # All other scenarios:
    #   35% Sinhala Unicode, 25% Romanized Sinhala, 25% Tamil, 15% Romanized Tamil
    use_romanized_tamil = False
    if scenario["type"] == "cross_lingual_duplicate":
        if report_idx == 0:
            use_tamil, use_romanized = False, False       # Sinhala Unicode
        elif report_idx == 1:
            use_tamil, use_romanized = True, False        # Tamil
        elif report_idx == 2:
            use_tamil, use_romanized, use_romanized_tamil = False, False, True  # Romanized Tamil
        else:
            lang_roll = random.random()
            use_romanized_tamil = lang_roll < 0.20
            use_tamil = (not use_romanized_tamil) and lang_roll < 0.45
            use_romanized = (not use_romanized_tamil) and (not use_tamil) and lang_roll < 0.70
    else:
        lang_roll = random.random()
        use_romanized_tamil = lang_roll < 0.15
        use_tamil = (not use_romanized_tamil) and lang_roll < 0.40
        use_romanized = (not use_romanized_tamil) and (not use_tamil) and lang_roll < 0.65

    # Time
    if scenario["type"] == "long_gap" and report_idx >= scenario["report_count"] // 2:
        toff = timedelta(minutes=random.randint(120, 240) + report_idx * random.randint(2, 8))
    elif scenario["escalating"]:
        toff = timedelta(minutes=report_idx * random.randint(2, 10))
    else:
        toff = timedelta(minutes=random.randint(0, 30) + report_idx * random.randint(1, 5))
    report_time = base_time + timedelta(minutes=scenario["base_offset"]) + toff

    # People
    pb = scenario["people_base"]
    if scenario["type"] == "conflicting_counts":
        people = pb if report_idx < 2 else (pb * 3 + random.randint(0, 5) if report_idx < 4 else pb + random.randint(-1, 2))
    elif scenario["escalating"] and report_idx > scenario["report_count"] // 2:
        people = pb + random.randint(2, 10)
    else:
        dev = max(1, int(pb * 0.3))
        people = max(0, pb + random.randint(-dev, dev))
    if channel == "sms" and random.random() < 0.15:
        people = None

    # Confidence
    clo, chi = profile["confidence_range"]
    confidence = round(random.uniform(clo, chi), 2)
    if channel == "sms":
        confidence = round(max(0.3, confidence - random.uniform(0.05, 0.15)), 2)

    # Location
    if use_romanized_tamil and loc_key in ROMANIZED_TAMIL_LOCATIONS:
        location_str = random.choice(ROMANIZED_TAMIL_LOCATIONS[loc_key])
    elif use_tamil and loc_key in TAMIL_LOCATIONS:
        location_str = random.choice(TAMIL_LOCATIONS[loc_key])
    elif use_romanized:
        location_str = random.choice(loc["romanized"])
    else:
        location_str = random.choice(loc["sinhala"])
    if not use_tamil and not use_romanized_tamil and random.random() < 0.12:
        location_str = random.choice(loc["landmark"])
    if channel == "sms" and random.random() < 0.05:
        location_str = ""

    # Key phrases — categories differ by disaster type
    if incident_type == "flood":
        cats = ["water_level", "help_needed", "situation"]
    else:  # fire
        cats = ["fire_status", "help_needed", "situation"]
    kp = [get_phrases(incident_type, c, use_romanized, use_tamil, use_romanized_tamil) for c in cats]
    kp.append(random.choice(profile["extra"]))

    # Full conversation transcript for voice calls
    transcript = None
    if channel == "voice" and random.random() < 0.75:
        if incident_type == "flood":
            conv_pool = VOICE_CONVERSATIONS_FLOOD
        elif incident_type == "fire":
            conv_pool = VOICE_CONVERSATIONS_FIRE
        else:
            conv_pool = VOICE_CONVERSATIONS_FIRE
        if use_romanized_tamil:
            transcript = random.choice(conv_pool.get("romanized_tamil", conv_pool["romanized"]))
        elif use_tamil:
            transcript = random.choice(conv_pool["tamil"])
        elif use_romanized:
            transcript = random.choice(conv_pool["romanized"])
        else:
            transcript = random.choice(conv_pool["sinhala"])

    # ── Scenario-specific overrides ──────────────────────────────────────────

    # victim_address: replace location with a precise house address
    if scenario["type"] == "victim_address":
        location_str = _make_address(loc_key, loc, use_tamil or use_romanized_tamil, use_romanized)

    # location_unknown: caller cannot state where they are
    elif scenario["type"] == "location_unknown":
        if use_tamil or use_romanized_tamil:
            location_str = random.choice(UNKNOWN_LOC_TAMIL)
        elif use_romanized:
            location_str = random.choice(UNKNOWN_LOC_ROMANIZED)
        else:
            location_str = random.choice(UNKNOWN_LOC_SINHALA)

    # minimal_message: 1-2 words, no location, no people count, low confidence
    elif scenario["type"] == "minimal_message":
        location_str = ""
        if incident_type == "flood":
            minimal_pool = MINIMAL_FLOOD
        else:
            minimal_pool = MINIMAL_FIRE
        kp = random.sample(minimal_pool, random.randint(1, 2))
        people = None
        confidence = round(random.uniform(0.20, 0.45), 2)

    # victim_panic: 1-5 word panic message — "help gampola", "flood!", "SOS kelaniya"
    # Location may or may not be present. No structured fields at all.
    elif scenario["type"] == "victim_panic":
        panic_pool = _make_panic_messages(loc_key, loc, incident_type)
        panic_msg = random.choice(panic_pool)
        # The panic message IS the entire content — key_phrases = [panic_msg]
        kp = [panic_msg]
        # Location: sometimes inferable from the message, sometimes totally absent
        if any(name.lower() in panic_msg.lower()
               for name in loc.get("romanized", []) + loc.get("sinhala", [])):
            # location token is in the message — keep it as location_raw too
            location_str = panic_msg
        else:
            location_str = ""
        people = None
        confidence = round(random.uniform(0.10, 0.35), 2)
        transcript = None   # no transcript — caller hung up immediately

    # same_street_neighbors: different house number per house_idx + house-specific damage
    elif scenario["type"] == "same_street_neighbors":
        house_idx = scenario.get("house_idx", 0)
        house_num = random.randint(10, 40) if house_idx == 0 else random.randint(50, 90)
        sub = random.choice(["", "/A", "/B", f"/{random.randint(1,5)}"])
        if use_tamil or use_romanized_tamil:
            town = random.choice(TAMIL_LOCATIONS.get(loc_key, [loc_key]))
            location_str = f"{house_num}{sub}, {random.choice(_STREET_TA)}, {town}"
        elif use_romanized:
            town = loc_key.replace("_City", "").replace("_Town", "")
            location_str = f"No {house_num}{sub}, {town} {random.choice(_STREET_EN)}"
        else:
            town = random.choice(loc["sinhala"])
            location_str = f"{house_num}{sub}, {town} {random.choice(_STREET_SI)}"
        # Replace first key phrase with house-specific structural/depth damage phrase
        if incident_type in ("flood", "fire"):
            damage = HOUSE_DAMAGE_FLOOD[house_idx] if incident_type == "flood" else HOUSE_DAMAGE_FIRE[house_idx]
            if use_tamil or use_romanized_tamil:
                kp[0] = random.choice(damage["tamil"])
            elif use_romanized:
                kp[0] = random.choice(damage["romanized"])
            else:
                kp[0] = random.choice(damage["sinhala"])

    # Urgency variation
    reported_urgency = urgency
    if random.random() < 0.12:
        reported_urgency = {"CRITICAL": "HIGH", "HIGH": "MEDIUM", "MEDIUM": "LOW", "LOW": "LOW"}[urgency]
    elif random.random() < 0.08:
        reported_urgency = {"CRITICAL": "CRITICAL", "HIGH": "CRITICAL", "MEDIUM": "HIGH", "LOW": "MEDIUM"}[urgency]

    has_timestamp = not (channel == "sms" and random.random() < 0.10)

    # Source ID
    if channel == "voice":
        voice_c[0] += 1
        src_id = f"call_{voice_c[0]:04d}"
    else:
        sms_c[0] += 1
        src_id = f"sms_{sms_c[0]:04d}"

    # Build report — simulate C1/C3 field name variations
    if channel == "sms" and random.random() < 0.3:
        report = {
            "type": incident_type,
            "loc_text": location_str,
            "urgency": str({"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}[reported_urgency]),
            "people": people,
            "confidence": confidence,
            "key_phrases": kp[:3],
        }
        if has_timestamp:
            report["ts"] = report_time.isoformat()
    else:
        report = {
            "incident_type": incident_type,
            "location_raw": location_str,
            "urgency": reported_urgency if random.random() > 0.1 else reported_urgency.lower(),
            "people_involved": people,
            "confidence": confidence,
            "key_phrases": kp[:3],
        }
        if has_timestamp:
            report["timestamp"] = report_time.isoformat()

    report["source_id"] = src_id
    report["channel"] = channel
    report["receive_time"] = (report_time + timedelta(seconds=random.uniform(0.1, 2.0))).isoformat()

    # Third-party reporting: add proxy phrases
    if scenario["type"] == "third_party" and random.random() < 0.6:
        if use_romanized_tamil:
            kp.insert(0, random.choice(ROMANIZED_TAMIL_THIRD_PARTY))
        elif use_tamil:
            kp.insert(0, random.choice(THIRD_PARTY_TAMIL))
        elif use_romanized:
            kp.insert(0, random.choice(THIRD_PARTY_ROMANIZED))
        else:
            kp.insert(0, random.choice(THIRD_PARTY_SINHALA))

    # Attach transcript to voice call reports
    if transcript:
        report["transcript"] = transcript

    # GN-level coordinate precision
    # minimal_message: no location info at all — ground truth coords also unknown
    if scenario["type"] == "minimal_message":
        gt_lat, gt_lng = None, None
    else:
        gt_lat, gt_lng = loc["lat"], loc["lng"]
    if _GN_BY_DS:
        gn = _get_nearby_gn(loc_key, loc)
        if gn:
            # For geographically_near scenarios, use deterministic GN selection
            # to ensure the two incidents get DIFFERENT GN divisions
            if scenario["type"] == "geographically_near":
                ds_name = loc_key.replace("_City", "").replace("_Town", "")
                gns = _GN_BY_DS.get(ds_name, _GN_BY_DS.get(loc.get("district", ""), []))
                sub_idx = scenario.get("gn_sub_index", 0)
                if len(gns) >= 2:
                    # Split GNs into two halves; each sub-incident uses a different half
                    half = len(gns) // 2
                    pool = gns[:half] if sub_idx == 0 else gns[half:]
                    gn = random.choice(pool) if pool else gn
            gt_lat = gn["lat"]
            gt_lng = gn["lng"]

    if use_romanized_tamil:
        lang_label = "romanized_tamil"
    elif use_tamil:
        lang_label = "tamil"
    elif use_romanized:
        lang_label = "romanized_sinhala"
    else:
        lang_label = "sinhala"

    report["_ground_truth"] = {
        "incident_id": scenario["id"],
        "actual_location": loc_key,
        "actual_urgency": urgency,
        "actual_people": scenario["people_base"],
        "incident_type": incident_type,
        "language": lang_label,
        "scenario_type": scenario["type"],
        "district": loc["district"],
        "province": loc["province"],
        "lat": gt_lat, "lng": gt_lng,
    }
    return report, report_time


def generate_for_type(incident_type, base_time, target_count=500):
    scenarios = build_scenarios(incident_type)
    raw = []
    vc, sc = [0], [0]
    target_voice = target_count // 2
    target_sms = target_count - target_voice
    voice_count, sms_count = 0, 0

    for scenario in scenarios:
        for i in range(scenario["report_count"]):
            if voice_count >= target_voice:
                ch = "sms"
            elif sms_count >= target_sms:
                ch = "voice"
            else:
                ch = random.choice(["voice", "sms"])

            report, rt = generate_report(scenario, i, base_time, ch, incident_type, vc, sc)
            raw.append((report, rt))
            if ch == "voice":
                voice_count += 1
            else:
                sms_count += 1

    # Trim/pad to target_count
    if len(raw) > target_count:
        raw = raw[:target_count]
    while len(raw) < target_count:
        loc_key = random.choice(list(LOCATIONS.keys()))
        ns = {"id": f"NOISE_{incident_type}_{len(raw)}", "location": loc_key,
              "urgency": random.choice(["LOW", "MEDIUM"]), "report_count": 1, "type": "noise",
              "base_offset": random.randint(0, 600), "people_base": random.randint(0, 2), "escalating": False}
        ch = "voice" if voice_count < target_voice else "sms"
        r, rt = generate_report(ns, 0, base_time, ch, incident_type, vc, sc)
        raw.append((r, rt))
        if ch == "voice":
            voice_count += 1
        else:
            sms_count += 1

    raw.sort(key=lambda x: x[1])
    return [r for r, _ in raw], voice_count, sms_count


def generate_dataset():
    base_time = datetime(2025, 5, 17, 6, 0, 0, tzinfo=timezone.utc)

    flood_reports,     fv, fs = generate_for_type("flood",     base_time,                       target_count=500)
    fire_reports,      rv, rs = generate_for_type("fire",      base_time + timedelta(hours=1),  target_count=500)
    all_reports = flood_reports + fire_reports

    lang_counts     = Counter(r["_ground_truth"]["language"]      for r in all_reports)
    scenario_counts = Counter(r["_ground_truth"]["scenario_type"] for r in all_reports)
    district_counts = Counter(r["_ground_truth"]["district"]      for r in all_reports)

    dataset = {
        "metadata": {
            "dataset_name": "C4 Flood+Fire Dataset — Sinhala + Romanized Sinhala + Tamil + Romanized Tamil",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_reports": len(all_reports),
            "flood_reports": len(flood_reports),
            "fire_reports": len(fire_reports),
            "voice_reports": fv + rv,
            "sms_reports": fs + rs,
            "language": "Sinhala Unicode + Romanized Sinhala + Tamil + Romanized Tamil",
            "provinces": ["Western", "Central"],
            "incident_types": ["flood", "fire"],
            "language_distribution": dict(lang_counts),
            "scenario_distribution": dict(scenario_counts),
            "district_distribution": dict(district_counts),
            "gn_precision": len(_GN_BY_DS) > 0,
            "per_type": {
                "flood": {"total": len(flood_reports), "voice": fv, "sms": fs},
                "fire":  {"total": len(fire_reports),  "voice": rv, "sms": rs},
            },
        },
        "reports": all_reports,
    }
    return dataset


if __name__ == "__main__":
    print("Generating C4 evaluation dataset (1000 reports)...")
    dataset = generate_dataset()

    os.makedirs("data", exist_ok=True)
    out = "data/disaster_dataset_1000.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2, default=str)

    m = dataset["metadata"]
    print(f"\n{'='*60}")
    print(f"  DATASET GENERATED")
    print(f"{'='*60}")
    print(f"  Total:     {m['total_reports']} reports")
    print(f"  Flood:     {m['per_type']['flood']['total']} (voice={m['per_type']['flood']['voice']}, sms={m['per_type']['flood']['sms']})")
    print(f"  Fire:      {m['per_type']['fire']['total']} (voice={m['per_type']['fire']['voice']}, sms={m['per_type']['fire']['sms']})")
    print(f"  Language:  {m['language_distribution']}")
    print(f"  Provinces: {m['provinces']}")
    print(f"  Scenarios: {m['scenario_distribution']}")
    print(f"  Districts: {m['district_distribution']}")
    print(f"  GN precision: {m['gn_precision']}")
    print(f"  Saved: {out}")
    print(f"{'='*60}")
