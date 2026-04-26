"""
GN Division Extractor — parses lka_admin4.geojson → compact data/gn_divisions.json
==================================================================================
Sri Lanka admin hierarchy:  Province → District → DS Division → GN Division
The GeoJSON (312 MB, 14,043 features) has polygon boundaries we don't need.
This script extracts just the centroid + names for our 5 target districts,
producing a ~1 MB JSON that the gazetteer loads at startup.

Run once:  python -m pipeline.gn_extractor
"""

import json
import sys
from pathlib import Path
from collections import Counter

TARGET_DISTRICTS = {"Colombo", "Gampaha", "Kalutara", "Kandy", "Ratnapura"}

GEOJSON_PATH = Path(__file__).parent.parent / "lka_admin4.geojson"
OUTPUT_PATH  = Path(__file__).parent.parent / "data" / "gn_divisions.json"


def extract(geojson_path: Path = GEOJSON_PATH,
            output_path: Path = OUTPUT_PATH,
            target_districts: set = TARGET_DISTRICTS) -> int:
    """
    Read the GeoJSON, keep only centroids + names for target districts.
    Returns the number of GN divisions extracted.
    """
    print(f"  Reading {geojson_path.name} ({geojson_path.stat().st_size/1e6:.0f} MB)...")
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)

    gn_list = []
    for feature in data["features"]:
        p = feature["properties"]
        district = p.get("adm2_name", "")
        if target_districts and district not in target_districts:
            continue

        gn_list.append({
            "name_en":     p.get("adm4_name", ""),
            "name_si":     p.get("adm4_name1") or "",
            "name_ta":     p.get("adm4_name2") or "",
            "pcode":       p.get("adm4_pcode", ""),
            "ds_division": p.get("adm3_name", ""),
            "ds_pcode":    p.get("adm3_pcode", ""),
            "district":    district,
            "province":    p.get("adm1_name", ""),
            "lat":         p.get("center_lat"),
            "lng":         p.get("center_lon"),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"gn_divisions": gn_list, "count": len(gn_list)}, f,
                  ensure_ascii=False, indent=1)

    print(f"  Extracted {len(gn_list)} GN divisions -> {output_path}")
    by_district = Counter(g["district"] for g in gn_list)
    for d, c in sorted(by_district.items()):
        print(f"    {d:15s}: {c:5d} GNs")

    return len(gn_list)


def extract_all(geojson_path: Path = GEOJSON_PATH,
                output_path: Path = None) -> int:
    """Extract ALL 14,043 GN divisions (for full-island coverage)."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "gn_divisions_all.json"
    return extract(geojson_path, output_path, target_districts=None)


if __name__ == "__main__":
    if not GEOJSON_PATH.exists():
        print(f"ERROR: {GEOJSON_PATH} not found")
        print("Download from HDX: https://data.humdata.org/dataset/cod-ab-lka")
        sys.exit(1)

    n = extract()
    print(f"\nDone. {n} GN divisions ready for gazetteer.")
