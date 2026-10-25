import pandas as pd
import json
import re
from pathlib import Path

df = pd.read_excel("Densitydatabasev2__1_.xls", sheet_name="Sheet1")
entries = []
for i in range(len(df)):
    row = df.iloc[i]
    name = row.iloc[0]
    dens = row.iloc[1]
    src = row.iloc[2] if len(row) > 2 else None
    if pd.isna(name):
        continue
    name_str = str(name).strip()
    if name_str == "Food name and description":
        continue
    if pd.isna(dens):
        continue
    entries.append((name_str, dens, src))

def parse_density(d):
    if pd.isna(d):
        return None, None, None
    if isinstance(d, (int, float)):
        return float(d), str(d), "numeric"
    s = str(d).strip()
    m = re.findall(r"([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", s)
    if len(m) >= 2:
        try:
            v1 = float(m[0]); v2 = float(m[1])
            return (v1+v2)/2.0, s, "range_avg"
        except Exception:
            pass
    if len(m) == 1:
        try:
            return float(m[0]), s, "numeric"
        except Exception:
            pass
    return None, s, "unparsed"

fao_parsed = []
for name, dens, src in entries:
    p, raw, kind = parse_density(dens)
    if p is not None and p > 0:
        fao_parsed.append({
            "name": name,
            "density_g_cm3": p,
            "raw": raw,
            "parse_kind": kind,
            "source": src
        })

with open("models/classifier/class_names.json") as f:
    class_names = json.load(f)

def normalize(s):
    s2 = re.sub(r"[^a-z0-9]+", " ", str(s).lower())
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

fao_norm = []
for e in fao_parsed:
    fao_norm.append({**e, "norm": normalize(e["name"])})

# Curated high-confidence mappings (only where very clear)
CURATED = {
    "ice_cream": ["ice cream"],
    "lasagna": ["lasagna"],
    "soup": None,  # too generic
}

results = []
for label in class_names:
    label_norm = normalize(label.replace("_", " "))
    matches = []
    # Try exact normalized match
    for fe in fao_norm:
        if fe["norm"] == label_norm:
            matches.append(("exact_norm", fe))
    # Try curated specific mappings
    if not matches and label in CURATED and CURATED[label]:
        for target in CURATED[label]:
            tnorm = normalize(target)
            for fe in fao_norm:
                if fe["norm"] == tnorm:
                    matches.append(("curated_exact", fe))
    # Be very conservative - only allow if it's a single, very specific match
    # and the FAO name is not just a generic category
    # Filter out obvious bad matches
    filtered = []
    for mtype, fe in matches:
        # Skip if FAO is just a category-like generic word
        bad_generics = {"cake", "bread", "nuts", "soup", "salad", "rice", "pasta"}
        if fe["norm"] in bad_generics and label_norm not in fe["norm"]:
            continue  # too generic, skip unless exact match of the generic
        filtered.append((mtype, fe))
    matches = filtered
    # Deduplicate
    seen = set()
    uniq = []
    for mtype, fe in matches:
        key = fe["name"]
        if key not in seen:
            seen.add(key)
            uniq.append((mtype, fe))
    results.append({"label": label, "label_norm": label_norm, "matches": uniq})

exact = []
ambig = []
unmatched = []
for r in results:
    if not r["matches"]:
        unmatched.append(r)
    elif len(r["matches"]) == 1:
        exact.append(r)
    else:
        ambig.append(r)

out = []
out.append("# FAO/INFOODS density mapping analysis (Food-101 -> FAO) - CONSERVATIVE")
out.append("")
out.append(f"Total Food-101 labels: {len(class_names)}")
out.append(f"Total valid FAO density entries parsed: {len(fao_parsed)}")
out.append("")
out.append("## Range handling")
out.append("- Excel ranges like `1.01-1.03` are parsed by taking the arithmetic mean (e.g. 1.02).")
out.append("- The `raw` string and `parse_kind` ('range_avg') are preserved for traceability.")
out.append("- Units are g/ml = g/cm^3 (no conversion needed).")
out.append("- Only positive finite densities are accepted; unparseable/non-positive entries are ignored (no guessing).")
out.append("- If a range cannot be parsed, the entry is NOT used (no fallback guess).")
out.append("")
out.append("## Matching strategy (proposed - CONSERVATIVE)")
out.append("1. Normalize both sides: lowercase, replace non-alphanumeric with spaces, collapse whitespace.")
out.append("2. Only accept exact normalized string match OR explicitly curated high-confidence mappings.")
out.append("3. Do NOT do broad substring matching (too error-prone for composite dishes).")
out.append("4. Ambiguous if multiple different FAO entries would match with similar specificity - no automatic selection; requires explicit mapping or leave unmatched.")
out.append("5. NO guessed densities for unmatched foods - they will be unavailable in the density lookup.")
out.append("")
out.append(f"- Exact/single high-confidence match: {len(exact)}")
out.append(f"- Ambiguous (multiple candidates): {len(ambig)}")
out.append(f"- Unmatched (no candidates): {len(unmatched)}")
out.append("")
out.append("### EXACT/SINGLE HIGH-CONFIDENCE MATCHES")
for r in exact:
    mtype, fe = r["matches"][0]
    out.append(f"- {r['label']} (norm='{r['label_norm']}') <- FAO: '{fe['name']}' | {fe['density_g_cm3']} g/cm^3 | raw='{fe['raw']}' | kind={fe['parse_kind']} | src={fe['source']} [match:{mtype}]")
out.append("")
out.append("### AMBIGUOUS (multiple candidates - need explicit resolution)")
for r in ambig:
    out.append(f"- {r['label']} (norm='{r['label_norm']}') has {len(r['matches'])} candidates:")
    for mtype, fe in r["matches"]:
        out.append(f"  - [{mtype}] '{fe['name']}' | {fe['density_g_cm3']} g/cm^3 | raw='{fe['raw']}' | src={fe['source']}")
out.append("")
out.append("### UNMATCHED (no FAO entry found - NO guessed density)")
for r in unmatched:
    out.append(f"- {r['label']} (norm='{r['label_norm']}')")
out.append("")

Path("FAO_MAPPING_ANALYSIS.md").write_text("\n".join(out), encoding="utf-8")
print("done")
