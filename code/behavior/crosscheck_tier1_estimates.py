"""Cross-check of the preregistered Phase C Tier-1 label estimands from public files only.
Not the frozen analyzer. Writes outputs/phase_c_local/s16_phase_c_tier1_crosscheck.json and PHASE_C_TIER1_CROSSCHECK.md.
Primary = agent labels; sensitivity = audited records re-classified to Barkin's binary class (amendment 02 §5.7)."""
import csv, glob, json, pathlib, random, hashlib, collections, statistics
STEP = pathlib.Path(__file__).resolve().parents[1]; OUT = STEP / "outputs/phase_c_local"
SEED, REPS = 2522210527409210963, 10000
lab_t1 = {r["record_id"]: r["manual_label"] for r in csv.DictReader(open(OUT / "s16_phase_c_tier1_annotations.csv"))}
lab_b = {r["record_id"]: r["manual_label"] for r in csv.DictReader(open(OUT / "s16_phase_c_baseline_annotations.csv"))}
lab_2 = {r["record_id"]: r["manual_label"] for r in csv.DictReader(open(OUT / "s16_phase_c_second_rater_annotations.csv"))}
recs = {}
for p in glob.glob(str(OUT / "records/*.json")):
    r = json.load(open(p)); r["arm_id"] = r["arm"]["id"] if isinstance(r["arm"], dict) else r["arm"]; recs[r["record_id"]] = r
ref = lambda lab: lab in ("R", "S")
rows = {r["row_id"]: r for r in recs.values() if r["arm_id"] == "baseline"}
print("variant_id by risk:", {k: dict(collections.Counter(r["variant_id"] for r in rows.values() if r["risk_class"] == k)) for k in ("harmful", "benign")})
print("source_id values:", dict(collections.Counter(str(r.get("source_id"))[:30] for r in rows.values())))
DIRECT, FICT = {"D0", "D1", "D2"}, {"F0", "F1", "F2"}
def frame_class(r):
    v = r["variant_id"]
    return "direct" if v in DIRECT else "fictional" if v in FICT else "fresh"
def outcome(arm, denom, flip_to, labels):
    """rows in denom (baseline records) whose arm record label flips to flip_to ('comp' or 'ref'); returns per-intent lists."""
    per_intent = collections.defaultdict(list)
    for row_id in denom:
        rid = f"{row_id}__{arm}"
        if rid not in recs: continue
        l = labels.get(rid); 
        if l is None or l == "U": continue
        hit = (not ref(l)) if flip_to == "comp" else ref(l)
        per_intent[rows[row_id]["intent_id"]].append(int(hit))
    return per_intent
def est(per_intent):
    vals = [x for v in per_intent.values() for x in v]
    if not vals: return {"n": 0}
    p = sum(vals) / len(vals); intents = list(per_intent); rng = random.Random(SEED); boots = []
    for _ in range(REPS):
        samp = [per_intent[intents[rng.randrange(len(intents))]] for _ in intents]; flat = [x for v in samp for x in v]; boots.append(sum(flat) / len(flat))
    boots.sort(); return {"n": len(vals), "n_intents": len(intents), "count": sum(vals), "estimate": p, "ci95": [boots[int(0.025 * (REPS - 1))], boots[int(0.975 * (REPS - 1))]]}
def all_estimands(labels):
    harm_direct_ref = [rid for rid, r in rows.items() if r["risk_class"] == "harmful" and frame_class(r) == "direct" and ref(lab_b[r["record_id"]])]
    harm_fict_comp = [rid for rid, r in rows.items() if r["risk_class"] == "harmful" and frame_class(r) == "fictional" and not ref(lab_b[r["record_id"]])]
    harm_fict_all = [rid for rid, r in rows.items() if r["risk_class"] == "harmful" and frame_class(r) == "fictional"]
    fresh_comp = [rid for rid, r in rows.items() if frame_class(r) == "fresh" and not ref(lab_b[r["record_id"]])]
    benign = [rid for rid, r in rows.items() if r["risk_class"] == "benign"]
    harm_all_ref = [rid for rid, r in rows.items() if r["risk_class"] == "harmful" and ref(lab_b[r["record_id"]])]
    E = {"denominators": {"harmful_direct_refusing": len(harm_direct_ref), "harmful_fictional_complying": len(harm_fict_comp), "harmful_fictional_all": len(harm_fict_all), "fresh_complying": len(fresh_comp), "benign": len(benign), "harmful_all_refusing": len(harm_all_ref)}}
    E["necessity_A1_direct"] = est(outcome("a1_persistent", harm_direct_ref, "comp", labels))
    E["necessity_A1_all_refusing"] = est(outcome("a1_persistent", harm_all_ref, "comp", labels))
    E["necessity_control_RND_A1_direct"] = est(outcome("rnd1707_a1_persistent", harm_direct_ref, "comp", labels))
    E["necessity_control_RND_A1_all_refusing"] = est(outcome("rnd1707_a1_persistent", harm_all_ref, "comp", labels))
    for arm, key in (("a2_natural", "sufficiency_A2"), ("rb_a2_natural", "sufficiency_RB_A2"), ("rd_a2_natural", "sufficiency_RD_A2"), ("rnd1707_a2_natural", "sufficiency_RND_A2")):
        E[key + "_fictional_complying"] = est(outcome(arm, harm_fict_comp, "ref", labels)); E[key + "_fresh_complying"] = est(outcome(arm, fresh_comp, "ref", labels))
    for arm, key in (("a3_benign", "benign_A3"), ("rb_a3_benign", "benign_RB_A3"), ("rd_a3_benign", "benign_RD_A3"), ("rnd1707_a3_benign", "benign_RND_A3")):
        E[key + "_refusal_like"] = est(outcome(arm, benign, "ref", labels))
    def route_share(arm_key, rb_key):
        a, b = E[arm_key], E[rb_key]
        if a.get("n") and a["ci95"][0] > 0 and a["estimate"] >= 0.10: return {"route_share": 1 - b["estimate"] / a["estimate"], "arm_effect": a["estimate"], "rb_effect": b["estimate"]}
        return {"route_share": None, "reason": "arm effect interval includes 0 or size < 0.10", "arm_effect": a.get("estimate"), "rb_effect": b.get("estimate")}
    E["route_share_A2_fictional"] = route_share("sufficiency_A2_fictional_complying", "sufficiency_RB_A2_fictional_complying")
    E["route_share_A3_benign"] = route_share("benign_A3_refusal_like", "benign_RB_A3_refusal_like")
    return E
primary = all_estimands(lab_t1)
sens_labels = dict(lab_t1); changed = 0
for rid, l in lab_2.items():
    if rid in sens_labels and ref(l) != ref(sens_labels[rid]): sens_labels[rid] = l; changed += 1
sensitivity = all_estimands(sens_labels)
# continuity: numeric deltas per arm vs baseline (public fields)
numeric = {}
for arm in sorted({r["arm_id"] for r in recs.values()} - {"baseline"}):
    d_ps, d_pr = [], []
    for r in recs.values():
        if r["arm_id"] != arm: continue
        b = rows[r["row_id"]]
        if isinstance(r.get("primary_score"), (int, float)) and isinstance(b.get("primary_score"), (int, float)): d_ps.append(r["primary_score"] - b["primary_score"])
        if isinstance(r.get("probe_value"), (int, float)) and isinstance(b.get("probe_value"), (int, float)): d_pr.append(r["probe_value"] - b["probe_value"])
    numeric[arm] = {"n": len(d_ps), "median_delta_primary_score": statistics.median(d_ps) if d_ps else None, "median_delta_probe": statistics.median(d_pr) if d_pr else None}
# label distributions per arm
dist = {arm: dict(sorted(collections.Counter(lab_t1[r["record_id"]] for r in recs.values() if r["arm_id"] == arm).items())) for arm in sorted({r["arm_id"] for r in recs.values()} - {"baseline"})}
record = {"schema_version": 1, "note": "cross-check from public files; not the frozen analyzer", "inputs_sha256": {n: hashlib.sha256(open(OUT / n, "rb").read()).hexdigest() for n in ("s16_phase_c_tier1_annotations.csv", "s16_phase_c_baseline_annotations.csv", "s16_phase_c_second_rater_annotations.csv")},
          "bootstrap": {"unit": "intent", "seed": SEED, "replicates": REPS}, "primary_agent_labels": primary, "sensitivity_barkin_reclassified": {"records_reclassified": changed, **sensitivity}, "label_distribution_by_arm": dist, "numeric_continuity_median_deltas": numeric}
record["payload_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != "payload_sha256"}, sort_keys=True).encode()).hexdigest()
json.dump(record, open(OUT / "s16_phase_c_tier1_crosscheck.json", "w"), indent=1, sort_keys=True)
f = lambda e: f"{e['count']}/{e['n']} = {e['estimate']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]" if e.get("n") else "n/a"
md = ["# Phase C Tier-1 cross-check of preregistered label estimands (public files; not the frozen analyzer)", "", f"Payload `{record['payload_sha256'][:16]}…`. Intent-cluster bootstrap, seed {SEED}, {REPS} replicates. Sensitivity = {changed} audited records re-classified to Barkin's binary class.", "", "| Estimand | Primary (agent labels) | Sensitivity |", "|---|---|---|"]
for k in [k for k in primary if k not in ("denominators",) and isinstance(primary[k], dict) and "estimate" in primary[k]]: md.append(f"| {k} | {f(primary[k])} | {f(sensitivity[k])} |")
md += ["", f"Denominators: {primary['denominators']}", "", f"Route share A2 (fictional): {primary['route_share_A2_fictional']}", f"Route share A3 (benign): {primary['route_share_A3_benign']}", "", "| Arm | label distribution | median Δ next-token score | median Δ probe |", "|---|---|---|---|"]
for arm in dist: md.append(f"| {arm} | {dist[arm]} | {numeric[arm]['median_delta_primary_score']:.3f} | {numeric[arm]['median_delta_probe']:.3f} |")
(STEP / "PHASE_C_TIER1_CROSSCHECK.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
