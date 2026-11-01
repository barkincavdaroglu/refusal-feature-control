"""Amendment-02 second-rater audit for Phase C Tier 1.

Reads the private key and Barkin's blind labels, writes the public second-rater annotation CSV
(record_id, completion_sha256, manual_label) and a public-safe audit JSON/MD with agreement, binary
Cohen's kappa, bootstrap CIs (flip set / non-flip set / pooled), per-arm breakdown, and the list of
records whose binary class Barkin assigned differently (input to the declared sensitivity estimates).
No prompt or completion text is read or written. Touches no frozen file.
"""
import csv, hashlib, json, pathlib, random, statistics, sys
STEP = pathlib.Path(__file__).resolve().parents[1]; OUT = STEP / "outputs/phase_c_local"; PRIV = OUT / "private"
KEY, LABELS = PRIV / "second_rater_blind_key.json", PRIV / "second_rater_labels_blind.csv"
PUBLIC_CSV = OUT / "s16_phase_c_second_rater_annotations.csv"
AUDIT_JSON, AUDIT_MD = OUT / "s16_phase_c_tier1_second_rater_audit.json", STEP / "PHASE_C_TIER1_SECOND_RATER_AUDIT.md"
BENCH_KAPPA, FLIP_MIN_AGREE, SEED, REPS = 0.8361096, 0.80, 2522210527409210963, 10000
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
binary = lambda lab: {"R": True, "S": True, "P": False, "C": False}.get(lab)

def kappa(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n == 0: return None, 0
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (None if pe == 1 else (po - pe) / (1 - pe)), n

def stats(items, rng):
    pairs = [(binary(i["barkin"]), binary(i["agent"])) for i in items]
    valid = [(a, b) for a, b in pairs if a is not None and b is not None]
    exact = sum(i["barkin"] == i["agent"] for i in items) / len(items)
    bin_agree = sum(a == b for a, b in valid) / len(valid) if valid else None
    k, n = kappa(valid)
    boots_k, boots_a = [], []
    for _ in range(REPS):
        sample = [valid[rng.randrange(len(valid))] for _ in valid]
        kk, _ = kappa(sample); boots_k.append(kk); boots_a.append(sum(a == b for a, b in sample) / len(sample))
    ks = sorted(x for x in boots_k if x is not None); as_ = sorted(boots_a)
    ci = lambda xs: [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]] if xs else None
    return {"n": len(items), "n_binary_valid": n, "exact_agreement": exact, "binary_agreement": bin_agree, "binary_agreement_ci95": ci(as_),
            "binary_kappa": k, "binary_kappa_ci95": ci(ks), "kappa_replicates_defined": len(ks), "u_count": sum(i["barkin"] == "U" for i in items)}

def main() -> int:
    key = json.load(open(KEY)); labels = {r["blind_id"]: r for r in csv.DictReader(open(LABELS))}
    recs = key["records"]; assert len(recs) == 55 and len(labels) == 55, (len(recs), len(labels))
    items = []
    for r in recs:
        lab = labels[r["blind_id"]]["manual_label"].strip().upper(); assert lab in {"R", "S", "P", "C", "U"}, lab
        items.append({**{k: r[k] for k in ("blind_id", "record_id", "row_id", "arm_id", "flip", "completion_sha256", "agent_label", "baseline_label")}, "barkin": lab, "agent": r["agent_label"], "note": labels[r["blind_id"]].get("annotation_notes", "")})
    # 1) public second-rater annotations (format of the baseline annotations table)
    with open(PUBLIC_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["record_id", "completion_sha256", "manual_label"]); w.writeheader()
        for i in sorted(items, key=lambda x: x["record_id"]): w.writerow({"record_id": i["record_id"], "completion_sha256": i["completion_sha256"], "manual_label": i["barkin"]})
    print("wrote", PUBLIC_CSV.name, sha(PUBLIC_CSV)[:16])
    # 2) agreement statistics
    rng = random.Random(SEED)
    flips, nonflips = [i for i in items if i["flip"]], [i for i in items if not i["flip"]]
    res = {"all_55": stats(items, rng), "flip_18": stats(flips, rng), "nonflip_37": stats(nonflips, rng)}
    per_arm = {}
    for arm in sorted({i["arm_id"] for i in items}):
        sub = [i for i in items if i["arm_id"] == arm]
        per_arm[arm] = {"n": len(sub), "n_flip": sum(i["flip"] for i in sub), "exact_agreement": sum(i["barkin"] == i["agent"] for i in sub) / len(sub),
                        "binary_agreement": sum(binary(i["barkin"]) == binary(i["agent"]) for i in sub if binary(i["barkin"]) is not None) / max(1, sum(binary(i["barkin"]) is not None for i in sub))}
    flip_sides = {"confirms_agent_flip": sum(binary(i["barkin"]) == binary(i["agent"]) for i in flips if binary(i["barkin"]) is not None),
                  "sides_with_baseline_class": sum(binary(i["barkin"]) == binary(i["baseline_label"]) and binary(i["barkin"]) != binary(i["agent"]) for i in flips if binary(i["barkin"]) is not None),
                  "u": sum(i["barkin"] == "U" for i in flips)}
    disagreements = [{k: i[k] for k in ("record_id", "row_id", "arm_id", "flip", "agent_label", "baseline_label")} | {"barkin_label": i["barkin"]} for i in items if binary(i["barkin"]) is not None and binary(i["barkin"]) != binary(i["agent"])]
    within_class = [{k: i[k] for k in ("record_id", "arm_id", "flip", "agent_label")} | {"barkin_label": i["barkin"]} for i in items if i["barkin"] != i["agent"] and binary(i["barkin"]) == binary(i["agent"])]
    pooled_ci = res["all_55"]["binary_kappa_ci95"]
    rule = {"benchmark_kappa": BENCH_KAPPA, "flip_set_binary_agreement": res["flip_18"]["binary_agreement"], "flip_set_min_agreement": FLIP_MIN_AGREE,
            "flip_set_below_threshold": res["flip_18"]["binary_agreement"] < FLIP_MIN_AGREE, "pooled_kappa_ci_excludes_benchmark_from_below": bool(pooled_ci and pooled_ci[1] < BENCH_KAPPA),
            "sensitivity_reclassified_record_count": len(disagreements)}
    rule["headline_is_sensitivity"] = rule["flip_set_below_threshold"]
    record = {"schema_version": 1, "amendment_02_sha256": key.get("amendment_sha256"), "inputs_sha256": {"key": sha(KEY), "labels_csv": sha(LABELS), "public_second_rater_csv": sha(PUBLIC_CSV)},
              "selection": key["selection"], "label_distribution": {"barkin": dict(sorted(statistics.Counter(i["barkin"] for i in items).items())) if hasattr(statistics, "Counter") else None},
              "agreement": res, "per_arm": per_arm, "flip_sides": flip_sides, "binary_disagreements": disagreements, "within_class_disagreements": within_class, "rule_5_7": rule,
              "bootstrap": {"seed": SEED, "replicates": REPS, "unit": "item", "percentile": [2.5, 97.5]}}
    import collections; record["label_distribution"] = {"barkin": dict(sorted(collections.Counter(i["barkin"] for i in items).items())), "agent_on_same_items": dict(sorted(collections.Counter(i["agent"] for i in items).items()))}
    record["payload_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != "payload_sha256"}, sort_keys=True).encode()).hexdigest()
    json.dump(record, open(AUDIT_JSON, "w"), indent=1, sort_keys=True); print("wrote", AUDIT_JSON.name, "payload", record["payload_sha256"])
    f = lambda x: "n/a" if x is None else f"{x:.3f}"; c = lambda ci: "n/a" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    md = ["# S16 Phase C Tier 1 second-rater audit (amendment 02)", "", f"Public-safe: no prompt or completion text. Source JSON `outputs/phase_c_local/s16_phase_c_tier1_second_rater_audit.json` (payload `{record['payload_sha256'][:16]}…`). Selection: all {key['selection']['flip_pool_count']} agent-labelled flips in the five primary arms plus {key['selection']['selected_nonflip_count']} hash-ranked non-flips, 55 items, labelled blind by Barkin.", "",
          "| Set | n | Exact agreement | Binary agreement [95% CI] | Binary kappa [95% CI] | U |", "|---|---|---|---|---|---|"]
    for name, r in res.items(): md.append(f"| {name} | {r['n']} | {f(r['exact_agreement'])} | {f(r['binary_agreement'])} {c(r['binary_agreement_ci95'])} | {f(r['binary_kappa'])} {c(r['binary_kappa_ci95'])} | {r['u_count']} |")
    md += ["", f"Benchmark (baseline audit) binary kappa: {BENCH_KAPPA:.4f}. Flip-set binary agreement {f(rule['flip_set_binary_agreement'])} vs minimum {FLIP_MIN_AGREE:.2f}: {'BELOW threshold, sensitivity estimates become the headline' if rule['flip_set_below_threshold'] else 'at or above threshold'}. Pooled kappa CI excludes the benchmark from below: {rule['pooled_kappa_ci_excludes_benchmark_from_below']}.", "",
           f"Flips ({len(flips)}): Barkin confirms the agent's flip in {flip_sides['confirms_agent_flip']}, sides with the baseline class in {flip_sides['sides_with_baseline_class']}, U in {flip_sides['u']}.", "",
           "| Arm | n | flips | Exact | Binary |", "|---|---|---|---|---|"] + [f"| {a} | {v['n']} | {v['n_flip']} | {f(v['exact_agreement'])} | {f(v['binary_agreement'])} |" for a, v in per_arm.items()]
    md += ["", f"Binary disagreements (re-classified to Barkin's class in the declared sensitivity estimates): {len(disagreements)}", "", "| record_id | arm | flip | agent | baseline | Barkin |", "|---|---|---|---|---|---|"] + [f"| {d['record_id']} | {d['arm_id']} | {d['flip']} | {d['agent_label']} | {d['baseline_label']} | {d['barkin_label']} |" for d in disagreements]
    md += ["", f"Within-class disagreements (R vs S or P vs C only): {len(within_class)}; labels: Barkin {record['label_distribution']['barkin']}, agent on the same items {record['label_distribution']['agent_on_same_items']}."]
    AUDIT_MD.write_text("\n".join(md) + "\n"); print("wrote", AUDIT_MD.name)
    print(json.dumps({"agreement": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("n", "exact_agreement", "binary_agreement", "binary_kappa", "binary_kappa_ci95", "u_count")} for k, v in res.items()}, "flip_sides": flip_sides, "rule": rule, "per_arm": per_arm}, indent=1, default=str))
    return 0

if __name__ == "__main__":
    sys.exit(main())
