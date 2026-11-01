#!/usr/bin/env python3
"""Hash-verified, post-hoc S16/S10 concordance; emits no prompt text."""
import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from safetensors import safe_open
from s16_phase_b_contract import atomic_json
SEED, REPS = 2522210527409210963, 10_000
FRAMES = ("D0", "D1", "D2", "F0", "F1", "F2")
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def load_json(path):
    with path.open() as file: return json.load(file)
def read_csv(path):
    with path.open(newline="") as file: return list(csv.DictReader(file))
def ranks(values):
    order, out, i = np.argsort(values, kind="mergesort"), np.empty(len(values)), 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]: j += 1
        out[order[i:j]], i = (i + j + 1) / 2, j
    return out
def spearman(rows):
    x, y = (np.array([r[key] for r in rows]) for key in ("c", "score"))
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])
def auc(rows):
    pos = np.array([r["c"] for r in rows if r["refusal"]])
    neg = np.array([r["c"] for r in rows if not r["refusal"]])
    if not len(pos) or not len(neg): return float("nan")
    return float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))
def table(rows):
    return {a: {y: sum(r["active"] == av and r["refusal"] == yv for r in rows)
                for y, yv in (("refusal_like", True), ("compliance_like", False))}
            for a, av in (("active", True), ("silent", False))}
def risk_difference(rows):
    rate = [np.mean([r["refusal"] for r in rows if r["active"] == active]) for active in (True, False)]
    return float(rate[0] - rate[1])
def cluster_ci(rows, statistic):
    groups = defaultdict(list)
    for row in rows: groups[row["intent_id"]].append(row)
    ids, rng, values = sorted(groups), np.random.Generator(np.random.PCG64(SEED)), []
    for _ in range(REPS):
        value = statistic([r for key in rng.choice(ids, len(ids), replace=True) for r in groups[key]])
        if np.isfinite(value): values.append(value)
    return {"lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5)),
            "defined_replicates": len(values), "requested_replicates": REPS}
def activation_summary(rows):
    values = np.array([r["c"] for r in rows])
    return {"n": len(rows), "active_count": int((values > 0).sum()),
            "active_fraction": float((values > 0).mean()), "mean": float(values.mean())}
def load_activations(step, indices):
    run = load_json(step / "outputs/phase_b_local/s16_phase_b_run_manifest.json")
    wanted, found = set(indices), {}
    for batch in run["completed_batches"]:
        if not wanted.intersection(batch["row_indices"]): continue
        path = step / "outputs/phase_b_local" / batch["relative_path"]
        if sha(path) != batch["sha256"]: raise ValueError(f"shard hash mismatch: batch {batch['batch_id']}")
        with safe_open(path, framework="pt", device="cpu") as file:
            rows = file.get_tensor("row_indices").tolist()
            values = file.get_tensor("c_activation_final").float().tolist()
        found.update({int(i): float(v) for i, v in zip(rows, values) if i in wanted})
    if set(found) != wanted: raise ValueError(f"missing authenticated activations: {len(wanted - set(found))}")
    return found
def main():
    root, step = Path(__file__).parents[3], Path(__file__).parents[1]
    s10, s1 = root / "steps/10-target-and-dataset-repair", root / "steps/01-verify-refusal-phenomenon"
    files = {"corpus": step / "outputs/s16_phase_b_corpus_manifest.json",
             "run": step / "outputs/phase_b_local/s16_phase_b_run_manifest.json",
             "validation_generation": s10 / "outputs/s10_phase_b_validation_generation.json",
             "expansion_generation": s10 / "outputs/s10_phase_b_expansion_generation.json",
             "validation_annotations": s10 / "outputs/s10_phase_b_validation_annotations.csv",
             "expansion_annotations": s10 / "outputs/s10_phase_b_expansion_annotations.csv",
             "validation_prompts_private": s10 / "private/s10_phase_b_prompts.csv",
             "expansion_prompts_private": s10 / "private/s10_phase_b_validation_expansion_prompts.csv",
             "s1_machine": s1 / "outputs/qwen3_4b_pilot_machine_summary.json",
             "s1_annotations": s1 / "outputs/qwen3_4b_manual_annotations.csv"}
    census_rows = load_json(files["corpus"])["rows"]
    census = {(r["intent_id"], r["variant_id"]): r for r in census_rows
              if r["source_id"] in {"s10_validation_prompts", "s10_validation_expansion_prompts"}}
    scores = sum((load_json(files[k])["scores"] for k in ("validation_generation", "expansion_generation")), [])
    prompts = {r["prompt_key"]: r for k in ("validation_prompts_private", "expansion_prompts_private") for r in read_csv(files[k])}
    anns = {r["prompt_key"]: r for k in ("validation_annotations", "expansion_annotations") for r in read_csv(files[k])}
    selected = list(census.values()) + [r for r in census_rows if
        (r["variant_id"] == "instruction_override" and r["risk_class"] == "harmful") or
        r["variant_id"] == "unsafe_model_simulation" or r["source_id"] == "s16_fresh_reference_pairs"]
    activations, joined, excluded = load_activations(step, [r["row_index"] for r in selected]), [], []
    for score in scores:
        key, reasons = score["prompt_key"], []
        prompt, ann, row = prompts.get(key), anns.get(key), census.get((score["intent_id"], score["frame_id"]))
        digest = hashlib.sha256(prompt["prompt"].encode()).hexdigest() if prompt else None
        if not prompt or not ann or not row: reasons.append("missing_required_record")
        else:
            checks = {"prompt_csv_hash": digest == prompt["prompt_sha256"], "generation_hash": digest == score["prompt_sha256"],
                      "census_hash": digest == row["raw_prompt_sha256"], "completion_hash": ann["completion_sha256"] == score["completion_sha256"],
                      "identity": (prompt["intent_id"], prompt["frame_id"], prompt["risk_class"]) == (score["intent_id"], score["frame_id"], score["risk_class"]),
                      "label": ann["manual_label"] in "RSPC"}
            reasons.extend(name for name, passed in checks.items() if not passed)
        if reasons: excluded.append({"prompt_key": key, "reasons": reasons}); continue
        label, value = ann["manual_label"], activations[row["row_index"]]
        joined.append({"intent_id": score["intent_id"], "frame": score["frame_id"], "risk": score["risk_class"],
                       "refusal": label in "RS", "c": value, "active": value > 0, "score": float(score["primary_score"])})
    harmful, benign = ([r for r in joined if r["risk"] == risk] for risk in ("harmful", "benign"))
    by_key, pairs = {(r["intent_id"], r["frame"]): r for r in harmful}, Counter()
    for intent in sorted({r["intent_id"] for r in harmful}):
        for k in range(3):
            direct, fiction = by_key[(intent, f"D{k}")], by_key[(intent, f"F{k}")]
            pairs[(direct["refusal"] and not fiction["refusal"], direct["active"] and not fiction["active"])] += 1
    rng, observed, extreme = np.random.Generator(np.random.PCG64(SEED)), spearman(harmful), 0
    intent_ids = sorted({r["intent_id"] for r in harmful})
    c_matrix = np.array([[by_key[(intent, frame)]["c"] for frame in FRAMES] for intent in intent_ids])
    score_rows = sorted(harmful, key=lambda r: (r["intent_id"], FRAMES.index(r["frame"])))
    for _ in range(REPS):
        permuted = [dict(row, c=float(value)) for row, value in zip(score_rows, c_matrix[rng.permutation(len(c_matrix))].ravel())]
        extreme += abs(spearman(permuted)) >= abs(observed)
    s1_labels = {r["prompt_key"]: r["manual_label"] for r in read_csv(files["s1_annotations"])}
    s1_keys = {r["prompt_key"] for r in load_json(files["s1_machine"])["completions"]}
    side = {}
    for name, variant, risk in (("instruction_override_harmful", "instruction_override", "harmful"),
                                ("unsafe_model_simulation_harmful", "unsafe_model_simulation", "harmful"),
                                ("unsafe_model_simulation_benign", "unsafe_model_simulation", "benign")):
        rows = [dict(r, c=activations[r["row_index"]]) for r in selected if r["variant_id"] == variant and r["risk_class"] == risk]
        labels = [s1_labels[k] for r in rows if (k := r["source_record_key"].split(":", 1)[-1]) in s1_keys and k in s1_labels]
        side[name] = activation_summary(rows) | {"s1_behavior_matched_count": len(labels), "s1_manual_label_counts": dict(sorted(Counter(labels).items()))}
    aucs = {name: {"estimate": auc(rows), "cluster_bootstrap_95ci": cluster_ci(rows, auc)} for name, rows in
            (("all_harmful", harmful), ("direct", [r for r in harmful if r["frame"][0] == "D"]),
             ("fictional", [r for r in harmful if r["frame"][0] == "F"]))}
    flips, silenced = sum(n for (f, _), n in pairs.items() if f), sum(n for (_, s), n in pairs.items() if s)
    fresh = [dict(r, c=activations[r["row_index"]]) for r in selected if r["source_id"] == "s16_fresh_reference_pairs"]
    result = {"schema_version": 1, "status": "complete_posthoc_exploratory",
      "claim_boundary": "No gate, threshold, selectivity, causal, or Phase-C-authority claim.",
      "bootstrap": {"unit": "intent", "method": "percentile", "rng": "NumPy PCG64", "seed": SEED, "replicates": REPS},
      "input_sha256": {key: sha(path) for key, path in files.items()},
      "join_verification": {"requested": len(scores), "included": len(joined), "excluded": excluded, "all_checks_passed": not excluded},
      "harmful": {"n": len(harmful), "intent_count": len(intent_ids), "active_by_behavior": table(harmful),
        "risk_difference": {"estimate": risk_difference(harmful), "cluster_bootstrap_95ci": cluster_ci(harmful, risk_difference)}, "auc": aucs,
        "paired_direct_fictional": {"n": sum(pairs.values()), "counts": {f"flip_{str(f).lower()}__silenced_{str(s).lower()}": pairs[(f, s)] for f in (True, False) for s in (True, False)},
          "p_silenced_given_flip": pairs[(True, True)] / flips, "p_flip_given_silenced": pairs[(True, True)] / silenced},
        "frames": {frame: activation_summary(rows := [r for r in harmful if r["frame"] == frame]) |
                   {"refusal_like_fraction": float(np.mean([r["refusal"] for r in rows]))} for frame in FRAMES},
        "spearman_primary_score": {"rho": observed, "intent_vector_permutation_p_two_sided": (extreme + 1) / (REPS + 1)}},
      "benign": {"n": len(benign), "intent_count": len({r["intent_id"] for r in benign}), "active_by_behavior": table(benign)},
      "side_findings": side, "fresh_reference_pairs": activation_summary(fresh) | {"behavior_labels_available": False}}
    output = step / "outputs/phase_b_local/s16_phase_b_behavior_concordance.json"
    atomic_json(output, result)
    print(json.dumps({"status": "complete", "output": str(output), "included": len(joined), "excluded": len(excluded)}))
if __name__ == "__main__": main()
