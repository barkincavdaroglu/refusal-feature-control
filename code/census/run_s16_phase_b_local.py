#!/usr/bin/env python3
"""Run the S16 Phase-B forward-only activation census on the local Mac."""
from __future__ import annotations
import argparse, importlib.metadata, math, os, pathlib, platform, sys, time, traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
import numpy as np
import s16_phase_b_contract as contract
from run_s16_phase_b import _authenticate_model_snapshot, _install_wrapper, _write_shard, batch_tokens, feature_values, feature_values_rowwise_all, load_feature_row, load_token_bundle, reference_rows
SCHEMA = "s16-phase-b-local-run-manifest/1"
AUDIT_ROWS = [493, 825, 1671, 4285]
TOKEN_BUNDLE_SHA256 = "663cb36353713c5dc70c16ae4d8068bb4b8d8645df0244e76f0cc482f7e4d948"
PROBE_DIRECTION_SHA256 = "3d2048a873ed05b334545e703e0253707aad9b3de76176eb1fc9d8a71b30d72e"
def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _log(path: pathlib.Path, message: str) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _parse_rows(spec: str | None, count: int) -> list[int]:
    if spec is None:
        return list(range(count))
    if ":" in spec and "," not in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise contract.ContractError("--rows must be START:STOP or comma-separated indices")
        start, stop = (int(value) for value in parts)
        rows = list(range(start, stop))
    else:
        rows = [int(value) for value in spec.split(",") if value]
    if not rows or len(rows) != len(set(rows)) or min(rows) < 0 or max(rows) >= count:
        raise contract.ContractError("--rows selection is empty, duplicated, or out of range")
    return sorted(rows)


def _feature_values(states, encoder, bias):
    import torch

    if states.dtype == torch.bfloat16:
        return feature_values(states, encoder, bias)
    pre_all = torch.matmul(states, encoder) + bias
    act_all = torch.relu(pre_all)
    final_pre = torch.stack(
        [torch.dot(encoder, states[index, -1]) + bias for index in range(states.shape[0])]
    )
    return pre_all, act_all, final_pre, torch.relu(final_pre)


def _rowwise_values(states, encoder, bias):
    import torch

    if states.dtype == torch.bfloat16:
        return feature_values_rowwise_all(states, encoder, bias)
    pre = torch.stack(
        [torch.stack([torch.dot(encoder, token) + bias for token in row]) for row in states]
    )
    return pre, torch.relu(pre)


def _audit(model, rows, encoder, bias, tolerance: float) -> dict[str, Any]:
    import torch

    captures: list[torch.Tensor] = []

    def hook(value, hook):
        captures.append(value.detach())
        return value

    def run(row_tensor, *, logits: bool):
        captures.clear()
        with torch.inference_mode():
            output = model.run_with_hooks(
                row_tensor,
                return_type="logits" if logits else None,
                fwd_hooks=[(contract.FEATURE_HOOK, hook)],
            )
        _, activation, _, final = _feature_values(captures[0], encoder, bias)
        return output, activation.detach(), final.detach()

    with torch.inference_mode():
        before = model(rows[0].unsqueeze(0), return_type="logits")
    _install_wrapper(model)
    after, first_activation, first_final = run(rows[0].unsqueeze(0), logits=True)
    wrapper_max = float((after.float() - before.float()).abs().max().cpu())
    single_activations = [first_activation[0].float().cpu()]
    final_parity = [
        float((first_activation[0, -1].float() - first_final[0].float()).abs().cpu())
    ]
    for row in rows[1:]:
        _, activation, final = run(row.unsqueeze(0), logits=False)
        single_activations.append(activation[0].float().cpu())
        final_parity.append(float((activation[0, -1].float() - final[0].float()).abs().cpu()))
    stacked = torch.stack(rows)
    _, batched, batched_final = run(stacked, logits=False)
    batch_max = max(
        float((batched[index].float().cpu() - single_activations[index]).abs().max())
        for index in range(4)
    )
    final_parity.extend(
        (batched[:, -1].float().cpu() - batched_final.float().cpu()).abs().tolist()
    )
    _, repeated, _ = run(stacked, logits=False)
    repeat_max = float((repeated.float() - batched.float()).abs().max().cpu())
    return {
        "wrapper_identity_max_abs_logit_difference": wrapper_max, "batch_activation_max_abs_difference": batch_max,
        "vectorized_vs_rowwise_final_max_abs_difference": float(max(final_parity)), "repeat_activation_max_abs_difference": repeat_max,
        "repeat_scope": "four_frozen_audit_rows_same_batched_path", "tolerance": tolerance,
        "normal_batch_schedule_allowed": wrapper_max == 0.0 and batch_max <= tolerance and max(final_parity) <= tolerance,
        "audit_invocations": 7, "audit_prompt_equivalents": 13,
    }


def _input_closure(step_root: pathlib.Path):
    all_work = step_root.parents[1]
    paths = {
        "corpus_manifest": step_root / "outputs/s16_phase_b_corpus_manifest.json", "batch_schedule": step_root / "private/phase_b/s16_phase_b_batch_schedule.json",
        "token_bundle": step_root / "private/phase_b/s16_phase_b_token_bundle.npz", "feature_bundle": step_root / "private/phase_b/s16_phase_b_feature_bundle.safetensors",
        "probe_direction": all_work / "steps/06-lexical-controls-and-causal-tests/outputs/qwen3_4b_refusal_probe_direction.npz", "feature_manifest": all_work / "steps/09-intervention-method-exploration/outputs/s9_e8b_transcoder_rows_manifest.json",
    }
    corpus = contract.load_authenticated_json(paths["corpus_manifest"], schema=contract.CORPUS_SCHEMA)
    schedule = contract.load_authenticated_json(paths["batch_schedule"], schema=contract.BATCH_SCHEDULE_SCHEMA)
    contract.validate_corpus_manifest(corpus)
    contract.validate_batch_schedule(schedule, corpus=corpus)
    expected = {"token_bundle": TOKEN_BUNDLE_SHA256, "feature_bundle": contract.FEATURE_BUNDLE_SHA256,
                "probe_direction": PROBE_DIRECTION_SHA256, "feature_manifest": contract.FEATURE_MANIFEST_SHA256}
    hashes = {name: contract.file_sha256(path) for name, path in paths.items()}
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise contract.ContractError("Phase-B local input file hash differs")
    tokens, offsets = load_token_bundle(paths["token_bundle"], corpus)
    return paths, hashes, corpus, schedule, tokens, offsets


def _schedule(schedule: Mapping[str, Any], selected: set[int], normal: bool):
    grouped = []
    for source in schedule["batches"]:
        indices = [index for index in source["row_indices"] if index in selected]
        if indices:
            grouped.append({"row_indices": indices, "token_count": source["token_count"]})
    if not normal:
        grouped = [
            {"row_indices": [index], "token_count": batch["token_count"]}
            for batch in grouped
            for index in batch["row_indices"]
        ]
    return [{"batch_id": index, **batch} for index, batch in enumerate(grouped)]


def _validate_resume(state: Mapping[str, Any], output_dir: pathlib.Path, batches) -> set[int]:
    expected = {batch["batch_id"]: batch for batch in batches}
    complete = set()
    for record in state.get("completed_batches", []):
        batch = expected.get(record.get("batch_id"))
        path = output_dir / record.get("relative_path", "")
        if (
            batch is None
            or record.get("row_indices") != batch["row_indices"]
            or not path.is_file()
            or contract.file_sha256(path) != record.get("sha256")
        ):
            raise contract.ContractError("Phase-B resumable shard differs")
        complete.add(record["batch_id"])
    return complete


def run_local(args: argparse.Namespace, progress: dict[str, Any]) -> dict[str, Any]:
    progress["stage"] = "input_authentication"
    paths, input_hashes, corpus, schedule, tokens, offsets = _input_closure(args.step_root)
    selected_rows = _parse_rows(args.rows, contract.EXPECTED_ROW_COUNT)
    selected = set(selected_rows)
    output_dir = args.output_dir
    manifest_path = output_dir / "s16_phase_b_run_manifest.json"
    log_path = output_dir / "run.log"
    if manifest_path.exists():
        state = contract.load_authenticated_json(manifest_path, schema=SCHEMA)
        if state["status"] == "complete":
            _log(log_path, "existing complete run authenticated; no model load")
            return state
        if state["status"].startswith("failed:"):
            raise contract.ContractError("failed run requires an explicit new recovery decision")
    else:
        state = {
            "schema_version": SCHEMA, "status": "running", "started_at_utc": progress["started_at_utc"],
            "device_requested": args.device, "dtype": args.dtype, "rows_spec": args.rows or "all",
            "selected_row_count": len(selected_rows), "input_hashes": input_hashes, "completed_batches": [],
            "forward_accounting": {"attempted_scientific_invocations": 0, "completed_unique_rows": 0, "completed_unique_tokens": 0},
            "model_load_count": 0,
        }
        contract.atomic_json(manifest_path, state)
    if any(
        state.get(key) != value
        for key, value in {"device_requested": args.device, "dtype": args.dtype,
                           "rows_spec": args.rows or "all", "selected_row_count": len(selected_rows),
                           "input_hashes": input_hashes}.items()
    ):
        raise contract.ContractError("Phase-B resume configuration differs")

    progress["stage"] = "model_load"
    os.environ.setdefault("HF_HOME", str(args.step_root.parents[1] / "local-runtime/hf_cache"))
    import torch
    from huggingface_hub import snapshot_download
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise contract.ContractError("MPS is unavailable")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    versions = {name: importlib.metadata.version(name) for name in contract.PACKAGE_VERSIONS}
    snapshot = pathlib.Path(
        snapshot_download(
            contract.MODEL_ID,
            revision=contract.MODEL_REVISION,
            allow_patterns=list(contract.MODEL_FILES),
            local_files_only=True,
        )
    )
    proof = _authenticate_model_snapshot(
        snapshot,
        {"model": {"snapshot_directory_name": contract.MODEL_REVISION, "files": contract.MODEL_FILES}},
    )
    state["model_load_count"] += 1
    contract.atomic_json(manifest_path, state)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = HookedTransformer.from_pretrained(
        contract.MODEL_ID,
        revision=contract.MODEL_REVISION,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
        dtype=dtype,
        tokenizer=tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder_cpu, _decoder_cpu, bias_cpu = load_feature_row(paths["feature_bundle"])
    encoder = encoder_cpu.to(device=device, dtype=dtype)
    bias = bias_cpu.to(device=device, dtype=dtype)

    progress["stage"] = "audit"
    if state.get("audit") is None:
        audit_tokens = [
            torch.as_tensor(tokens[int(offsets[i]) : int(offsets[i + 1])], dtype=torch.long, device=device)
            for i in AUDIT_ROWS
        ]
        state["audit"] = _audit(model, audit_tokens, encoder, bias, 0.125)
        state["model_snapshot_proof"] = proof
        state["runtime"] = {
            "machine": platform.machine(), "macos": platform.mac_ver()[0],
            "python": sys.version.split()[0], "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "packages": versions,
            "differences_from_v3_pins": {
                name: {"v3": contract.PACKAGE_VERSIONS[name], "actual": value}
                for name, value in versions.items()
                if value != contract.PACKAGE_VERSIONS[name]
            },
        }
        contract.atomic_json(manifest_path, state)
    else:
        _install_wrapper(model)
    if state["audit"]["wrapper_identity_max_abs_logit_difference"] != 0.0:
        raise contract.ContractError("Phase-B wrapper identity audit failed")

    normal = bool(state["audit"]["normal_batch_schedule_allowed"])
    batches = _schedule(schedule, selected, normal)
    state["execution_branch"] = "frozen_equal_length_batches" if normal else "prespecified_batch_size_one_fallback"
    state["branch_scientific_invocation_ceiling"] = math.ceil(1.05 * len(batches))
    complete = _validate_resume(state, output_dir, batches)
    progress["stage"] = "census"
    census_start = time.perf_counter()
    for batch in batches:
        if batch["batch_id"] in complete:
            continue
        account = state["forward_accounting"]
        if len(selected) > 6600 or account["attempted_scientific_invocations"] + 1 > state["branch_scientific_invocation_ceiling"]:
            raise contract.ContractError("Phase-B local forward cap exceeded")
        indices = batch["row_indices"]
        account["attempted_scientific_invocations"] += 1
        contract.atomic_json(manifest_path, state)
        batch_tensor = torch.as_tensor(batch_tokens(tokens, offsets, indices), dtype=torch.long, device=device)
        captures: dict[str, Any] = {}
        counts: dict[str, int] = {}

        def capture(name):
            def hook(value, hook):
                counts[name] = counts.get(name, 0) + 1
                captures[name] = value.detach()
                return value
            return hook

        hooks = [(name, capture(name)) for name in (contract.FEATURE_HOOK, contract.PRIMARY_REFERENCE_HOOK, "blocks.31.hook_resid_pre")]
        with torch.inference_mode():
            model.run_with_hooks(batch_tensor, return_type=None, fwd_hooks=hooks)
        if set(counts) != set(name for name, _ in hooks) or any(value != 1 for value in counts.values()):
            raise contract.ContractError("Phase-B hook fire closure differs")
        pre_vectorized, act_vectorized, final_pre, final_act = _feature_values(captures[contract.FEATURE_HOOK], encoder, bias)
        pre_all, act_all = (pre_vectorized, act_vectorized) if normal else _rowwise_values(captures[contract.FEATURE_HOOK], encoder, bias)
        last_user_pre = torch.stack([
            torch.dot(encoder, captures[contract.FEATURE_HOOK][offset, corpus["rows"][index]["user_token_end_exclusive"] - 1]) + bias
            for offset, index in enumerate(indices)
        ])
        ref_offsets = reference_rows(corpus, indices)
        empty = torch.empty((0, 2560), dtype=dtype)
        tensors = {
            "row_indices": torch.as_tensor(indices, dtype=torch.int64),
            "c_preactivation_all": pre_all.cpu().contiguous(),
            "c_activation_all": act_all.cpu().contiguous(),
            "c_preactivation_final": final_pre.cpu().contiguous(),
            "c_activation_final": final_act.cpu().contiguous(),
            "c_preactivation_last_user": last_user_pre.cpu().contiguous(),
            "c_activation_last_user": torch.relu(last_user_pre).cpu().contiguous(),
            "reference_row_indices": torch.as_tensor([indices[offset] for offset in ref_offsets], dtype=torch.int64),
            "reference_resid_post": captures[contract.PRIMARY_REFERENCE_HOOK][ref_offsets, -1].cpu().contiguous() if ref_offsets else empty,
            "reference_resid_pre": captures["blocks.31.hook_resid_pre"][ref_offsets, -1].cpu().contiguous() if ref_offsets else empty,
            "reference_mlp_hook_in": captures[contract.FEATURE_HOOK][ref_offsets, -1].cpu().contiguous() if ref_offsets else empty,
        }
        token_count = len(indices) * batch["token_count"]
        record = {
            "batch_id": batch["batch_id"],
            "row_indices": indices,
            "prompt_equivalents": len(indices),
            "token_equivalents": token_count,
            **_write_shard(output_dir, batch["batch_id"], tensors),
        }
        state["completed_batches"].append(record)
        state["completed_batches"].sort(key=lambda value: value["batch_id"])
        account["completed_unique_rows"] += len(indices)
        account["completed_unique_tokens"] += token_count
        contract.atomic_json(manifest_path, state)
    census_seconds = time.perf_counter() - census_start
    state.update({
        "status": "complete",
        "finished_at_utc": _utc(),
        "wall_seconds": time.perf_counter() - progress["started_monotonic"],
        "census_wall_seconds": census_seconds,
        "scientific_tokens_per_second": state["forward_accounting"]["completed_unique_tokens"] / census_seconds,
        "totals": {
            "rows": state["forward_accounting"]["completed_unique_rows"],
            "tokens": state["forward_accounting"]["completed_unique_tokens"],
            "shards": len(state["completed_batches"]),
        },
    })
    contract.atomic_json(manifest_path, state)
    _log(log_path, f"complete rows={state['totals']['rows']} tokens={state['totals']['tokens']} shards={state['totals']['shards']}")
    return contract.load_authenticated_json(manifest_path, schema=SCHEMA)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--rows")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args(argv)
    if args.dtype == "fp32" and args.device != "cpu":
        parser.error("fp32 is permitted only with --device cpu")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.step_root = args.step_root.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = args.output_dir.resolve(strict=True)
    progress = {"stage": "startup", "started_at_utc": _utc(), "started_monotonic": time.perf_counter()}
    try:
        run_local(args, progress)
        return 0
    except Exception:
        detail = traceback.format_exc()
        sys.stderr.write(detail)
        log_path = args.output_dir / "run.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(detail)
        manifest_path = args.output_dir / "s16_phase_b_run_manifest.json"
        state = {"schema_version": SCHEMA, "started_at_utc": progress["started_at_utc"]}
        if manifest_path.exists():
            try:
                state = contract.load_authenticated_json(manifest_path, schema=SCHEMA)
            except Exception:
                pass
        state.update({"status": f"failed:{progress['stage']}", "finished_at_utc": _utc(), "wall_seconds": time.perf_counter() - progress["started_monotonic"]})
        contract.atomic_json(manifest_path, state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
