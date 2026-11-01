#!/usr/bin/env python3
"""Build E8b's authenticated compact C/M/D transcoder-row bundle.

Only three rows are needed by the frozen E8a design: nominated source C,
equal-L2 matched source M, and receiver diagnostic D.  The large upstream
layer files are never downloaded wholesale.  This builder delegates the
already-tested safetensors header/range logic to E5b's range helper and records
the hash of every semantic input and fetched range in a public manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests
import torch
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from safetensors import safe_open

from extract_s9_e8b_transcoder_rows import (
    SOURCE_REPO,
    SOURCE_REVISION,
    TENSOR_NAMES,
    _fetch_tensor_rows,
    _load_header,
)
from s9_contract import canonical_sha256, sha256_file, validate_stamped_payload
from s9_e8_contract import validate_e8a_manifest, validate_e8a_plan


BUNDLE_SCHEMA_VERSION = 1
BUNDLE_SEMANTICS_VERSION = "s9-e8b-transcoder-row-bundle-v1"
BUNDLE_STATUS = "complete_authenticated_e8b_transcoder_row_bundle"
RANGE_HELPER_NAME = "extract_s9_e8b_transcoder_rows.py"

REQUIRED_FEATURE_ROWS: dict[str, tuple[int, int, str]] = {
    "L31:F49541": (31, 49541, "C"),
    "L31:F139261": (31, 139261, "M"),
    "L32:F50401": (32, 50401, "D"),
}
EXPECTED_BUNDLE_KEYS = frozenset(
    {
        f"layer_{layer}.{name}"
        for layer in (31, 32)
        for name in (*TENSOR_NAMES, "feature_indices")
    }
)
_SAFETENSORS_DTYPES = {
    torch.bfloat16: "BF16",
    torch.int64: "I64",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    payload["payload_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def save_deterministic_safetensors(
    tensors: Mapping[str, torch.Tensor],
    path: Path,
    *,
    metadata: Mapping[str, str],
) -> None:
    """Write a canonical safetensors file with stable key and data ordering."""

    if not tensors:
        raise ValueError("A deterministic safetensors bundle cannot be empty")
    header: dict[str, Any] = {
        "__metadata__": {key: str(metadata[key]) for key in sorted(metadata)}
    }
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().contiguous().cpu()
        dtype = _SAFETENSORS_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported deterministic safetensors dtype: {tensor.dtype}")
        blob = tensor.view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"".join(blobs))


def required_feature_rows(
    e8a_manifest: Mapping[str, Any], e8a_plan: Mapping[str, Any]
) -> dict[int, list[int]]:
    """Validate E8a and return its exact, non-expandable C/M/D row request."""

    validate_e8a_manifest(e8a_manifest)
    validate_e8a_plan(e8a_plan, e8a_manifest)
    route = e8a_plan["route_definition"]
    observed = {
        route["nominated_source_feature"]: "C",
        route["control_source_feature"]: "M",
        route["nominated_receiver_feature"]: "D",
    }
    expected = {
        feature: specification[2]
        for feature, specification in REQUIRED_FEATURE_ROWS.items()
    }
    if observed != expected:
        raise ValueError("E8b C/M/D feature identities differ from frozen E8a")
    return {
        31: [49541, 139261],
        32: [50401],
    }


def _feature_manifest_rows() -> list[dict[str, Any]]:
    return [
        {
            "feature": feature,
            "layer": layer,
            "feature_index": index,
            "role": role,
        }
        for feature, (layer, index, role) in REQUIRED_FEATURE_ROWS.items()
    ]


def validate_transcoder_bundle_manifest(
    payload: Mapping[str, Any],
    *,
    e8a_manifest: Mapping[str, Any],
    e8a_plan: Mapping[str, Any],
    e8a_manifest_path: Path | None = None,
    e8a_plan_path: Path | None = None,
    bundle_path: Path | None = None,
) -> None:
    """Authenticate a completed compact bundle manifest without model code."""

    validate_stamped_payload(payload)
    required_feature_rows(e8a_manifest, e8a_plan)
    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("E8b transcoder bundle schema differs")
    if payload.get("analysis_semantics_version") != BUNDLE_SEMANTICS_VERSION:
        raise ValueError("E8b transcoder bundle semantics differ")
    if payload.get("status") != BUNDLE_STATUS:
        raise ValueError("E8b transcoder bundle is incomplete")
    if payload.get("features") != _feature_manifest_rows():
        raise ValueError("E8b transcoder bundle feature closure differs")
    if payload.get("source") != {
        "repo_id": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "full_source_files_downloaded": False,
        "http_range_required": True,
    }:
        raise ValueError("E8b transcoder source contract differs")
    if payload.get("counts") != {
        "layers": 2,
        "unique_features": 3,
        "bundle_tensors": 8,
    }:
        raise ValueError("E8b transcoder bundle counts differ")
    if set(payload.get("bundle", {}).get("keys", [])) != set(EXPECTED_BUNDLE_KEYS):
        raise ValueError("E8b transcoder bundle key closure differs")
    producer = Path(__file__).resolve()
    if payload.get("producer") != {
        "path": producer.name,
        "file_sha256": sha256_file(producer),
    }:
        raise ValueError("E8b transcoder bundle producer bytes differ")
    if payload.get("e8a_manifest", {}).get("payload_sha256") != e8a_manifest.get(
        "payload_sha256"
    ):
        raise ValueError("E8b transcoder bundle references another E8a manifest")
    if payload.get("e8a_plan", {}).get("payload_sha256") != e8a_plan.get(
        "payload_sha256"
    ):
        raise ValueError("E8b transcoder bundle references another E8a plan")
    helper = Path(__file__).with_name(RANGE_HELPER_NAME)
    if payload.get("range_helper") != {
        "path": RANGE_HELPER_NAME,
        "file_sha256": sha256_file(helper),
    }:
        raise ValueError("E8b transcoder bundle used other range-helper bytes")
    expected_layers = required_feature_rows(e8a_manifest, e8a_plan)
    layers = payload.get("layers")
    if not isinstance(layers, list) or len(layers) != len(expected_layers):
        raise ValueError("E8b transcoder layer provenance closure differs")
    by_layer = {
        record.get("layer"): record
        for record in layers
        if isinstance(record, Mapping)
    }
    if set(by_layer) != set(expected_layers):
        raise ValueError("E8b transcoder layer provenance identities differ")
    for layer, indices in expected_layers.items():
        record = by_layer[layer]
        if (
            record.get("filename") != f"layer_{layer}.safetensors"
            or record.get("resolved_commit") != SOURCE_REVISION
            or record.get("feature_indices") != indices
            or not _is_sha256(record.get("source_file_etag"))
            or not _is_sha256(record.get("safetensors_header_sha256"))
        ):
            raise ValueError(f"E8b transcoder provenance differs at layer {layer}")
        tensor_records = record.get("tensors")
        if not isinstance(tensor_records, Mapping) or set(tensor_records) != set(
            TENSOR_NAMES
        ):
            raise ValueError(f"E8b transcoder tensor provenance differs at layer {layer}")
        for name, tensor_record in tensor_records.items():
            if not isinstance(tensor_record, Mapping):
                raise ValueError("E8b transcoder tensor provenance is malformed")
            ranges = tensor_record.get("ranges")
            if not isinstance(ranges, list) or not ranges:
                raise ValueError("E8b transcoder range provenance is missing")
            for range_record in ranges:
                if (
                    not isinstance(range_record, Mapping)
                    or not _is_sha256(range_record.get("sha256"))
                    or not isinstance(range_record.get("byte_start"), int)
                    or not isinstance(range_record.get("byte_end"), int)
                    or range_record["byte_start"] < 0
                    or range_record["byte_end"] < range_record["byte_start"]
                ):
                    raise ValueError("E8b transcoder range provenance is malformed")
    if e8a_manifest_path is not None and payload["e8a_manifest"].get(
        "file_sha256"
    ) != sha256_file(e8a_manifest_path):
        raise ValueError("E8b transcoder bundle E8a manifest file hash differs")
    if e8a_plan_path is not None and payload["e8a_plan"].get(
        "file_sha256"
    ) != sha256_file(e8a_plan_path):
        raise ValueError("E8b transcoder bundle E8a plan file hash differs")
    if bundle_path is not None:
        if payload.get("bundle", {}).get("file_sha256") != sha256_file(bundle_path):
            raise ValueError("E8b transcoder row-bundle hash differs")
        with safe_open(str(bundle_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(EXPECTED_BUNDLE_KEYS):
                raise ValueError("E8b completed transcoder bundle keys differ")
            if handle.metadata() != {
                "source_repo": SOURCE_REPO,
                "source_revision": SOURCE_REVISION,
                "e8a_manifest_payload_sha256": e8a_manifest["payload_sha256"],
                "e8a_plan_payload_sha256": e8a_plan["payload_sha256"],
            }:
                raise ValueError("E8b completed transcoder bundle metadata differs")
            for layer, indices in required_feature_rows(e8a_manifest, e8a_plan).items():
                stored = handle.get_tensor(f"layer_{layer}.feature_indices").tolist()
                if stored != indices:
                    raise ValueError(
                        f"E8b completed transcoder indices differ at layer {layer}"
                    )
                expected_shapes = {
                    "W_dec": [len(indices), 2560],
                    "W_enc": [len(indices), 2560],
                    "b_enc": [len(indices)],
                }
                for name, shape in expected_shapes.items():
                    tensor = handle.get_tensor(f"layer_{layer}.{name}")
                    if tensor.dtype != torch.bfloat16 or list(tensor.shape) != shape:
                        raise ValueError(
                            f"E8b completed transcoder {name} differs at layer {layer}"
                        )


def extract_bundle(
    *,
    e8a_manifest_path: Path,
    e8a_plan_path: Path,
    bundle_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    e8a_manifest = json.loads(e8a_manifest_path.read_text(encoding="utf-8"))
    e8a_plan = json.loads(e8a_plan_path.read_text(encoding="utf-8"))
    by_layer = required_feature_rows(e8a_manifest, e8a_plan)
    if e8a_plan.get("manifest_file_sha256") != sha256_file(e8a_manifest_path):
        raise ValueError("E8a plan manifest file hash differs at E8b extraction")

    tensors: dict[str, torch.Tensor] = {}
    source_layers: list[dict[str, Any]] = []
    session = requests.Session()
    try:
        for ordinal, (layer, indices) in enumerate(by_layer.items(), 1):
            filename = f"layer_{layer}.safetensors"
            url = hf_hub_url(
                repo_id=SOURCE_REPO,
                filename=filename,
                revision=SOURCE_REVISION,
            )
            metadata = get_hf_file_metadata(url, token=False)
            if metadata.commit_hash != SOURCE_REVISION:
                raise ValueError(
                    f"Layer {layer} resolved to another transcoder revision"
                )
            if not isinstance(metadata.etag, str) or len(metadata.etag) != 64:
                raise ValueError(f"Layer {layer} lacks a SHA-256-like ETag")
            header_length, header, header_sha256 = _load_header(
                session, metadata.location
            )
            if any(name not in header for name in TENSOR_NAMES):
                raise ValueError(f"Layer {layer} lacks a required transcoder tensor")
            layer_record: dict[str, Any] = {
                "layer": layer,
                "filename": filename,
                "resolved_commit": metadata.commit_hash,
                "source_file_size": metadata.size,
                "source_file_etag": metadata.etag,
                "safetensors_header_length": header_length,
                "safetensors_header_sha256": header_sha256,
                "feature_indices": indices,
                "tensors": {},
            }
            for name in TENSOR_NAMES:
                tensor, ranges = _fetch_tensor_rows(
                    session=session,
                    url=metadata.location,
                    tensor=header[name],
                    header_length=header_length,
                    indices=indices,
                )
                tensors[f"layer_{layer}.{name}"] = tensor
                layer_record["tensors"][name] = {
                    "source_dtype": header[name]["dtype"],
                    "source_shape": header[name]["shape"],
                    "bundle_shape": list(tensor.shape),
                    "ranges": ranges,
                }
            tensors[f"layer_{layer}.feature_indices"] = torch.tensor(
                indices, dtype=torch.int64
            )
            source_layers.append(layer_record)
            print(
                f"extracted E8b layer {layer} "
                f"({ordinal}/{len(by_layer)}; {len(indices)} rows)",
                flush=True,
            )
    finally:
        session.close()

    if set(tensors) != set(EXPECTED_BUNDLE_KEYS):
        raise ValueError("E8b extracted transcoder tensor closure differs")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    save_deterministic_safetensors(
        tensors,
        temporary,
        metadata={
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "e8a_manifest_payload_sha256": e8a_manifest["payload_sha256"],
            "e8a_plan_payload_sha256": e8a_plan["payload_sha256"],
        },
    )
    temporary.replace(bundle_path)

    payload: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "analysis_semantics_version": BUNDLE_SEMANTICS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": BUNDLE_STATUS,
        "producer": {
            "path": Path(__file__).name,
            "file_sha256": sha256_file(Path(__file__).resolve()),
        },
        "range_helper": {
            "path": RANGE_HELPER_NAME,
            "file_sha256": sha256_file(Path(__file__).with_name(RANGE_HELPER_NAME)),
        },
        "e8a_manifest": {
            "path": e8a_manifest_path.name,
            "file_sha256": sha256_file(e8a_manifest_path),
            "payload_sha256": e8a_manifest["payload_sha256"],
        },
        "e8a_plan": {
            "path": e8a_plan_path.name,
            "file_sha256": sha256_file(e8a_plan_path),
            "payload_sha256": e8a_plan["payload_sha256"],
        },
        "source": {
            "repo_id": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "full_source_files_downloaded": False,
            "http_range_required": True,
        },
        "features": _feature_manifest_rows(),
        "counts": {
            "layers": 2,
            "unique_features": 3,
            "bundle_tensors": 8,
        },
        "layers": source_layers,
        "bundle": {
            "path": bundle_path.name,
            "file_sha256": sha256_file(bundle_path),
            "keys": sorted(EXPECTED_BUNDLE_KEYS),
            "public": False,
        },
    }
    _atomic_json(manifest_path, payload)
    validate_transcoder_bundle_manifest(
        payload,
        e8a_manifest=e8a_manifest,
        e8a_plan=e8a_plan,
        e8a_manifest_path=e8a_manifest_path,
        e8a_plan_path=e8a_plan_path,
        bundle_path=bundle_path,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e8a-manifest", type=Path, required=True)
    parser.add_argument("--e8a-plan", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = extract_bundle(
        e8a_manifest_path=args.e8a_manifest,
        e8a_plan_path=args.e8a_plan,
        bundle_path=args.bundle,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "features": payload["counts"]["unique_features"],
                "bundle_sha256": payload["bundle"]["file_sha256"],
                "payload_sha256": payload["payload_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
