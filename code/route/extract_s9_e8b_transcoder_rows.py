#!/usr/bin/env python3
"""Read exact E8b transcoder rows with authenticated HTTP byte ranges.

This E8b-owned helper intentionally contains only the range primitives needed
by the compact C/M/D bundle builder. It has no dependency on historical E5b
contracts, so the hash-closed E8b execution sources import in a clean room
without changing the byte-pinned E5b extractor.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import requests
import torch


SOURCE_REPO = "mwhanna/qwen3-4b-transcoders"
SOURCE_REVISION = "94d176260ac39ce2f882b8b09aba8c118df29bb3"
TENSOR_NAMES = ("W_dec", "W_enc", "b_enc")
DTYPE_BYTES = {"BF16": 2}


def consecutive_groups(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        raise ValueError("At least one feature index is required")
    ordered = sorted(indices)
    if len(ordered) != len(set(ordered)) or ordered[0] < 0:
        raise ValueError("Feature indices must be unique non-negative integers")
    groups: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append((start, previous))
        start = previous = value
    groups.append((start, previous))
    return groups


def tensor_row_ranges(
    *,
    tensor: dict[str, Any],
    header_length: int,
    indices: list[int],
) -> list[dict[str, int]]:
    dtype = tensor.get("dtype")
    shape = tensor.get("shape")
    offsets = tensor.get("data_offsets")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    if not isinstance(shape, list) or not shape or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in shape
    ):
        raise ValueError("Safetensors shape is invalid")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("Safetensors data offsets are invalid")
    if any(index >= shape[0] for index in indices):
        raise ValueError("Selected feature index exceeds tensor shape")
    row_elements = math.prod(shape[1:]) if len(shape) > 1 else 1
    row_bytes = row_elements * DTYPE_BYTES[dtype]
    data_start = 8 + header_length + int(offsets[0])
    expected_size = shape[0] * row_bytes
    if int(offsets[1]) - int(offsets[0]) != expected_size:
        raise ValueError("Safetensors tensor byte size differs from its shape")
    return [
        {
            "first_index": first,
            "last_index": last,
            "byte_start": data_start + first * row_bytes,
            "byte_end": data_start + (last + 1) * row_bytes - 1,
            "row_bytes": row_bytes,
        }
        for first, last in consecutive_groups(indices)
    ]


def _read_range(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
    *,
    retries: int = 4,
) -> bytes:
    expected = end - start + 1
    if start < 0 or end < start:
        raise ValueError("Invalid HTTP byte range")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                timeout=60,
            )
            if response.status_code != 206:
                response.close()
                raise RuntimeError(
                    f"Server ignored exact range {start}-{end}: "
                    f"status {response.status_code}"
                )
            content_range = response.headers.get("content-range", "")
            if not content_range.startswith(f"bytes {start}-{end}/"):
                response.close()
                raise RuntimeError(
                    f"Server returned the wrong content range: {content_range!r}"
                )
            data = response.raw.read(expected + 1)
            response.close()
            if len(data) != expected:
                raise RuntimeError(
                    f"Range {start}-{end} returned {len(data)} bytes, "
                    f"expected {expected}"
                )
            return data
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch byte range {start}-{end}") from last_error


def _load_header(
    session: requests.Session,
    url: str,
) -> tuple[int, dict[str, Any], str]:
    first = _read_range(session, url, 0, 7)
    header_length = int.from_bytes(first, "little")
    if header_length <= 0 or header_length > 100_000_000:
        raise ValueError(f"Implausible safetensors header length: {header_length}")
    header_bytes = _read_range(session, url, 8, 8 + header_length - 1)
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header root is not an object")
    return header_length, header, hashlib.sha256(header_bytes).hexdigest()


def _decode_bf16(payload: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    expected = math.prod(shape) * 2
    if len(payload) != expected:
        raise ValueError(f"BF16 payload has {len(payload)} bytes, expected {expected}")
    return torch.frombuffer(
        bytearray(payload), dtype=torch.bfloat16
    ).clone().reshape(shape)


def _fetch_tensor_rows(
    *,
    session: requests.Session,
    url: str,
    tensor: dict[str, Any],
    header_length: int,
    indices: list[int],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    ranges = tensor_row_ranges(
        tensor=tensor,
        header_length=header_length,
        indices=indices,
    )
    rows_by_index: dict[int, bytes] = {}
    provenance = []
    row_bytes = ranges[0]["row_bytes"]
    for record in ranges:
        payload = _read_range(
            session,
            url,
            record["byte_start"],
            record["byte_end"],
        )
        provenance.append(
            {
                **record,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        for offset, index in enumerate(
            range(record["first_index"], record["last_index"] + 1)
        ):
            start = offset * row_bytes
            rows_by_index[index] = payload[start : start + row_bytes]
    ordered = b"".join(rows_by_index[index] for index in indices)
    trailing_shape = tuple(int(value) for value in tensor["shape"][1:])
    output_shape = (len(indices), *trailing_shape)
    return _decode_bf16(ordered, output_shape), provenance


__all__ = [
    "SOURCE_REPO",
    "SOURCE_REVISION",
    "TENSOR_NAMES",
    "consecutive_groups",
    "tensor_row_ranges",
    "_fetch_tensor_rows",
    "_load_header",
]
