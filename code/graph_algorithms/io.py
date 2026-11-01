"""Loading attribution graphs into :class:`~agtk.core.AGraph`.

Supports circuit-tracer ``.pt`` graphs (the format this project produces).
Loading a ``.pt`` requires the ``circuit_tracer`` package to be importable,
because the file pickles ``UnifiedConfig`` and ``LogitTarget`` objects; point
:func:`set_circuit_tracer_path` at a checkout, or run under an environment where
it is installed.

Privacy: the prompt string stored in a graph is *not* copied into
:class:`AGraph`. Only its length and hash are kept, so reports can never leak
prompt text. Graphs are identified by file stem (e.g. ``LC-H01-D0``).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np
import scipy.sparse as sp

from .core import EMBED, ERROR, FEATURE, TARGET, AGraph

_DENSE_ZERO_TOL = 0.0


def set_circuit_tracer_path(path: str | pathlib.Path) -> None:
    """Prepend a circuit-tracer checkout to ``sys.path`` so ``.pt`` files unpickle."""
    p = str(pathlib.Path(path).expanduser().resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def file_sha256(path: str | pathlib.Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


_FEATURE_TYPES = {
    "cross layer transcoder": FEATURE,
    "mlp reconstruction error": ERROR,
    "embedding": EMBED,
    "logit": TARGET,
}


def _json_int(value: Any, context: str) -> int:
    """Parse an integer-valued JSON field without truncating floats or booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value and value.lstrip("-").isdigit():
        return int(value)
    raise ValueError(f"{context} must be an integer")


def _json_float(value: Any, context: str) -> float:
    """Parse a finite JSON number without accepting booleans or numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def load_json(path: str | pathlib.Path, name: str | None = None) -> AGraph:
    """Load a circuit-tracer / Neuronpedia graph JSON.

    This is the public-graph path: the JSON files served by Neuronpedia contain
    the pruned graph with ``nodes`` and ``links``, so no transcoders, model or
    ``circuit_tracer`` install are needed.

    Node ids follow circuit-tracer's convention: ``{layer}_{feature}_{pos}`` for
    features, ``{layer}_-1_{pos}`` for reconstruction errors, ``E_{vocab}_{pos}``
    for embeddings, and a ``logit`` feature type for targets. Links carry a
    signed ``weight``.
    """
    path = pathlib.Path(path)
    with open(path) as f:
        blob = json.load(f)
    if not isinstance(blob, dict):
        raise ValueError(f"{path.name}: JSON root must be an object")
    metadata = blob.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path.name}: metadata must be an object")

    raw_nodes = blob.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError(f"{path.name}: nodes must be a non-empty list")
    if "links" in blob:
        links = blob["links"]
    else:
        links = blob.get("edges", [])
    if not isinstance(links, list):
        raise ValueError(f"{path.name}: links/edges must be a list")

    index: dict[str, int] = {}
    kinds, layers, poss, fids, acts = [], [], [], [], []
    target_probabilities: list[float | None] = []
    target_display_layers: list[int] = []
    for i, nd in enumerate(raw_nodes):
        if not isinstance(nd, dict):
            raise ValueError(f"{path.name}: node {i} must be an object")
        nid = nd.get("node_id")
        if not isinstance(nid, str) or not nid:
            raise ValueError(f"{path.name}: node {i} has no valid node_id")
        if nid in index:
            raise ValueError(f"{path.name}: duplicate node_id {nid!r}")
        index[nid] = i

        feature_type = nd.get("feature_type")
        if not isinstance(feature_type, str) or feature_type not in _FEATURE_TYPES:
            raise ValueError(
                f"{path.name}: node {nid!r} has unknown feature_type "
                f"{feature_type!r}"
            )
        kind = _FEATURE_TYPES[feature_type]
        kinds.append(kind)

        if kind == EMBED:
            layer = -1
        elif kind == TARGET:
            try:
                display_layer = _json_int(
                    nd["layer"], f"{path.name}: target {nid!r} display layer"
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"{path.name}: target {nid!r} has invalid display layer"
                ) from exc
            if display_layer < 1:
                raise ValueError(
                    f"{path.name}: target {nid!r} has invalid display layer"
                )
            target_display_layers.append(display_layer)
            layer = -1
        else:
            try:
                layer = _json_int(nd["layer"], f"{path.name}: node {nid!r} layer")
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"{path.name}: node {nid!r} has invalid layer"
                ) from exc
            if layer < 0:
                raise ValueError(f"{path.name}: node {nid!r} has negative layer")
        layers.append(layer)

        try:
            pos = _json_int(nd["ctx_idx"], f"{path.name}: node {nid!r} ctx_idx")
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path.name}: node {nid!r} has invalid ctx_idx") from exc
        if pos < 0:
            raise ValueError(f"{path.name}: node {nid!r} has negative ctx_idx")
        poss.append(pos)

        fid = -1
        if kind == FEATURE:
            parts = nid.split("_")
            if len(parts) != 3:
                raise ValueError(f"{path.name}: feature node_id {nid!r} is malformed")
            try:
                id_layer = _json_int(parts[0], f"{path.name}: feature node_id layer")
                fid = _json_int(parts[1], f"{path.name}: feature node_id feature")
                id_pos = _json_int(parts[2], f"{path.name}: feature node_id position")
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}: feature node_id {nid!r} is malformed"
                ) from exc
            if fid < 0 or id_layer != layer or id_pos != pos:
                raise ValueError(
                    f"{path.name}: feature node_id {nid!r} contradicts its fields"
                )
        elif kind == ERROR:
            if nid != f"{layer}_-1_{pos}":
                raise ValueError(
                    f"{path.name}: error node_id {nid!r} contradicts its fields"
                )
        elif kind == EMBED:
            parts = nid.split("_")
            try:
                id_vocab = _json_int(
                    parts[1] if len(parts) == 3 else None,
                    f"{path.name}: embedding node_id vocab",
                )
                id_pos = _json_int(
                    parts[2] if len(parts) == 3 else None,
                    f"{path.name}: embedding node_id position",
                )
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}: embedding node_id {nid!r} contradicts its fields"
                ) from exc
            if parts[0] != "E" or id_vocab < 0 or id_pos != pos:
                raise ValueError(
                    f"{path.name}: embedding node_id {nid!r} contradicts its fields"
                )
        else:  # TARGET
            parts = nid.split("_")
            try:
                id_layer = _json_int(
                    parts[0] if len(parts) == 3 else None,
                    f"{path.name}: target node_id layer",
                )
                id_vocab = _json_int(
                    parts[1] if len(parts) == 3 else None,
                    f"{path.name}: target node_id vocab",
                )
                id_pos = _json_int(
                    parts[2] if len(parts) == 3 else None,
                    f"{path.name}: target node_id position",
                )
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}: target node_id {nid!r} contradicts its fields"
                ) from exc
            if (
                id_layer != target_display_layers[-1]
                or id_vocab < 0
                or id_pos != pos
            ):
                raise ValueError(
                    f"{path.name}: target node_id {nid!r} contradicts its fields"
                )
        fids.append(fid)

        activation = nd.get("activation")
        if activation is None:
            acts.append(np.nan)
        else:
            try:
                activation = _json_float(
                    activation, f"{path.name}: node {nid!r} activation"
                )
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}: node {nid!r} has invalid activation"
                ) from exc
            acts.append(activation)

        if kind == TARGET:
            raw_probability = nd.get("token_prob")
            if raw_probability is None:
                target_probabilities.append(None)
                continue
            try:
                probability = _json_float(
                    raw_probability, f"{path.name}: target {nid!r} token_prob"
                )
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}: target {nid!r} has invalid token_prob"
                ) from exc
            if probability < 0:
                raise ValueError(
                    f"{path.name}: target {nid!r} has invalid token_prob"
                )
            target_probabilities.append(probability)

    if not target_probabilities:
        raise ValueError(f"{path.name}: graph contains no target nodes")
    if len(set(target_display_layers)) != 1:
        raise ValueError(f"{path.name}: target display layers disagree")
    n_layers = target_display_layers[0] - 1
    model_layers = [
        layer for layer, kind in zip(layers, kinds) if kind in (FEATURE, ERROR)
    ]
    if any(layer >= n_layers for layer in model_layers):
        raise ValueError(
            f"{path.name}: feature/error layer is outside the declared model depth"
        )
    probability_presence = [value is not None for value in target_probabilities]
    if any(probability_presence) and not all(probability_presence):
        raise ValueError(
            f"{path.name}: token_prob must be present for every target or none"
        )
    target_weights_arr = np.asarray(
        [1.0 if value is None else value for value in target_probabilities],
        dtype=np.float64,
    )
    if float(target_weights_arr.sum()) <= 1e-12:
        # Older exports can omit useful target probabilities. Uniform weights
        # retain the old loader's behaviour without letting a zero row destroy
        # the super-sink walk.
        target_weights_arr[:] = 1.0

    n = len(raw_nodes)
    rows, cols, data = [], [], []
    seen_edges: set[tuple[int, int]] = set()
    for i, lk in enumerate(links):
        if not isinstance(lk, dict):
            raise ValueError(f"{path.name}: link {i} must be an object")
        source, target = lk.get("source"), lk.get("target")
        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or source not in index
            or target not in index
        ):
            raise ValueError(
                f"{path.name}: link {i} references an unknown node "
                f"({source!r} -> {target!r})"
            )
        try:
            weight = _json_float(lk["weight"], f"{path.name}: link {i} weight")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"{path.name}: link {i} has invalid or non-finite weight"
            ) from exc
        if weight == 0.0:
            continue
        s, t = index[source], index[target]
        if s == t:
            raise ValueError(f"{path.name}: link {i} is a self-loop")
        edge = (t, s)
        if edge in seen_edges:
            raise ValueError(
                f"{path.name}: duplicate edge {source!r} -> {target!r}"
            )
        seen_edges.add(edge)
        rows.append(t)  # rows are targets, matching the .pt convention
        cols.append(s)
        data.append(weight)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    kinds_arr = np.array(kinds, dtype=object)
    meta = {
        "source_file": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "format": "neuronpedia-json",
        "scan_name": metadata.get("scan"),
        "slug": metadata.get("slug"),
        "source_schema_version": metadata.get("schema_version"),
        "weights": "abs",
        "normalize": "incoming",
        "note": "JSON graphs are already pruned by the exporter",
    }
    return AGraph(
        name=name or path.stem,
        A=A,
        kind=kinds_arr,
        layer=np.array(layers, dtype=np.int32),
        pos=np.array(poss, dtype=np.int32),
        feature_id=np.array(fids, dtype=np.int64),
        activation=np.array(acts, dtype=np.float64),
        target_weights=target_weights_arr,
        n_layers=n_layers,
        n_pos=int(max(poss) + 1),
        meta=meta,
    )


def load_pt(
    path: str | pathlib.Path,
    name: str | None = None,
    drop_tol: float = _DENSE_ZERO_TOL,
    hash_file: bool = True,
) -> AGraph:
    """Load a circuit-tracer ``Graph`` saved with ``Graph.to_pt``.

    Args:
        path: path to the ``.pt`` file.
        name: identifier; defaults to the file stem.
        drop_tol: absolute weight below which edges are dropped when converting
            the dense adjacency to sparse (0.0 keeps every stored non-zero).
        hash_file: record the file sha256 in ``meta`` (set False to skip the read).
    """
    import torch  # local import: only the .pt path needs torch

    path = pathlib.Path(path)
    d = torch.load(path, weights_only=False, map_location="cpu")
    cfg = d["cfg"]

    A_dense = d["adjacency_matrix"]
    n_pos = int(len(d["input_tokens"]))
    n_layers = int(cfg.n_layers)
    n_targets = int(len(d["logit_targets"]))
    n_err = n_layers * n_pos
    N = int(A_dense.shape[0])
    n_feat = N - (n_err + n_pos + n_targets)
    if n_feat != int(len(d["selected_features"])):
        raise ValueError(
            f"{path.name}: partition mismatch -- implied {n_feat} feature nodes but "
            f"selected_features has {len(d['selected_features'])}"
        )

    A_np = A_dense.detach().cpu().float().numpy().astype(np.float32)
    if drop_tol > 0:
        A_np[np.abs(A_np) < drop_tol] = 0.0
    A = sp.csr_matrix(A_np)
    del A_np, A_dense

    kind = np.empty(N, dtype=object)
    layer = np.full(N, -1, dtype=np.int32)
    pos = np.full(N, -1, dtype=np.int32)
    feature_id = np.full(N, -1, dtype=np.int64)
    activation = np.full(N, np.nan, dtype=np.float64)

    sel = d["selected_features"].detach().cpu().numpy().astype(np.int64)
    act_feats = d["active_features"].detach().cpu().numpy().astype(np.int64)
    act_vals = d["activation_values"].detach().cpu().float().numpy().astype(np.float64)
    kind[:n_feat] = FEATURE
    layer[:n_feat] = act_feats[sel, 0]
    pos[:n_feat] = act_feats[sel, 1]
    feature_id[:n_feat] = act_feats[sel, 2]
    if act_vals.shape[0] == act_feats.shape[0]:
        activation[:n_feat] = act_vals[sel]
    elif act_vals.shape[0] == n_feat:  # older graphs store only selected activations
        activation[:n_feat] = act_vals
    err_slice = slice(n_feat, n_feat + n_err)
    kind[err_slice] = ERROR
    off = np.arange(n_err)
    layer[err_slice] = off // n_pos  # error nodes are layer-major
    pos[err_slice] = off % n_pos
    emb_slice = slice(n_feat + n_err, n_feat + n_err + n_pos)
    kind[emb_slice] = EMBED
    pos[emb_slice] = np.arange(n_pos)
    kind[n_feat + n_err + n_pos :] = TARGET

    targets = d["logit_targets"]
    meta: dict[str, Any] = {
        "source_file": str(path),
        "file_sha256": file_sha256(path) if hash_file else None,
        "scan_name": d.get("scan_name"),
        "vocab_size": int(d.get("vocab_size", -1)),
        "n_active_features": int(act_feats.shape[0]),
        "prompt_len_chars": len(d.get("input_string", "")),
        "prompt_sha256": hashlib.sha256(
            d.get("input_string", "").encode("utf-8")
        ).hexdigest(),
        "target_names": [getattr(t, "token_str", "?") for t in targets],
        "target_vocab_idx": [int(getattr(t, "vocab_idx", -1)) for t in targets],
        "virtual_target": [
            int(getattr(t, "vocab_idx", -1)) >= int(d.get("vocab_size", 1 << 62))
            for t in targets
        ],
        "weights": "abs",
        "normalize": "incoming",
    }

    return AGraph(
        name=name or path.stem,
        A=A,
        kind=kind,
        layer=layer,
        pos=pos,
        feature_id=feature_id,
        activation=activation,
        target_weights=d["logit_probabilities"]
        .detach()
        .cpu()
        .float()
        .numpy()
        .astype(np.float64),
        n_layers=n_layers,
        n_pos=n_pos,
        meta=meta,
    )
