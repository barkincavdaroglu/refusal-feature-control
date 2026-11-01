"""Core graph representation for attribution graphs.

An attribution graph produced by circuit-tracer is a layered DAG whose nodes are

    [ features | error nodes | embedding (token) nodes | target (logit) nodes ]

with the raw adjacency matrix stored as ``A[target, source]`` (rows are targets,
columns are sources) holding *signed* local linear contributions.

Empirically verified structure (Qwen3-4B graphs, 2026-08-18):

* error and embedding rows are all-zero: they are **pure sources**;
* only feature rows and target rows have incoming edges;
* the target has direct incoming edges from embedding nodes, so no feature-only
  vertex cut can disconnect the inputs from the target (see ``mass.py``).

This module holds the container; all analysis lives in the sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

FEATURE, ERROR, EMBED, TARGET = "feature", "error", "embed", "target"
KINDS = (FEATURE, ERROR, EMBED, TARGET)


@dataclass
class AGraph:
    """An attribution graph with a node table and a sparse signed adjacency matrix.

    Attributes:
        name: short identifier (e.g. ``LC-H01-D0``). Never a prompt string.
        A: signed adjacency, CSR, ``A[target, source]`` (circuit-tracer orientation).
        kind: per-node kind, one of :data:`KINDS`.
        layer: per-node layer (-1 for embed/target).
        pos: per-node token position (-1 for target).
        feature_id: transcoder feature id (-1 for non-features).
        activation: feature activation value (nan for non-features).
        target_weights: weights over target nodes (``logit_probabilities``).
        meta: provenance -- file hash, pruning config, weight semantics, etc.
    """

    name: str
    A: sp.csr_matrix
    kind: np.ndarray
    layer: np.ndarray
    pos: np.ndarray
    feature_id: np.ndarray
    activation: np.ndarray
    target_weights: np.ndarray
    n_layers: int
    n_pos: int
    meta: dict[str, Any] = field(default_factory=dict)
    sink_index: int | None = None  # set when targets were collapsed into a super-sink

    # ---- basic properties -------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.A.shape[0]

    def mask(self, kind: str) -> np.ndarray:
        return self.kind == kind

    def idx(self, kind: str) -> np.ndarray:
        return np.flatnonzero(self.kind == kind)

    @property
    def sink(self) -> int:
        """Index of the single target node, or of the virtual super-sink."""
        if self.sink_index is not None:
            return int(self.sink_index)
        targets = self.idx(TARGET)
        if len(targets) != 1:
            raise ValueError(
                f"{self.name}: {len(targets)} target nodes and no super-sink; "
                f"call super_sink() first"
            )
        return int(targets[0])

    def counts(self) -> dict[str, int]:
        return {k: int((self.kind == k).sum()) for k in KINDS}

    def describe(self, i: int) -> str:
        """Human-readable node label. Never includes prompt text."""
        k = self.kind[i]
        if k == FEATURE:
            return f"L{self.layer[i]}:F{self.feature_id[i]}@p{self.pos[i]}"
        if k == ERROR:
            return f"ERR:L{self.layer[i]}@p{self.pos[i]}"
        if k == EMBED:
            return f"EMB@p{self.pos[i]}"
        return "TARGET"

    # ---- derived matrices -------------------------------------------------

    def normalized(self) -> sp.csr_matrix:
        """|A| with each row (incoming edges of a target) normalised to sum 1.

        Matches ``circuit_tracer.graph.normalize_matrix`` so that masses computed
        here are commensurable with circuit-tracer's own influence scores.
        """
        M = abs(self.A).astype(np.float64).tocsr()
        rs = np.asarray(M.sum(axis=1)).ravel()
        inv = np.zeros_like(rs)
        nz = rs > 1e-10
        inv[nz] = 1.0 / rs[nz]
        return sp.diags(inv) @ M

    def super_sink(self) -> "AGraph":
        """Collapse multiple target nodes into one virtual sink.

        Edges into the super-sink carry the target weights, so a graph with
        several logit targets is analysed exactly like a single-target graph.
        """
        t = self.idx(TARGET)
        if len(t) == 1:
            return self
        n = self.n_nodes
        A = self.A.tolil()
        A.resize((n + 1, n + 1))
        w = np.asarray(self.target_weights, dtype=np.float64)
        for j, ti in enumerate(t):
            A[n, ti] = w[j] if j < len(w) else 1.0
        return AGraph(
            name=self.name,
            A=A.tocsr(),
            kind=np.append(self.kind, TARGET),
            layer=np.append(self.layer, -1),
            pos=np.append(self.pos, -1),
            feature_id=np.append(self.feature_id, -1),
            activation=np.append(self.activation, np.nan),
            target_weights=np.array([1.0]),
            n_layers=self.n_layers,
            n_pos=self.n_pos,
            meta={**self.meta, "super_sink": True, "collapsed_targets": len(t)},
            sink_index=n,
        )

    def sink_seed(self) -> np.ndarray:
        """Unit mass placed on the sink (start of the backward walk)."""
        v = np.zeros(self.n_nodes, dtype=np.float64)
        v[self.sink] = 1.0
        return v

    # ---- pruning ----------------------------------------------------------

    def prune_edges(self, edge_threshold: float = 0.98) -> "AGraph":
        """Keep the smallest edge set carrying ``edge_threshold`` of edge influence.

        Follows circuit-tracer's convention: edge score = normalised weight times
        the influence of the edge's target node. Returns a new AGraph with the
        same node set (isolated nodes are retained so that node indices stay
        comparable across pruning configurations).
        """
        if not 0.0 <= edge_threshold <= 1.0:
            raise ValueError(f"edge_threshold must lie in [0, 1], got {edge_threshold}")

        from .mass import backward_mass

        seed = self.sink_seed()
        An = self.normalized()
        node_w = backward_mass(An, seed) + seed  # the sink itself carries weight 1

        coo = self.A.tocoo()
        if coo.nnz == 0:
            return self
        row_sums = np.asarray(abs(self.A).sum(axis=1)).ravel()
        denom = np.where(row_sums > 1e-10, row_sums, 1.0)
        scores = (np.abs(coo.data) / denom[coo.row]) * node_w[coo.row]

        order = np.argsort(-scores)
        csum = np.cumsum(scores[order])
        total = csum[-1]
        if total <= 0:
            return self
        keep_n = min(int(np.searchsorted(csum, edge_threshold * total)) + 1, len(order))
        sel = order[:keep_n]
        A2 = sp.csr_matrix(
            (coo.data[sel], (coo.row[sel], coo.col[sel])), shape=self.A.shape
        )
        out = self.copy_with(A2)
        out.meta = {
            **self.meta,
            "edge_threshold": edge_threshold,
            "edges_kept": int(keep_n),
            "edges_before": int(coo.nnz),
        }
        return out

    def active_mask(self) -> np.ndarray:
        """Nodes that still carry at least one edge (plus the target)."""
        deg = np.asarray(abs(self.A).sum(axis=0)).ravel() + np.asarray(
            abs(self.A).sum(axis=1)
        ).ravel()
        return (deg > 0) | self.mask(TARGET)

    def copy_with(self, A: sp.csr_matrix) -> "AGraph":
        return AGraph(
            name=self.name,
            A=A.tocsr(),
            kind=self.kind.copy(),
            layer=self.layer.copy(),
            pos=self.pos.copy(),
            feature_id=self.feature_id.copy(),
            activation=self.activation.copy(),
            target_weights=self.target_weights.copy(),
            n_layers=self.n_layers,
            n_pos=self.n_pos,
            meta=dict(self.meta),
            sink_index=self.sink_index,
        )

    def forward_edges(self) -> sp.csr_matrix:
        """Edges in source -> target orientation (transpose of ``A``)."""
        return self.A.T.tocsr()
