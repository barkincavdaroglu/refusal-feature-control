"""Outcome-blind, candidate-level redundancy measurements.

The older :mod:`agtk.redundancy` module summarizes a whole attribution graph
rooted at its logit target.  Calibration studies need a slightly different
object: a structural record for one candidate feature, sometimes with an
internal feature (rather than a logit) as the receiver.  This module supplies
that record without accepting intervention outcomes or predictor scores.

All path-mass quantities use the toolkit's unsigned convention: ``abs(A)``
row-normalized over a node's incoming edges.  Signed local transport is a
different estimand and deliberately does not enter these indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow

from .core import EMBED, ERROR, FEATURE, AGraph
from .mass import DEFAULT_MAX_ITER, DEFAULT_TOL, backward_mass


def _node_index(g: AGraph, value: int, name: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer node index") from exc
    if value < 0 or value >= g.n_nodes:
        raise IndexError(f"{name}={value} is outside [0, {g.n_nodes})")
    return value


def _mask(g: AGraph, value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        return np.ones(g.n_nodes, dtype=bool)
    out = np.asarray(value, dtype=bool)
    if out.shape != (g.n_nodes,):
        raise ValueError(f"{name} must have shape ({g.n_nodes},), got {out.shape}")
    return out.copy()


def _threshold(value: float) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"edge_threshold must be finite and in [0, 1], got {value}")
    return value


@dataclass(frozen=True)
class CandidateInterception:
    """Mass of receiver-rooted walks that cross one candidate feature."""

    absolute: float
    relative: float
    interceptable_denominator: float
    reachable: bool

    def as_dict(self) -> dict:
        return {
            "absolute": float(self.absolute),
            "relative": float(self.relative),
            "interceptable_denominator": float(self.interceptable_denominator),
            "reachable": bool(self.reachable),
        }


@dataclass(frozen=True)
class CandidateExcludedCurve:
    """Greedy ``C_k`` curve with the focal candidate unavailable for selection.

    ``relative_to_all`` uses the mass interceptable by *any* feature, including
    the excluded candidate, as its denominator.  ``relative_to_available`` uses
    only mass interceptable by the remaining candidate set.  Reporting both
    prevents an unreachable alternative set from looking artificially strong.
    """

    selected: tuple[int, ...]
    requested_k: int
    cumulative_absolute: tuple[float, ...]
    marginal_absolute: tuple[float, ...]
    cumulative_relative_to_all: tuple[float, ...]
    marginal_relative_to_all: tuple[float, ...]
    cumulative_relative_to_available: tuple[float, ...]
    marginal_relative_to_available: tuple[float, ...]
    focal_marginal_absolute_after_k: tuple[float, ...]
    focal_marginal_relative_to_all_after_k: tuple[float, ...]
    focal_marginal_relative_to_available_after_k: tuple[float, ...]
    all_interceptable_denominator: float
    available_interceptable_denominator: float

    def as_dict(self, g: AGraph | None = None) -> dict:
        out = {
            "selected": [int(v) for v in self.selected],
            "requested_k": int(self.requested_k),
            "cumulative_absolute": [float(v) for v in self.cumulative_absolute],
            "marginal_absolute": [float(v) for v in self.marginal_absolute],
            "cumulative_relative_to_all": [
                float(v) for v in self.cumulative_relative_to_all
            ],
            "marginal_relative_to_all": [
                float(v) for v in self.marginal_relative_to_all
            ],
            "cumulative_relative_to_available": [
                float(v) for v in self.cumulative_relative_to_available
            ],
            "marginal_relative_to_available": [
                float(v) for v in self.marginal_relative_to_available
            ],
            # Element j is the focal candidate's marginal interception after
            # the deterministic candidate-excluded greedy competitor set S_j
            # has been selected. The curve includes j=0 and therefore has
            # requested_k + 1 entries, even when fewer than k positive-gain
            # competitors exist (S_j then remains unchanged).
            "focal_marginal_absolute_after_k": [
                float(v) for v in self.focal_marginal_absolute_after_k
            ],
            "focal_marginal_relative_to_all_after_k": [
                float(v) for v in self.focal_marginal_relative_to_all_after_k
            ],
            "focal_marginal_relative_to_available_after_k": [
                float(v) for v in self.focal_marginal_relative_to_available_after_k
            ],
            "all_interceptable_denominator": float(
                self.all_interceptable_denominator
            ),
            "available_interceptable_denominator": float(
                self.available_interceptable_denominator
            ),
        }
        if g is not None:
            out["selected_labels"] = [g.describe(v) for v in self.selected]
        return out


@dataclass(frozen=True)
class ErrorOriginShare:
    """Where receiver-rooted input walks terminate.

    ``error_direct_one_edge`` is the probability of the literal
    ``error -> receiver`` edge family.  ``error_mediated`` is the remaining
    error-origin mass, including routes through one or more upstream features.
    The denominator excludes pruning-induced dangling-feature endpoints and is
    therefore recorded explicitly.
    """

    error_absolute: float
    embed_absolute: float
    source_denominator: float
    error_share: float
    error_direct_one_edge: float
    error_mediated: float
    dangling_absolute: float

    def as_dict(self) -> dict:
        return {key: float(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class CandidateRedundancyReport:
    """Unsigned structural indices for one candidate and one receiver."""

    candidate: int
    receiver: int
    edge_threshold: float
    edges_before: int
    edges_kept: int
    candidate_interception: CandidateInterception
    candidate_to_receiver_vertex_disjoint_routes: int
    candidate_excluded_curve: CandidateExcludedCurve
    candidate_input_error_origin: ErrorOriginShare

    def as_dict(self, g: AGraph | None = None) -> dict:
        out = {
            "estimand_family": "unsigned_absolute_flow",
            "candidate": int(self.candidate),
            "receiver": int(self.receiver),
            "edge_threshold": float(self.edge_threshold),
            "edges_before": int(self.edges_before),
            "edges_kept": int(self.edges_kept),
            "candidate_interception": self.candidate_interception.as_dict(),
            "candidate_to_receiver_vertex_disjoint_routes": int(
                self.candidate_to_receiver_vertex_disjoint_routes
            ),
            "candidate_excluded_curve": self.candidate_excluded_curve.as_dict(g),
            "candidate_input_error_origin": self.candidate_input_error_origin.as_dict(),
        }
        if g is not None:
            out["candidate_label"] = g.describe(self.candidate)
            out["receiver_label"] = g.describe(self.receiver)
        return out


class ReceiverRootedMass:
    """Unsigned backward-walk calculations rooted at an arbitrary graph node."""

    def __init__(
        self,
        g: AGraph,
        receiver: int,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
        normalized: sp.csr_matrix | None = None,
    ) -> None:
        self.g = g
        self.receiver = _node_index(g, receiver, "receiver")
        if normalized is None:
            self.An = g.normalized()
        else:
            self.An = normalized.tocsr()
            if self.An.shape != g.A.shape:
                raise ValueError(
                    f"normalized matrix has shape {self.An.shape}, expected {g.A.shape}"
                )
        self.AnT = self.An.T.tocsr()
        self.seed = np.zeros(g.n_nodes, dtype=np.float64)
        self.seed[self.receiver] = 1.0
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._base: np.ndarray | None = None

    def total(self, absorbing: np.ndarray | None = None) -> np.ndarray:
        if absorbing is not None:
            absorbing = np.asarray(absorbing, dtype=bool)
            if absorbing.shape != (self.g.n_nodes,):
                raise ValueError(
                    f"absorbing must have shape ({self.g.n_nodes},), "
                    f"got {absorbing.shape}"
                )
            return backward_mass(
                self.An,
                self.seed,
                absorbing=absorbing,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        if self._base is None:
            self._base = backward_mass(
                self.An,
                self.seed,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        return self._base

    def hit_probability(self, nodes: np.ndarray) -> np.ndarray:
        """Probability of a strict future hit, for every possible walk root."""
        nodes = np.asarray(nodes, dtype=bool)
        if nodes.shape != (self.g.n_nodes,):
            raise ValueError(
                f"nodes must have shape ({self.g.n_nodes},), got {nodes.shape}"
            )
        gamma = np.zeros(self.g.n_nodes, dtype=np.float64)
        for _ in range(self.max_iter):
            nxt = self.An @ np.where(nodes, 1.0, gamma)
            if np.max(np.abs(nxt - gamma)) <= self.tol:
                return nxt
            gamma = nxt
        raise RuntimeError("hit-probability iteration did not converge")

    def intercepted(self, nodes: Sequence[int]) -> float:
        idx = np.asarray(list(nodes), dtype=int)
        if idx.size == 0:
            return 0.0
        if np.any(idx < 0) or np.any(idx >= self.g.n_nodes):
            raise IndexError("interception set contains an invalid node index")
        absorbing = np.zeros(self.g.n_nodes, dtype=bool)
        absorbing[idx] = True
        return float(self.total(absorbing)[idx].sum())

    def marginal_gains(self, absorbing: np.ndarray) -> np.ndarray:
        absorbing = np.asarray(absorbing, dtype=bool)
        beta = self.total(absorbing)
        gamma = self.hit_probability(absorbing)
        return beta * (1.0 - gamma)

    def greedy(
        self, k: int, candidates: np.ndarray
    ) -> tuple[list[int], list[float]]:
        """Deterministic greedy coverage; node index breaks exact score ties."""
        if int(k) != k or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k}")
        candidates = np.asarray(candidates, dtype=bool)
        if candidates.shape != (self.g.n_nodes,):
            raise ValueError(
                f"candidates must have shape ({self.g.n_nodes},), "
                f"got {candidates.shape}"
            )
        absorbing = np.zeros(self.g.n_nodes, dtype=bool)
        chosen: list[int] = []
        curve: list[float] = []
        covered = 0.0
        for _ in range(int(k)):
            gains = self.marginal_gains(absorbing)
            gains = np.where(candidates & ~absorbing, gains, -np.inf)
            best = float(np.max(gains))
            if not np.isfinite(best) or best <= 1e-15:
                break
            # np.flatnonzero returns ascending indices, making ties stable even
            # when a sparse input's edge order changes.
            tied = np.flatnonzero(np.isclose(gains, best, rtol=0.0, atol=1e-15))
            node = int(tied[0])
            covered += float(gains[node])
            absorbing[node] = True
            chosen.append(node)
            curve.append(covered)
        return chosen, curve

    def candidate_excluded_curve(
        self,
        focal: int,
        k: int,
        candidates: np.ndarray,
    ) -> CandidateExcludedCurve:
        """Greedy competitor coverage and focal gain after every prefix.

        The expensive marginal-gain vector used to select each competitor also
        contains the focal candidate's gain. Recording both in the same loop
        makes C1/C2/C5/C10 sensitivities one k=10 pass rather than four runs.
        """
        focal = _node_index(self.g, focal, "focal")
        if int(k) != k or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k}")
        all_candidates = np.asarray(candidates, dtype=bool)
        if all_candidates.shape != (self.g.n_nodes,):
            raise ValueError(
                f"candidates must have shape ({self.g.n_nodes},), "
                f"got {all_candidates.shape}"
            )
        if not all_candidates[focal]:
            raise ValueError("focal must be present in candidates")

        available = all_candidates.copy()
        available[focal] = False
        all_denom = float(self.hit_probability(all_candidates)[self.receiver])
        available_denom = float(self.hit_probability(available)[self.receiver])
        absorbing = np.zeros(self.g.n_nodes, dtype=bool)
        chosen: list[int] = []
        cumulative: list[float] = []
        focal_after_k: list[float] = []
        covered = 0.0

        for step in range(int(k) + 1):
            gains = self.marginal_gains(absorbing)
            focal_after_k.append(max(float(gains[focal]), 0.0))
            if step == int(k):
                break
            competitor_gains = np.where(available & ~absorbing, gains, -np.inf)
            best = float(np.max(competitor_gains))
            if not np.isfinite(best) or best <= 1e-15:
                # For every larger j, S_j is the same exhausted set. Preserve
                # exact C_k indexing instead of returning a short curve.
                focal_after_k.extend([focal_after_k[-1]] * (int(k) - step))
                break
            tied = np.flatnonzero(
                np.isclose(competitor_gains, best, rtol=0.0, atol=1e-15)
            )
            node = int(tied[0])
            absorbing[node] = True
            covered += max(float(gains[node]), 0.0)
            chosen.append(node)
            cumulative.append(covered)

        marginal = np.diff(np.r_[0.0, np.asarray(cumulative, dtype=np.float64)])

        def norm(values: Iterable[float], denom: float) -> tuple[float, ...]:
            if denom <= 1e-15:
                return tuple(0.0 for _ in values)
            return tuple(
                min(max(float(value) / denom, 0.0), 1.0) for value in values
            )

        cumulative_t = tuple(float(value) for value in cumulative)
        marginal_t = tuple(float(value) for value in marginal)
        focal_t = tuple(float(value) for value in focal_after_k)
        return CandidateExcludedCurve(
            selected=tuple(chosen),
            requested_k=int(k),
            cumulative_absolute=cumulative_t,
            marginal_absolute=marginal_t,
            cumulative_relative_to_all=norm(cumulative_t, all_denom),
            marginal_relative_to_all=norm(marginal_t, all_denom),
            cumulative_relative_to_available=norm(cumulative_t, available_denom),
            marginal_relative_to_available=norm(marginal_t, available_denom),
            focal_marginal_absolute_after_k=focal_t,
            focal_marginal_relative_to_all_after_k=norm(focal_t, all_denom),
            focal_marginal_relative_to_available_after_k=norm(
                focal_t, available_denom
            ),
            all_interceptable_denominator=all_denom,
            available_interceptable_denominator=available_denom,
        )

    def error_origin_share(self) -> ErrorOriginShare:
        g, mass = self.g, self.total()
        err, emb = g.mask(ERROR), g.mask(EMBED)
        error_absolute = float(mass[err].sum())
        embed_absolute = float(mass[emb].sum())
        denominator = error_absolute + embed_absolute
        direct = float(self.An[self.receiver, err].sum())
        # A tiny negative can arise only from subtraction roundoff.
        mediated = max(error_absolute - direct, 0.0)
        terminal = np.asarray(self.An.sum(axis=1)).ravel() <= 1e-12
        dangling = terminal & g.mask(FEATURE)
        if g.kind[self.receiver] == FEATURE:
            dangling = dangling.copy()
            dangling[self.receiver] = False
        return ErrorOriginShare(
            error_absolute=error_absolute,
            embed_absolute=embed_absolute,
            source_denominator=denominator,
            error_share=(error_absolute / denominator if denominator > 1e-15 else 0.0),
            error_direct_one_edge=direct,
            error_mediated=mediated,
            dangling_absolute=float(mass[dangling].sum()),
        )


def prune_for_receiver(g: AGraph, receiver: int, edge_threshold: float) -> AGraph:
    """Independently prune one graph using influence from ``receiver``.

    Every call starts from ``g``.  Passing a sequence of thresholds to
    :func:`candidate_redundancy_across_thresholds` therefore does not perform
    progressive pruning, and invocation order cannot change the retained set.
    """
    receiver = _node_index(g, receiver, "receiver")
    edge_threshold = _threshold(edge_threshold)
    coo = g.A.tocoo()
    if coo.nnz == 0 or edge_threshold == 0.0:
        out = g.copy_with(sp.csr_matrix(g.A.shape, dtype=g.A.dtype))
        out.meta = {
            **g.meta,
            "edge_threshold": edge_threshold,
            "pruning_receiver": receiver,
            "edges_before": int(coo.nnz),
            "edges_kept": 0,
            "pruning_rule": "receiver_rooted_cumulative_unsigned_edge_influence",
        }
        return out

    eng = ReceiverRootedMass(g, receiver)
    node_weight = eng.total() + eng.seed
    row_sums = np.asarray(abs(g.A).sum(axis=1)).ravel()
    denom = np.where(row_sums > 1e-12, row_sums, 1.0)
    score = (np.abs(coo.data) / denom[coo.row]) * node_weight[coo.row]
    positive = score > 0.0
    if not positive.any():
        out = g.copy_with(sp.csr_matrix(g.A.shape, dtype=g.A.dtype))
        out.meta = {
            **g.meta,
            "edge_threshold": edge_threshold,
            "pruning_receiver": receiver,
            "edges_before": int(coo.nnz),
            "edges_kept": 0,
            "pruning_rule": "receiver_rooted_cumulative_unsigned_edge_influence",
        }
        return out

    pos = np.flatnonzero(positive)
    # Highest influence first, then target row and source column.  Sparse
    # storage order is not part of the estimand and may differ across loaders.
    order = pos[np.lexsort((coo.col[pos], coo.row[pos], -score[pos]))]
    cumulative = np.cumsum(score[order])
    cutoff = edge_threshold * float(cumulative[-1])
    keep_n = int(np.searchsorted(cumulative, cutoff, side="left")) + 1
    chosen = order[:keep_n]
    A = sp.csr_matrix(
        (coo.data[chosen], (coo.row[chosen], coo.col[chosen])), shape=g.A.shape
    )
    out = g.copy_with(A)
    out.meta = {
        **g.meta,
        "edge_threshold": edge_threshold,
        "pruning_receiver": receiver,
        "edges_before": int(coo.nnz),
        "edges_kept": int(A.nnz),
        "pruning_rule": "receiver_rooted_cumulative_unsigned_edge_influence",
    }
    return out


def prune_for_receiver_across_thresholds(
    g: AGraph,
    receiver: int,
    edge_thresholds: Sequence[float] = (0.80, 0.90, 0.98),
) -> dict[str, AGraph]:
    """Independently prune at several thresholds from one receiver ordering.

    The receiver-rooted mass, per-edge influence score, and deterministic edge
    order do not depend on the requested cumulative-mass threshold.  Computing
    them once and taking an independent prefix for each threshold is therefore
    exactly equivalent to repeated :func:`prune_for_receiver` calls, while
    avoiding repeated full-graph sparse walks and ``O(E log E)`` sorts.

    Returned keys use the same stable formatting as
    :func:`candidate_redundancy_across_thresholds`.  Each returned graph is a
    fresh copy constructed from the authenticated input graph, never from a
    graph produced for another threshold.
    """
    receiver = _node_index(g, receiver, "receiver")
    thresholds = [_threshold(value) for value in edge_thresholds]
    if not thresholds:
        raise ValueError("edge_thresholds must be non-empty")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("edge_thresholds must be unique")

    coo = g.A.tocoo()
    edges_before = int(coo.nnz)

    def empty(threshold: float) -> AGraph:
        out = g.copy_with(sp.csr_matrix(g.A.shape, dtype=g.A.dtype))
        out.meta = {
            **g.meta,
            "edge_threshold": threshold,
            "pruning_receiver": receiver,
            "edges_before": edges_before,
            "edges_kept": 0,
            "pruning_rule": "receiver_rooted_cumulative_unsigned_edge_influence",
        }
        return out

    if coo.nnz == 0 or all(value == 0.0 for value in thresholds):
        return {format(value, ".12g"): empty(value) for value in thresholds}

    eng = ReceiverRootedMass(g, receiver)
    node_weight = eng.total() + eng.seed
    row_sums = np.asarray(abs(g.A).sum(axis=1)).ravel()
    denom = np.where(row_sums > 1e-12, row_sums, 1.0)
    score = (np.abs(coo.data) / denom[coo.row]) * node_weight[coo.row]
    positive = score > 0.0
    if not positive.any():
        return {format(value, ".12g"): empty(value) for value in thresholds}

    pos = np.flatnonzero(positive)
    # Match prune_for_receiver byte-for-byte: descending score, then target
    # row and source column.  Threshold invocation order cannot affect this
    # single frozen ordering.
    order = pos[np.lexsort((coo.col[pos], coo.row[pos], -score[pos]))]
    cumulative = np.cumsum(score[order])
    total = float(cumulative[-1])
    result: dict[str, AGraph] = {}
    for threshold in thresholds:
        key = format(threshold, ".12g")
        if threshold == 0.0:
            result[key] = empty(threshold)
            continue
        cutoff = threshold * total
        keep_n = int(np.searchsorted(cumulative, cutoff, side="left")) + 1
        chosen = order[:keep_n]
        A = sp.csr_matrix(
            (coo.data[chosen], (coo.row[chosen], coo.col[chosen])),
            shape=g.A.shape,
        )
        out = g.copy_with(A)
        out.meta = {
            **g.meta,
            "edge_threshold": threshold,
            "pruning_receiver": receiver,
            "edges_before": edges_before,
            "edges_kept": int(A.nnz),
            "pruning_rule": "receiver_rooted_cumulative_unsigned_edge_influence",
        }
        result[key] = out
    return result


def candidate_interception_fraction(
    g: AGraph, candidate: int, receiver: int
) -> CandidateInterception:
    """Unsigned walk share intercepted by ``candidate`` below ``receiver``."""
    candidate = _node_index(g, candidate, "candidate")
    receiver = _node_index(g, receiver, "receiver")
    if g.kind[candidate] != FEATURE:
        raise ValueError("candidate must be a feature node")
    if candidate == receiver:
        raise ValueError("candidate and receiver must be distinct")
    eng = ReceiverRootedMass(g, receiver)
    candidates = g.mask(FEATURE).copy()
    candidates[receiver] = False
    denominator = float(eng.hit_probability(candidates)[receiver])
    absolute = eng.intercepted([candidate])
    relative = absolute / denominator if denominator > 1e-15 else 0.0
    return CandidateInterception(
        absolute=absolute,
        relative=min(max(relative, 0.0), 1.0),
        interceptable_denominator=denominator,
        reachable=absolute > 1e-15,
    )


def candidate_excluded_c_k(
    g: AGraph,
    candidate: int,
    receiver: int,
    k: int = 5,
    candidates: np.ndarray | None = None,
) -> CandidateExcludedCurve:
    """Greedy competitor curve and focal marginal gain after each ``S_k``.

    ``S_k`` is chosen greedily from feature candidates with the focal candidate
    excluded. Besides reporting competitor coverage, the result reports the
    focal candidate's marginal gain after every prefix ``S_0 .. S_k``.
    """
    candidate = _node_index(g, candidate, "candidate")
    receiver = _node_index(g, receiver, "receiver")
    if int(k) != k or k < 0:
        raise ValueError(f"k must be a non-negative integer, got {k}")
    if candidates is None:
        all_candidates = g.mask(FEATURE).copy()
    else:
        all_candidates = _mask(g, candidates, "candidates") & g.mask(FEATURE)
    all_candidates[receiver] = False
    eng = ReceiverRootedMass(g, receiver)
    return eng.candidate_excluded_curve(candidate, int(k), all_candidates)


def internal_vertex_disjoint_route_count(
    g: AGraph,
    source: int | Sequence[int],
    receiver: int,
    keep: np.ndarray | None = None,
) -> int:
    """Count internally vertex-disjoint directed routes into ``receiver``.

    With one source, source and receiver are uncapacitated endpoints, so a
    direct edge and any internally disjoint indirect routes are all counted.
    With multiple sources, a virtual super-source is added and each listed
    source may contribute at most one route; this is the standard source-set
    extension and prevents duplicated origins from inflating the count.
    """
    receiver = _node_index(g, receiver, "receiver")
    if isinstance(source, (int, np.integer)):
        sources = [_node_index(g, int(source), "source")]
        source_is_set = False
    else:
        sources = [_node_index(g, value, "source") for value in source]
        source_is_set = True
        if not sources:
            return 0
        if len(set(sources)) != len(sources):
            raise ValueError("source indices must be unique")
    if receiver in sources:
        raise ValueError("source and receiver must be distinct")
    keep_mask = _mask(g, keep, "keep")
    if not keep_mask[receiver] or not any(keep_mask[s] for s in sources):
        return 0

    live_sources = [s for s in sources if keep_mask[s]]
    # Work only on the source/receiver corridor.  A full NetworkX conversion of
    # a production attribution graph can multiply the memory footprint of its
    # tens of millions of edges; sparse Boolean reachability keeps this usable
    # on those graphs.
    fwd = (abs(g.forward_edges()) > 0).astype(np.float64).tocsr()

    def closure(matrix: sp.csr_matrix, seed: np.ndarray) -> np.ndarray:
        reached = seed.copy()
        frontier = seed.copy()
        while frontier.any():
            nxt = ((matrix @ frontier.astype(np.float64)) > 0) & keep_mask & ~reached
            if not nxt.any():
                break
            reached |= nxt
            frontier = nxt
        return reached

    seed_from = np.zeros(g.n_nodes, dtype=bool)
    seed_from[live_sources] = True
    # fwd[source, target], hence fwd.T propagates source -> target and fwd
    # propagates receiver -> predecessor.
    from_source = closure(fwd.T.tocsr(), seed_from)
    seed_to = np.zeros(g.n_nodes, dtype=bool)
    seed_to[receiver] = True
    to_receiver = closure(fwd, seed_to)
    corridor = from_source & to_receiver & keep_mask
    if not corridor[receiver]:
        return 0

    coo = fwd.tocoo()
    edge_keep = corridor[coo.row] & corridor[coo.col]
    edge_source = coo.row[edge_keep]
    edge_target = coo.col[edge_keep]

    # For a single endpoint source, a literal source->receiver edge contains no
    # internal vertex.  Count that route once and omit the infinite-capacity
    # edge from max flow.  Source-set mode instead splits each source with unit
    # capacity, so its direct edge can remain in the network.
    direct = 0
    if not source_is_set:
        direct = int(
            np.any((edge_source == live_sources[0]) & (edge_target == receiver))
        )

    split_mask = corridor.copy()
    split_mask[receiver] = False
    if not source_is_set:
        split_mask[live_sources[0]] = False
    split_nodes = np.flatnonzero(split_mask)
    if split_nodes.size == 0:
        return direct
    slot = np.full(g.n_nodes, -1, dtype=np.int64)
    slot[split_nodes] = np.arange(len(split_nodes), dtype=np.int64)
    n_split = len(split_nodes)
    S, T = 2 * n_split, 2 * n_split + 1
    INF = int(min(n_split + len(live_sources) + 2, np.iinfo(np.int32).max // 8))

    rows: list[np.ndarray] = [np.arange(n_split, dtype=np.int64)]
    cols: list[np.ndarray] = [n_split + np.arange(n_split, dtype=np.int64)]
    data: list[np.ndarray] = [np.ones(n_split, dtype=np.int64)]

    if source_is_set:
        source_slots = slot[np.asarray(live_sources, dtype=int)]
        rows.append(np.full(len(source_slots), S, dtype=np.int64))
        cols.append(source_slots)
        data.append(np.full(len(source_slots), INF, dtype=np.int64))

    src_split = slot[edge_source] >= 0
    dst_split = slot[edge_target] >= 0
    into_receiver = edge_target == receiver
    if source_is_set:
        # All route origins are split nodes in this mode.
        internal = src_split & dst_split
        terminal = src_split & into_receiver
        rows.extend([n_split + slot[edge_source[internal]], n_split + slot[edge_source[terminal]]])
        cols.extend([slot[edge_target[internal]], np.full(int(terminal.sum()), T, dtype=np.int64)])
        data.extend([
            np.full(int(internal.sum()), INF, dtype=np.int64),
            np.full(int(terminal.sum()), INF, dtype=np.int64),
        ])
    else:
        source_node = live_sources[0]
        from_endpoint = edge_source == source_node
        # The endpoint's direct edge was counted separately above.
        start = from_endpoint & dst_split
        internal = src_split & dst_split
        terminal = src_split & into_receiver
        rows.extend([
            np.full(int(start.sum()), S, dtype=np.int64),
            n_split + slot[edge_source[internal]],
            n_split + slot[edge_source[terminal]],
        ])
        cols.extend([
            slot[edge_target[start]],
            slot[edge_target[internal]],
            np.full(int(terminal.sum()), T, dtype=np.int64),
        ])
        data.extend([
            np.full(int(start.sum()), INF, dtype=np.int64),
            np.full(int(internal.sum()), INF, dtype=np.int64),
            np.full(int(terminal.sum()), INF, dtype=np.int64),
        ])

    capacity = sp.csr_matrix(
        (np.concatenate(data).astype(np.int32), (np.concatenate(rows), np.concatenate(cols))),
        shape=(T + 1, T + 1),
    )
    capacity.sum_duplicates()
    capacity.data = np.minimum(capacity.data, np.int32(INF))
    if capacity[S].nnz == 0 or capacity[:, T].nnz == 0:
        return direct
    return direct + int(maximum_flow(capacity, S, T).flow_value)


def analyze_candidate_redundancy(
    g: AGraph,
    candidate: int,
    receiver: int | None = None,
    *,
    k: int = 5,
    edge_threshold: float = 1.0,
) -> CandidateRedundancyReport:
    """Compute the complete outcome-blind index record for one candidate."""
    candidate = _node_index(g, candidate, "candidate")
    if g.kind[candidate] != FEATURE:
        raise ValueError("candidate must be a feature node")
    if receiver is None:
        rooted = g.super_sink()
        receiver = rooted.sink
    else:
        rooted = g
        receiver = _node_index(rooted, receiver, "receiver")
    if candidate == receiver:
        raise ValueError("candidate and receiver must be distinct")
    edge_threshold = _threshold(edge_threshold)
    pruned = prune_for_receiver(rooted, receiver, edge_threshold)
    interception = candidate_interception_fraction(pruned, candidate, receiver)
    active = pruned.active_mask()
    routes = internal_vertex_disjoint_route_count(
        pruned, candidate, receiver, keep=active
    )
    curve = candidate_excluded_c_k(pruned, candidate, receiver, k=k)
    candidate_inputs = ReceiverRootedMass(pruned, candidate).error_origin_share()
    return CandidateRedundancyReport(
        candidate=candidate,
        receiver=receiver,
        edge_threshold=edge_threshold,
        edges_before=int(rooted.A.nnz),
        edges_kept=int(pruned.A.nnz),
        candidate_interception=interception,
        candidate_to_receiver_vertex_disjoint_routes=routes,
        candidate_excluded_curve=curve,
        candidate_input_error_origin=candidate_inputs,
    )


def candidate_redundancy_across_thresholds(
    g: AGraph,
    candidate: int,
    receiver: int | None = None,
    *,
    k: int = 5,
    edge_thresholds: Sequence[float] = (0.80, 0.90, 0.98),
) -> dict[str, CandidateRedundancyReport]:
    """Run independent, order-invariant pruning sensitivities."""
    thresholds = [_threshold(value) for value in edge_thresholds]
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("edge_thresholds must be unique")
    return {
        format(value, ".12g"): analyze_candidate_redundancy(
            g,
            candidate,
            receiver,
            k=k,
            edge_threshold=value,
        )
        for value in thresholds
    }
