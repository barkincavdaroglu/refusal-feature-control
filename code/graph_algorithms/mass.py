"""Path-mass computations on attribution graphs.

Semantics (the foundation for everything else in this toolkit)
--------------------------------------------------------------
Take ``|A|`` row-normalised over incoming edges -- exactly circuit-tracer's
``normalize_matrix``. Each target node's incoming edges then form a probability
distribution over its sources, so the quantity circuit-tracer calls "node
influence" is the law of a **backward random walk started at the target**:

    start at the sink; repeatedly step from the current node to one of its
    sources with probability ``An[current, source]``; stop when a node with no
    incoming edges is reached. In an unpruned graph that endpoint is ordinarily
    an error or embedding node. Pruning can also leave a retained feature with
    no retained inputs; such a feature is a separate ``dangling`` endpoint.

Because the graph is a DAG the walk visits any node at most once, so

    ``mass(v) = P(the walk passes through v)``

and the walk terminates with probability 1. This gives influence a genuine
conservation law -- ``embedding + error + dangling mass == 1`` -- which is what
makes cuts, interception curves and dominators well defined rather than
heuristic. Interception of a node set ``S`` is then a weighted **coverage
function** over walk realisations, hence monotone submodular, so the greedy
algorithm in :func:`MassEngine.greedy_interception` carries the standard
``1 - 1/e`` guarantee and is exact at ``k = 1``.

Mass is decomposed by where a walk *ends* and by whether it crossed a feature on
the way. The distinction matters and an earlier version of this file got it
wrong:

* ``error_shadow`` -- mass **terminating** at a reconstruction-error node. This
  is emphatically *not* the mass beyond reach of a feature intervention: a walk
  can start at an error source, traverse several features, and reach the target,
  and any of those features can intercept it. Across this project's graphs the
  error-origin *feature-mediated* mass averages 0.33-0.47 depending on pruning.
* ``direct_floor`` / ``error_direct`` -- mass on single-edge embedding -> target
  and error -> target paths.
* ``feature_free`` -- the genuinely uninterceptable part: mass on walks that
  never visit a feature at all, which for these graphs is exactly
  ``direct_floor + error_direct``. ``max_interceptable = 1 - feature_free``.
  This pooled maximum includes mass absorbed at a dangling feature because that
  walk did visit an intervenable feature. Source-conditioned quantities exclude
  dangling endpoints unless a source mode explicitly includes them (none does).

``feature_free`` is computed as ``1 - P(the walk ever visits a feature)`` rather
than from the sink's first step, so it stays correct when several targets have
been collapsed into a virtual super-sink.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .core import EMBED, ERROR, FEATURE, AGraph

DEFAULT_MAX_ITER = 5000
DEFAULT_TOL = 1e-14


def backward_mass(
    An: sp.csr_matrix,
    seed: np.ndarray,
    absorbing: np.ndarray | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> np.ndarray:
    """Expected visits of the backward walk, i.e. circuit-tracer node influence.

    Args:
        An: row-normalised absolute adjacency (targets x sources).
        seed: initial mass (unit mass on the sink for the standard walk).
        absorbing: boolean mask; mass arriving at these nodes is counted but not
            propagated further. Used to compute marginal interception gains.
    """
    AnT = An.T.tocsr()
    return _walk(AnT, seed, absorbing, max_iter, tol)


def _walk(
    AnT: sp.csr_matrix,
    seed: np.ndarray,
    absorbing: np.ndarray | None,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    cur = np.asarray(seed, dtype=np.float64).copy()
    total = np.zeros_like(cur)
    for _ in range(max_iter):
        if absorbing is not None:
            cur = np.where(absorbing, 0.0, cur)
        cur = AnT @ cur
        total += cur
        if cur.sum() <= tol:
            break
    else:  # pragma: no cover - only on pathological (cyclic) input
        raise RuntimeError("backward walk did not converge; graph may not be a DAG")
    return total


@dataclass
class MassDecomposition:
    """How the unit of target-directed path mass is distributed."""

    represented: float  # absorbed at embedding nodes
    error_shadow: float  # absorbed at reconstruction-error nodes
    direct_floor: float  # embedding -> target in one edge
    error_direct: float  # error -> target in one edge
    feature_free: float  # all paths containing no feature node
    max_interceptable: float  # 1 - feature_free
    dangling: float  # absorbed at feature nodes that have no modelled inputs
    conservation: float  # mass over all absorbing nodes, should be ~1

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


class MassEngine:
    """Caches the normalised matrix and answers mass questions about one graph."""

    def __init__(self, g: AGraph, max_iter: int = DEFAULT_MAX_ITER, tol: float = DEFAULT_TOL):
        self.g = g.super_sink()
        self.An = self.g.normalized()
        self.AnT = self.An.T.tocsr()
        self.seed = self.g.sink_seed()
        self.max_iter = max_iter
        self.tol = tol
        self._base: np.ndarray | None = None

    def total(self, absorbing: np.ndarray | None = None) -> np.ndarray:
        if absorbing is None:
            if self._base is None:
                self._base = _walk(self.AnT, self.seed, None, self.max_iter, self.tol)
            return self._base
        return _walk(self.AnT, self.seed, absorbing, self.max_iter, self.tol)

    # ---- decomposition ----------------------------------------------------

    def decomposition(self) -> MassDecomposition:
        g, b = self.g, self.total()
        sink = g.sink
        emb, err = g.mask(EMBED), g.mask(ERROR)
        # Mass that reaches a source without ever crossing a feature. Making
        # features absorbing is construction-independent; reading the sink's
        # first step is not, and gave 0 under a super-sink because the first
        # step lands on a collapsed target rather than on a source.
        b_free = self.total(absorbing=g.mask(FEATURE))
        direct = float(b_free[emb].sum())
        err_direct = float(b_free[err].sum())
        # P(the walk never visits a feature). Equivalent to the sink's non-feature
        # first-step mass on a single-target graph, but also correct when targets
        # were collapsed into a super-sink, where the first step lands on a
        # target node rather than on a source.
        free = float(1.0 - self.hit_probability(g.mask(FEATURE))[sink])
        dangling = self.absorbing_nodes() & g.mask(FEATURE)
        return MassDecomposition(
            represented=float(b[emb].sum()),
            error_shadow=float(b[err].sum()),
            direct_floor=direct,
            error_direct=err_direct,
            feature_free=free,
            max_interceptable=1.0 - free,
            dangling=float(b[dangling].sum()),
            conservation=float(b[self.absorbing_nodes()].sum()),
        )

    def absorbing_nodes(self) -> np.ndarray:
        """Nodes with no incoming edges: where the backward walk terminates."""
        return np.asarray(self.An.sum(axis=1)).ravel() <= 1e-12

    def source_mask(self, mode: str) -> np.ndarray:
        """Terminal nodes counted as sources for a conditioned analysis."""
        g = self.g
        if mode == "embed":
            return g.mask(EMBED)
        if mode == "error":
            return g.mask(ERROR)
        if mode == "both":
            return g.mask(EMBED) | g.mask(ERROR)
        raise ValueError(f"source_mode must be 'embed', 'error' or 'both', got {mode!r}")

    def source_mass(self, mode: str = "embed") -> np.ndarray:
        """Per-node mass restricted to walks that terminate in the chosen sources.

        ``P(visit v AND terminate in S) = P(visit v) * P(from v, terminate in S)``
        by the Markov property, and because every node of ``S`` is absorbing,
        the second factor is exactly :meth:`hit_probability`. One extra DAG pass.

        Using this rather than the unconditioned mass is what keeps a cut's
        capacities and its source set describing the same population of walks --
        the mismatch flagged as A10 in ``REPAIR_LEDGER.md``.
        """
        S = self.source_mask(mode)
        # hit_probability counts hits strictly after the current node, so a
        # source scores 0 for itself; a walk standing on a source has already
        # terminated there, which is a hit.
        q = np.where(S, 1.0, self.hit_probability(S))
        return self.total() * q

    # ---- source-conditioned interception --------------------------------

    def source_probability(self, mode: str = "embed") -> float:
        """``P(the walk terminates in the chosen source set)``.

        This is the denominator a conditioned analysis is measured against. It
        is *not* ``source_mass(mode).sum()``, which sums expected visits over
        every node and can exceed 1.
        """
        return float(self.hit_probability(self.source_mask(mode))[self.g.sink])

    def conditioned_max_interceptable(self, mode: str = "embed") -> float:
        """``P(terminate in S AND cross at least one feature)``.

        A walk absorbed at a pruning-induced dangling feature does not terminate
        in ``S`` and is therefore outside this conditioned universe, even though
        the pooled :attr:`MassDecomposition.max_interceptable` counts it.
        """
        S = self.source_mask(mode)
        free = self.total(absorbing=self.g.mask(FEATURE))  # never crosses a feature
        return float(self.source_probability(mode) - free[S].sum())

    def _reach_source_avoiding(self, S: np.ndarray, avoid: np.ndarray) -> np.ndarray:
        """``P(from v, terminate in S without ever hitting ``avoid``)``."""
        gamma = np.zeros(self.g.n_nodes, dtype=np.float64)
        for _ in range(self.max_iter):
            nxt = self.An @ np.where(S, 1.0, np.where(avoid, 0.0, gamma))
            nxt = np.where(avoid, 0.0, np.where(S, 1.0, nxt))
            if np.max(np.abs(nxt - gamma)) <= 1e-14:
                gamma = nxt
                break
            gamma = nxt
        else:  # pragma: no cover
            raise RuntimeError("conditioned reach iteration did not converge")
        return gamma

    def intercepted_conditioned(
        self, T: list[int] | np.ndarray, mode: str = "embed"
    ) -> float:
        """``P(hit T AND terminate in S)``.

        Absorbing at ``T`` truncates the walk before it can reach a source, so
        this cannot be read off the absorbed walk directly. Decompose by first
        hit: ``sum over v in T of P(first hit of T is v) * P(from v, terminate
        in S)``.
        """
        T = np.atleast_1d(np.asarray(T, dtype=int))
        if T.size == 0:
            return 0.0
        absorbing = np.zeros(self.g.n_nodes, dtype=bool)
        absorbing[T] = True
        first_hit = self.total(absorbing)
        q = self.hit_probability(self.source_mask(mode))
        return float((first_hit[T] * q[T]).sum())

    def marginal_gains_conditioned(
        self, absorbing: np.ndarray, mode: str = "embed"
    ) -> np.ndarray:
        """Exact conditioned gain of adding each node to ``absorbing``.

        ``P(hit v AND never hit T AND terminate in S)`` factorises into
        ``P(reach v before T) * P(from v, terminate in S without hitting T)``.
        """
        beta = self.total(absorbing)
        r = self._reach_source_avoiding(self.source_mask(mode), absorbing)
        return beta * r

    def greedy_interception_conditioned(
        self, k: int = 20, mode: str = "embed", candidates: np.ndarray | None = None
    ) -> tuple[list[int], list[float]]:
        """Greedy coverage of walks that terminate in the chosen source set."""
        cand = self.candidate_mask() if candidates is None else candidates
        absorbing = np.zeros(self.g.n_nodes, dtype=bool)
        chosen: list[int] = []
        curve: list[float] = []
        covered = 0.0
        for _ in range(k):
            gains = self.marginal_gains_conditioned(absorbing, mode)
            gains = np.where(cand & ~absorbing, gains, -np.inf)
            v = int(np.argmax(gains))
            if not np.isfinite(gains[v]) or gains[v] <= 1e-15:
                break
            covered += float(gains[v])
            chosen.append(v)
            curve.append(covered)
            absorbing[v] = True
        return chosen, curve

    def source_conditioned(self) -> dict[str, float]:
        """Split terminal-source mass by origin and feature mediation.

        The four path families, which the earlier single ``error_shadow`` number
        conflated, cover walks ending at an embedding or error source. Their
        ``total`` is ``1 - dangling`` on a pruned graph, not necessarily one:

        ``embed_direct``    embedding -> target, no feature. Uninterceptable.
        ``error_direct``    error -> target, no feature. Uninterceptable.
        ``embed_mediated``  starts at an embedding, crosses >=1 feature.
        ``error_mediated``  starts at an error node, crosses >=1 feature.

        The last one is the scientifically interesting cell: the dictionary is
        incomplete there, yet a feature intervention can still intercept the
        route. Reporting it separately is what stops "error shadow" from being
        read as "beyond reach".
        """
        g, b = self.g, self.total()
        emb, err = g.mask(EMBED), g.mask(ERROR)
        b_free = self.total(absorbing=g.mask(FEATURE))
        embed_direct = float(b_free[emb].sum())
        error_direct = float(b_free[err].sum())
        out = {
            "embed_direct": embed_direct,
            "error_direct": error_direct,
            "embed_mediated": float(b[emb].sum()) - embed_direct,
            "error_mediated": float(b[err].sum()) - error_direct,
        }
        out["uninterceptable"] = embed_direct + error_direct
        out["interceptable"] = out["embed_mediated"] + out["error_mediated"]
        out["total"] = out["uninterceptable"] + out["interceptable"]
        return out

    # ---- interception -----------------------------------------------------

    def candidate_mask(self) -> np.ndarray:
        """Nodes an intervention could plausibly target: feature nodes."""
        return self.g.mask(FEATURE)

    def intercepted(self, S: list[int] | np.ndarray) -> float:
        """Exact mass of walks that hit the set ``S`` at least once."""
        S = np.atleast_1d(np.asarray(S, dtype=int))
        if S.size == 0:
            return 0.0
        absorbing = np.zeros(self.g.n_nodes, dtype=bool)
        absorbing[S] = True
        return float(self.total(absorbing)[S].sum())

    def hit_probability(self, S: np.ndarray) -> np.ndarray:
        """For every node ``v``, the probability a walk *started at* ``v`` hits ``S``.

        Solves the DAG recursion
        ``gamma(v) = sum_u An[v, u] * (1 if u in S else gamma(u))``
        by iteration, which converges in at most the depth of the DAG.

        Convention: the hit must occur **strictly after** the current node, so a
        node that is itself in ``S`` scores 0 unless it can reach ``S`` again.
        Callers that want to count ``v`` itself must handle that themselves --
        see :meth:`source_mass`.
        """
        S = np.asarray(S, dtype=bool)
        gamma = np.zeros(self.g.n_nodes, dtype=np.float64)
        for _ in range(self.max_iter):
            nxt = self.An @ np.where(S, 1.0, gamma)
            if np.max(np.abs(nxt - gamma)) <= 1e-14:
                gamma = nxt
                break
            gamma = nxt
        else:  # pragma: no cover
            raise RuntimeError("hit-probability iteration did not converge")
        return gamma

    def marginal_gains(self, absorbing: np.ndarray) -> np.ndarray:
        """Exact marginal interception gain of adding each node to ``absorbing``.

        The gain of ``v`` is ``P(hit v and never hit S)``, which factorises by the
        Markov property into ``P(reach v before S) * P(from v, never hit S)``:
        the first factor is the mass of the ``S``-absorbed walk at ``v``, the
        second is ``1 - gamma_S(v)``. Both come from one DAG pass each, so a
        greedy step costs two passes rather than one walk per candidate.

        Using only the first factor -- the obvious shortcut -- double counts
        walks that reach ``v`` first and hit ``S`` afterwards.
        """
        beta = self.total(absorbing)
        gamma = self.hit_probability(absorbing)
        return beta * (1.0 - gamma)

    def greedy_interception(
        self, k: int = 20, candidates: np.ndarray | None = None
    ) -> tuple[list[int], list[float]]:
        """Greedy maximum-coverage selection on the (submodular) coverage function.

        Returns:
            (selected node indices, cumulative intercepted mass after each pick)
        """
        cand = self.candidate_mask() if candidates is None else candidates
        absorbing = np.zeros(self.g.n_nodes, dtype=bool)
        chosen: list[int] = []
        curve: list[float] = []
        covered = 0.0
        for _ in range(k):
            gains = self.marginal_gains(absorbing)
            gains = np.where(cand & ~absorbing, gains, -np.inf)
            v = int(np.argmax(gains))
            if not np.isfinite(gains[v]) or gains[v] <= 1e-15:
                break
            covered += float(gains[v])
            chosen.append(v)
            curve.append(covered)
            absorbing[v] = True
        return chosen, curve

    def best_pair_over_top_m(self, top_m: int = 150) -> tuple[tuple[int, int], float]:
        """Best 2-node interception among pairs meeting the ``top_m`` shortlist.

        **Not a global optimum.** Only pairs with at least one member among the
        ``top_m`` highest-mass nodes are searched, so the result is exact over
        that shortlist and a lower bound otherwise. It coincides with the global
        optimum only when ``top_m`` covers every candidate, which is the case in
        the unit tests but not on real graphs.

        For each candidate ``u`` the exact gain of every partner is available in
        one shot from :meth:`marginal_gains`, so this is ``top_m`` passes rather
        than ``top_m^2`` set evaluations. Cross-checks greedy at ``k = 2``.
        """
        b = self.total()
        cand_mask = self.candidate_mask()
        cand = np.flatnonzero(cand_mask)
        if cand.size == 0:
            return (0, 0), 0.0
        top = cand[np.argsort(-b[cand])[:top_m]]
        best, best_val = (int(top[0]), int(top[0])), float(b[top[0]])
        for u in top:
            absorbing = np.zeros(self.g.n_nodes, dtype=bool)
            absorbing[int(u)] = True
            gains = self.marginal_gains(absorbing)
            gains = np.where(cand_mask & ~absorbing, gains, -np.inf)
            v = int(np.argmax(gains))
            if not np.isfinite(gains[v]):
                continue
            val = float(b[u] + gains[v])
            if val > best_val:
                best_val, best = val, (int(u), v)
        return best, best_val

    # ---- per-node diagnostics --------------------------------------------

    def error_adjacency(self) -> np.ndarray:
        """Per node, the fraction of its incoming mass coming from error nodes."""
        err = self.g.mask(ERROR)
        inc_err = np.asarray(self.An[:, err].sum(axis=1)).ravel()
        inc_all = np.asarray(self.An.sum(axis=1)).ravel()
        out = np.zeros_like(inc_all)
        nz = inc_all > 1e-12
        out[nz] = inc_err[nz] / inc_all[nz]
        return out

    def percolation(
        self, order: np.ndarray, steps: int = 20
    ) -> list[float]:
        """Cumulative intercepted mass when nodes are removed in a fixed order."""
        absorbing = np.zeros(self.g.n_nodes, dtype=bool)
        covered, curve = 0.0, []
        for v in order[:steps]:
            covered += float(self.marginal_gains(absorbing)[int(v)])
            absorbing[int(v)] = True
            curve.append(covered)
        return curve

    # Backwards-compatible alias. The old name claimed a global exactness the
    # search does not have; prefer :meth:`best_pair_over_top_m`.
    exact_best_pair = best_pair_over_top_m
