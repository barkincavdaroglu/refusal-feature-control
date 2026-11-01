"""Pure local topology features for the S9 E7 receiver-route panel.

The unsigned family treats row-normalized ``abs(A)`` as the established
backward random walk and asks how much receiver-started walk mass reaches the
declared sender before terminating elsewhere. The signed family first converts
Circuit Tracer attribution edges

``A[target, source] = activation(source) * J(target <- source)``

into virtual local Jacobians, then composes those coefficients by dynamic
programming. Raw attribution edges are never multiplied along paths. Gates are
applied to intermediate sparse features; the receiver gate is deliberately
excluded because E7 predicts its preactivation response.

Both families are graph-local predictors. They do not establish a physical
wire, a mediated percentage, or model-native route dependence; E7c's full
receiver restoration and rescue interventions adjudicate that separate claim.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp


TOL = 1e-12
MAX_ITER = 256


@dataclass(frozen=True)
class UnsignedRouteFeatures:
    receiver_coverage: float
    bypass_share: float
    direct_coverage: float
    indirect_coverage: float
    route_effective_count: float
    route_max_branch_share: float
    error_bypass_mass: float
    embedding_bypass_mass: float
    dangling_bypass_mass: float
    error_fraction_of_bypass: float
    walk_conservation: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignedTransportFeatures:
    direct_jacobian: float
    indirect_signed_jacobian: float
    total_signed_jacobian: float
    total_absolute_path_strength: float
    cancellation_ratio: float
    conflict_fraction: float
    path_sign_relation: str
    receiver_active: bool
    intermediate_active_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReceiverLayerCoverage:
    layer: int
    active_feature_count: int
    maximum_sender_coverage: float
    mean_sender_coverage: float
    p90_sender_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_sparse(matrix: sp.spmatrix, *, name: str) -> sp.csr_matrix:
    result = matrix.astype(np.float64).tocsr()
    if result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.isfinite(result.data).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def normalize_absolute_rows(raw_adjacency: sp.spmatrix) -> sp.csr_matrix:
    """Return incoming-edge probabilities in Circuit Tracer orientation."""

    raw = _finite_sparse(raw_adjacency, name="raw_adjacency")
    absolute = abs(raw).tocsr()
    row_sums = np.asarray(absolute.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sums)
    nonzero = row_sums > TOL
    inverse[nonzero] = 1.0 / row_sums[nonzero]
    return (sp.diags(inverse) @ absolute).tocsr()


def _validate_indices(n: int, source_index: int, receiver_index: int) -> None:
    for name, value in (("source_index", source_index), ("receiver_index", receiver_index)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < n:
            raise ValueError(f"{name} is outside the graph")
    if source_index == receiver_index:
        raise ValueError("sender and receiver must be distinct")


def _receiver_walk(
    normalized: sp.csr_matrix,
    *,
    receiver_index: int,
    source_index: int,
) -> np.ndarray:
    """Expected backward-walk visits with the declared sender absorbing."""

    current = np.zeros(normalized.shape[0], dtype=np.float64)
    current[receiver_index] = 1.0
    total = np.zeros_like(current)
    for _ in range(MAX_ITER):
        current[source_index] = 0.0
        current = normalized.T @ current
        total += current
        if float(current.sum()) <= TOL:
            break
    else:
        raise RuntimeError("receiver-started backward walk did not terminate")
    return total


def _hit_probabilities(normalized: sp.csr_matrix, *, source_index: int) -> np.ndarray:
    """Probability that a backward walk hits the sender strictly downstream."""

    hit = np.zeros(normalized.shape[0], dtype=np.float64)
    sender = np.zeros_like(hit, dtype=bool)
    sender[source_index] = True
    for _ in range(MAX_ITER):
        nxt = normalized @ np.where(sender, 1.0, hit)
        if float(np.max(np.abs(nxt - hit))) <= TOL:
            return nxt
        hit = nxt
    raise RuntimeError("sender hit-probability recursion did not converge")


def compute_sender_hit_probabilities(
    *, normalized_adjacency: sp.spmatrix, source_index: int
) -> np.ndarray:
    """Return unsigned sender coverage for every possible receiver node."""

    normalized = _finite_sparse(normalized_adjacency, name="normalized_adjacency")
    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise ValueError("source_index must be an integer")
    if not 0 <= source_index < normalized.shape[0]:
        raise ValueError("source_index is outside the graph")
    row_sums = np.asarray(normalized.sum(axis=1)).ravel()
    nonzero = row_sums > TOL
    if (normalized.data < -TOL).any() or np.any(
        np.abs(row_sums[nonzero] - 1.0) > 1e-10
    ):
        raise ValueError("normalized_adjacency must be non-negative and row-normalized")
    return _hit_probabilities(normalized, source_index=source_index)


def summarize_receiver_layers(
    *,
    sender_hit_probabilities: np.ndarray,
    node_kinds: np.ndarray,
    layers: np.ndarray,
    positions: np.ndarray,
    activations: np.ndarray,
    source_layer: int,
    token_position: int,
) -> list[ReceiverLayerCoverage]:
    """Summarize sender coverage over active feature markers in each later MLP."""

    hit = np.asarray(sender_hit_probabilities, dtype=np.float64)
    kinds = np.asarray(node_kinds)
    layer = np.asarray(layers)
    position = np.asarray(positions)
    activation = np.asarray(activations, dtype=np.float64)
    if any(len(values) != len(hit) for values in (kinds, layer, position, activation)):
        raise ValueError("receiver-layer arrays have inconsistent lengths")
    if not np.isfinite(hit).all() or (hit < -TOL).any() or (hit > 1.0 + TOL).any():
        raise ValueError("sender hit probabilities must lie in [0, 1]")
    output: list[ReceiverLayerCoverage] = []
    eligible = (
        (kinds == "feature")
        & (position == token_position)
        & (layer > source_layer)
        & (activation > 0)
    )
    for receiver_layer in sorted(set(int(value) for value in layer[eligible])):
        values = hit[eligible & (layer == receiver_layer)]
        output.append(
            ReceiverLayerCoverage(
                layer=receiver_layer,
                active_feature_count=int(len(values)),
                maximum_sender_coverage=float(values.max()),
                mean_sender_coverage=float(values.mean()),
                p90_sender_coverage=float(np.quantile(values, 0.90)),
            )
        )
    return output


def compute_unsigned_route_features(
    *,
    normalized_adjacency: sp.spmatrix,
    node_kinds: np.ndarray,
    source_index: int,
    receiver_index: int,
    sender_hit_probabilities: np.ndarray | None = None,
) -> UnsignedRouteFeatures:
    """Measure sender coverage and the remaining receiver-started walk mass."""

    normalized = _finite_sparse(normalized_adjacency, name="normalized_adjacency")
    if (normalized.data < -TOL).any():
        raise ValueError("normalized_adjacency must be non-negative")
    row_sums = np.asarray(normalized.sum(axis=1)).ravel()
    nonzero = row_sums > TOL
    if np.any(np.abs(row_sums[nonzero] - 1.0) > 1e-10):
        raise ValueError("normalized_adjacency must be row-normalized")
    kinds = np.asarray(node_kinds)
    if len(kinds) != normalized.shape[0]:
        raise ValueError("node_kinds length differs from adjacency")
    _validate_indices(normalized.shape[0], source_index, receiver_index)

    if sender_hit_probabilities is None:
        hit = _hit_probabilities(normalized, source_index=source_index)
    else:
        hit = np.asarray(sender_hit_probabilities, dtype=np.float64)
        if len(hit) != normalized.shape[0]:
            raise ValueError("sender_hit_probabilities length differs from adjacency")
        if not np.isfinite(hit).all() or (hit < -TOL).any() or (hit > 1.0 + TOL).any():
            raise ValueError("sender_hit_probabilities must lie in [0, 1]")
    visits = _receiver_walk(
        normalized,
        receiver_index=receiver_index,
        source_index=source_index,
    )
    coverage = float(hit[receiver_index])
    if not math.isclose(coverage, float(visits[source_index]), rel_tol=1e-9, abs_tol=1e-10):
        raise ArithmeticError("sender coverage disagrees across walk recursions")
    incoming = normalized.getrow(receiver_index)
    contributions: list[float] = []
    direct = 0.0
    if incoming.nnz:
        # The same recursion as hit probability, evaluated once per immediate
        # receiver branch. A second walk per branch is unnecessary: start each
        # branch with unit mass and read absorption at the declared sender.
        for predecessor, probability in zip(incoming.indices, incoming.data, strict=True):
            predecessor = int(predecessor)
            probability = float(probability)
            if predecessor == source_index:
                branch = probability
                direct += branch
            else:
                branch = probability * float(hit[predecessor])
            if branch > TOL:
                contributions.append(branch)
    branch_total = sum(contributions)
    if not math.isclose(branch_total, coverage, rel_tol=1e-9, abs_tol=1e-10):
        raise ArithmeticError("unsigned receiver-coverage branch decomposition failed")

    bypass = max(0.0, 1.0 - coverage)
    absorbing = np.asarray(normalized.sum(axis=1)).ravel() <= TOL
    error_mass = float(visits[(kinds == "error") & absorbing].sum())
    embedding_mass = float(visits[(kinds == "embed") & absorbing].sum())
    dangling_mask = (kinds == "feature") & absorbing
    dangling_mask[source_index] = False
    dangling_mass = float(visits[dangling_mask].sum())
    conservation = coverage + error_mass + embedding_mass + dangling_mass
    if not math.isclose(conservation, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ArithmeticError(
            f"receiver-started walk mass is not conserved: {conservation}"
        )
    if coverage > TOL:
        shares = np.asarray(contributions, dtype=np.float64) / coverage
        effective_count = float(1.0 / np.square(shares).sum())
        max_branch = float(shares.max())
    else:
        effective_count = 0.0
        max_branch = 0.0
    return UnsignedRouteFeatures(
        receiver_coverage=coverage,
        bypass_share=bypass,
        direct_coverage=direct,
        indirect_coverage=max(0.0, coverage - direct),
        route_effective_count=effective_count,
        route_max_branch_share=max_branch,
        error_bypass_mass=error_mass,
        embedding_bypass_mass=embedding_mass,
        dangling_bypass_mass=dangling_mass,
        error_fraction_of_bypass=(error_mass / bypass if bypass > TOL else 0.0),
        walk_conservation=conservation,
    )


def _path_sign_relation(direct: float, indirect: float, total: float) -> str:
    if abs(total) <= TOL and abs(direct) + abs(indirect) > TOL:
        return "opposed_cancellation"
    if abs(direct) <= TOL and abs(indirect) <= TOL:
        return "no_local_transport"
    if abs(direct) <= TOL:
        return "indirect_only"
    if abs(indirect) <= TOL:
        return "direct_only"
    if direct * indirect < 0:
        return "opposed_partial"
    return "aligned"


def compute_signed_transport_features(
    *,
    raw_adjacency: sp.spmatrix,
    node_kinds: np.ndarray,
    layers: np.ndarray,
    positions: np.ndarray,
    activations: np.ndarray,
    source_index: int,
    receiver_index: int,
) -> SignedTransportFeatures:
    """Compose virtual feature Jacobians in strict layer order."""

    raw = _finite_sparse(raw_adjacency, name="raw_adjacency")
    n = raw.shape[0]
    _validate_indices(n, source_index, receiver_index)
    kinds = np.asarray(node_kinds)
    layer = np.asarray(layers)
    position = np.asarray(positions)
    activation = np.asarray(activations, dtype=np.float64)
    if any(len(values) != n for values in (kinds, layer, position, activation)):
        raise ValueError("signed-transport node arrays differ from adjacency")
    if kinds[source_index] != "feature" or kinds[receiver_index] != "feature":
        raise ValueError("signed transport requires feature sender and receiver")
    if position[source_index] != position[receiver_index]:
        raise ValueError("signed transport requires sender and receiver at the same token position")
    if layer[source_index] >= layer[receiver_index]:
        raise ValueError("signed transport requires sender before receiver")
    if not math.isfinite(float(activation[source_index])) or activation[source_index] <= 0:
        raise ValueError("signed transport requires an active sender feature")

    final_position = position[source_index]
    selected = np.flatnonzero(
        (kinds == "feature")
        & (position == final_position)
        & (layer >= layer[source_index])
        & (layer <= layer[receiver_index])
    )
    local = {int(node): index for index, node in enumerate(selected)}
    source_local = local[source_index]
    receiver_local = local[receiver_index]
    sub = raw[selected][:, selected].tocoo()
    if np.any(layer[selected[sub.row]] <= layer[selected[sub.col]]):
        raise ValueError("signed feature edges must follow strictly increasing layer order")

    coefficients = sub.tocsr(copy=True)
    for row in range(coefficients.shape[0]):
        start, end = coefficients.indptr[row], coefficients.indptr[row + 1]
        columns = coefficients.indices[start:end]
        denominators = activation[selected[columns]]
        values = coefficients.data[start:end]
        invalid = (np.abs(values) > TOL) & (~np.isfinite(denominators) | (denominators <= 0))
        if invalid.any():
            raise ValueError("nonzero attribution edge leaves an inactive feature")
        coefficients.data[start:end] = np.divide(
            values,
            denominators,
            out=np.zeros_like(values),
            where=denominators > 0,
        )
    absolute_coefficients = abs(coefficients).tocsr()

    derivative = np.zeros(len(selected), dtype=np.float64)
    absolute_derivative = np.zeros(len(selected), dtype=np.float64)
    derivative[source_local] = 1.0
    absolute_derivative[source_local] = 1.0
    receiver_pre = 0.0
    receiver_abs = 0.0
    for current_layer in sorted(set(int(value) for value in layer[selected])):
        if current_layer <= int(layer[source_index]):
            continue
        rows = np.flatnonzero(layer[selected] == current_layer)
        pre = coefficients[rows] @ derivative
        pre_abs = absolute_coefficients[rows] @ absolute_derivative
        for offset, row in enumerate(rows):
            if row == receiver_local:
                receiver_pre = float(pre[offset])
                receiver_abs = float(pre_abs[offset])
            gate = 1.0 if activation[selected[row]] > 0 else 0.0
            derivative[row] = gate * float(pre[offset])
            absolute_derivative[row] = gate * float(pre_abs[offset])

    direct = float(coefficients[receiver_local, source_local])
    indirect = receiver_pre - direct
    cancellation_ratio = (
        min(1.0, abs(receiver_pre) / receiver_abs) if receiver_abs > TOL else 0.0
    )
    return SignedTransportFeatures(
        direct_jacobian=direct,
        indirect_signed_jacobian=indirect,
        total_signed_jacobian=receiver_pre,
        total_absolute_path_strength=receiver_abs,
        cancellation_ratio=cancellation_ratio,
        conflict_fraction=(1.0 - cancellation_ratio if receiver_abs > TOL else 0.0),
        path_sign_relation=_path_sign_relation(direct, indirect, receiver_pre),
        receiver_active=bool(activation[receiver_index] > 0),
        intermediate_active_count=int(
            np.sum(
                (layer[selected] > layer[source_index])
                & (layer[selected] < layer[receiver_index])
                & (activation[selected] > 0)
            )
        ),
    )
