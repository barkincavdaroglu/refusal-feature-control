"""E7a receiver-restoration and receiver-rescue semantics on a hook DAG.

The implementation is deliberately model-free. It treats a transformer module
as two distinct hook boundaries:

``module_input -> module_output``

while a residual state bypasses the module output and rejoins it downstream.
The sender is patched into a baseline run from a complete counterfactual cache.
Receiver restoration and rescue then create explicit hybrid worlds by replacing
either complete module inputs or complete module outputs. The two boundaries
are never pooled.

These operations measure intervention-defined receiver dependence and
sufficiency. They do not identify a unique mediated path or a percentage
mediated, even though the sender-total bookkeeping identity is exact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any

from path_patching_semantics import HookDAG


class ReceiverBoundary(str, Enum):
    """Complete model-native boundary replaced by an E7 intervention."""

    MODULE_INPUT = "module_input"
    MODULE_OUTPUT = "module_output"


@dataclass(frozen=True)
class ReceiverModuleBoundary:
    """Hook names for one transformer-shaped receiver module.

    ``residual_bypass_node`` names the residual state that goes around the
    module write. Under receiver restoration it must remain in the
    sender-patched world; under rescue it remains in the baseline world.
    """

    name: str
    input_node: str
    output_node: str
    residual_bypass_node: str

    def __post_init__(self) -> None:
        fields = {
            "name": self.name,
            "input_node": self.input_node,
            "output_node": self.output_node,
            "residual_bypass_node": self.residual_bypass_node,
        }
        invalid = [key for key, value in fields.items() if not isinstance(value, str) or not value]
        if invalid:
            raise ValueError(f"Receiver boundary fields must be non-empty strings: {invalid}")
        hook_nodes = (self.input_node, self.output_node, self.residual_bypass_node)
        if len(set(hook_nodes)) != len(hook_nodes):
            raise ValueError(f"{self.name}: input, output, and residual bypass must be distinct")


@dataclass(frozen=True)
class RouteContrasts:
    """Raw E7 endpoint worlds and continuous intervention contrasts."""

    y0: float
    ys: float
    ys_r0: float
    yr_s: float
    t_s: float
    d_r: float
    b_r: float
    q_r: float
    bookkeeping_error: float


@dataclass(frozen=True)
class ReceiverRouteResult:
    """All cached and hybrid states for one receiver boundary and set."""

    boundary: ReceiverBoundary
    sender_nodes: tuple[str, ...]
    receivers: tuple[ReceiverModuleBoundary, ...]
    baseline: Mapping[str, Any]
    counterfactual: Mapping[str, Any]
    sender_patched: Mapping[str, Any]
    receiver_restored: Mapping[str, Any]
    receiver_rescue: Mapping[str, Any]
    contrasts: RouteContrasts


def _scalar_endpoint(value: Any, *, world: str) -> float:
    if isinstance(value, Real):
        return float(value)
    numel = getattr(value, "numel", None)
    item = getattr(value, "item", None)
    if callable(numel) and callable(item) and int(numel()) == 1:
        return float(item())
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{world} endpoint must be scalar, got {type(value).__name__}") from error


def _coerce_boundary(boundary: ReceiverBoundary | str) -> ReceiverBoundary:
    try:
        return ReceiverBoundary(boundary)
    except ValueError as error:
        allowed = [item.value for item in ReceiverBoundary]
        raise ValueError(f"Unknown receiver boundary {boundary!r}; expected one of {allowed}") from error


def _evaluate_with_overrides(
    graph: HookDAG,
    root_values: Mapping[str, Any],
    node_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one hook-DAG world with complete topological replacements.

    This E7-only delta intentionally lives outside ``path_patching_semantics``:
    that E5a source file is hash-pinned into completed experimental plans.
    """

    supplied = set(root_values)
    expected = set(graph.roots)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValueError(f"Root inputs differ; missing={missing}, unexpected={unexpected}")
    graph_names = {node.name for node in graph.nodes}
    unknown_overrides = sorted(set(node_overrides) - graph_names)
    if unknown_overrides:
        raise ValueError(f"Unknown node overrides: {unknown_overrides}")

    states: dict[str, Any] = {}
    for node in graph.nodes:
        if node.name in node_overrides:
            states[node.name] = node_overrides[node.name]
        elif not node.parents:
            states[node.name] = root_values[node.name]
        else:
            assert node.function is not None
            states[node.name] = node.function(
                {parent: states[parent] for parent in node.parents}
            )
    return states


def _validate_contract(
    graph: HookDAG,
    *,
    sender_nodes: Sequence[str],
    receivers: Sequence[ReceiverModuleBoundary],
) -> None:
    graph_names = {node.name for node in graph.nodes}
    positions = {node.name: index for index, node in enumerate(graph.nodes)}

    if not sender_nodes:
        raise ValueError("At least one sender node is required")
    if len(sender_nodes) != len(set(sender_nodes)):
        raise ValueError("Sender nodes must be distinct")
    unknown_senders = sorted(set(sender_nodes) - graph_names)
    if unknown_senders:
        raise ValueError(f"Unknown sender nodes: {unknown_senders}")

    if not receivers:
        raise ValueError("At least one receiver module is required")
    receiver_names = [receiver.name for receiver in receivers]
    if len(receiver_names) != len(set(receiver_names)):
        raise ValueError("Receiver modules must have distinct names")

    intervention_nodes: list[str] = []
    for receiver in receivers:
        declared = {
            receiver.input_node,
            receiver.output_node,
            receiver.residual_bypass_node,
        }
        unknown = sorted(declared - graph_names)
        if unknown:
            raise ValueError(f"{receiver.name}: unknown receiver nodes: {unknown}")
        if (receiver.input_node, receiver.output_node) not in graph.edges:
            raise ValueError(
                f"{receiver.name}: module output must directly consume the declared module input"
            )
        if positions[receiver.input_node] >= positions[receiver.output_node]:
            raise ValueError(f"{receiver.name}: module input must precede module output")
        for sender in sender_nodes:
            if positions[sender] >= positions[receiver.input_node]:
                raise ValueError(
                    f"{receiver.name}: sender {sender} must precede the receiver input"
                )
        intervention_nodes.extend((receiver.input_node, receiver.output_node))

    if len(intervention_nodes) != len(set(intervention_nodes)):
        raise ValueError("Receiver modules must use distinct input and output hook nodes")
    overlap = sorted(set(sender_nodes) & set(intervention_nodes))
    if overlap:
        raise ValueError(f"Sender and receiver intervention nodes overlap: {overlap}")


def _route_contrasts(
    *,
    y0: float,
    ys: float,
    ys_r0: float,
    yr_s: float,
) -> RouteContrasts:
    t_s = ys - y0
    d_r = ys - ys_r0
    b_r = ys_r0 - y0
    q_r = yr_s - y0
    bookkeeping_error = t_s - (d_r + b_r)
    if not math.isclose(t_s, d_r + b_r, rel_tol=1e-12, abs_tol=1e-12):
        raise ArithmeticError(
            "Sender-total bookkeeping failed: "
            f"T_S={t_s}, D_R={d_r}, B_R={b_r}, error={bookkeeping_error}"
        )
    return RouteContrasts(
        y0=y0,
        ys=ys,
        ys_r0=ys_r0,
        yr_s=yr_s,
        t_s=t_s,
        d_r=d_r,
        b_r=b_r,
        q_r=q_r,
        bookkeeping_error=bookkeeping_error,
    )


def measure_receiver_route(
    *,
    graph: HookDAG,
    baseline_roots: Mapping[str, Any],
    counterfactual_roots: Mapping[str, Any],
    sender_nodes: Sequence[str],
    receivers: Sequence[ReceiverModuleBoundary],
    boundary: ReceiverBoundary | str = ReceiverBoundary.MODULE_OUTPUT,
) -> ReceiverRouteResult:
    """Measure sender total, receiver dependence, remainder, and rescue.

    The sender patch copies only ``sender_nodes`` from the complete
    counterfactual cache into the baseline roots. Receiver restoration adds
    baseline receiver overrides to that sender-patched run. Receiver rescue
    instead copies the sender-induced receiver state into an otherwise baseline
    run. Passing multiple receivers performs one simultaneous receiver-set
    intervention.
    """

    selected_boundary = _coerce_boundary(boundary)
    sender_tuple = tuple(sender_nodes)
    receiver_tuple = tuple(receivers)
    _validate_contract(
        graph,
        sender_nodes=sender_tuple,
        receivers=receiver_tuple,
    )

    baseline = graph.evaluate(baseline_roots)
    counterfactual = graph.evaluate(counterfactual_roots)
    sender_overrides = {name: counterfactual[name] for name in sender_tuple}
    sender_patched = _evaluate_with_overrides(graph, baseline_roots, sender_overrides)

    def receiver_node(receiver: ReceiverModuleBoundary) -> str:
        if selected_boundary is ReceiverBoundary.MODULE_INPUT:
            return receiver.input_node
        return receiver.output_node

    restoration_overrides = {
        receiver_node(receiver): baseline[receiver_node(receiver)]
        for receiver in receiver_tuple
    }
    receiver_restored = _evaluate_with_overrides(
        graph,
        baseline_roots,
        {**sender_overrides, **restoration_overrides},
    )
    rescue_overrides = {
        receiver_node(receiver): sender_patched[receiver_node(receiver)]
        for receiver in receiver_tuple
    }
    receiver_rescue = _evaluate_with_overrides(graph, baseline_roots, rescue_overrides)

    contrasts = _route_contrasts(
        y0=_scalar_endpoint(baseline[graph.output], world="Y0"),
        ys=_scalar_endpoint(sender_patched[graph.output], world="YS"),
        ys_r0=_scalar_endpoint(receiver_restored[graph.output], world="YS,R0"),
        yr_s=_scalar_endpoint(receiver_rescue[graph.output], world="YR(S)"),
    )
    return ReceiverRouteResult(
        boundary=selected_boundary,
        sender_nodes=sender_tuple,
        receivers=receiver_tuple,
        baseline=baseline,
        counterfactual=counterfactual,
        sender_patched=sender_patched,
        receiver_restored=receiver_restored,
        receiver_rescue=receiver_rescue,
        contrasts=contrasts,
    )
