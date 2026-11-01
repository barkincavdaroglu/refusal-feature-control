"""Exact path-patching semantics for a declared hook-level DAG.

This module is deliberately independent of TransformerLens and the Qwen model.
It is a small executable specification for the operation that E5a calls a
``selected_path_only`` patch:

1. cache a reference and a counterfactual forward pass;
2. replace one or more declared sender states with their counterfactual values;
3. let those values travel only across explicitly selected edges; and
4. after a declared release node, recompute its ordinary downstream descendants.

The result is exact for the DAG and hook granularity supplied to :class:`HookDAG`.
It is not a claim that the declared hooks expose every microscopic transformer
edge.  The production experiment must state that coarser boundary separately.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


NodeFunction = Callable[[Mapping[str, Any]], Any]
Edge = tuple[str, str]


@dataclass(frozen=True)
class DAGNode:
    """One node in a topologically ordered hook DAG.

    Root nodes have no parents and no function; callers provide their values to
    :meth:`HookDAG.evaluate`.  Every non-root node receives a mapping containing
    exactly its named parent values.
    """

    name: str
    parents: tuple[str, ...] = ()
    function: NodeFunction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("DAG node names must be non-empty strings")
        if len(self.parents) != len(set(self.parents)):
            raise ValueError(f"{self.name}: parent names must be unique")
        if self.parents and self.function is None:
            raise ValueError(f"{self.name}: non-root nodes require a function")
        if not self.parents and self.function is not None:
            raise ValueError(f"{self.name}: root nodes must not define a function")


@dataclass(frozen=True)
class PathPatchResult:
    """Reference, counterfactual, and treeified path-patch states."""

    reference: Mapping[str, Any]
    counterfactual: Mapping[str, Any]
    patched: Mapping[str, Any]
    selected_edges: tuple[Edge, ...]
    source_overrides: tuple[str, ...]
    release_nodes: tuple[str, ...]
    open_nodes: tuple[str, ...]


class HookDAG:
    """A strict topological computation graph and exact path-splice oracle."""

    def __init__(self, nodes: Sequence[DAGNode], *, output: str) -> None:
        if not nodes:
            raise ValueError("HookDAG requires at least one node")
        self.nodes = tuple(nodes)
        self.output = output
        self._by_name: dict[str, DAGNode] = {}
        self._position: dict[str, int] = {}
        for position, node in enumerate(self.nodes):
            if node.name in self._by_name:
                raise ValueError(f"Duplicate DAG node: {node.name}")
            unknown = [parent for parent in node.parents if parent not in self._by_name]
            if unknown:
                raise ValueError(
                    f"{node.name}: parents must appear earlier in topological order: {unknown}"
                )
            self._by_name[node.name] = node
            self._position[node.name] = position
        if output not in self._by_name:
            raise ValueError(f"Unknown output node: {output}")
        self.roots = tuple(node.name for node in self.nodes if not node.parents)
        self.edges = frozenset(
            (parent, node.name) for node in self.nodes for parent in node.parents
        )
        children: dict[str, list[str]] = defaultdict(list)
        for source, target in self.edges:
            children[source].append(target)
        self._children = {
            name: tuple(sorted(values, key=self._position.__getitem__))
            for name, values in children.items()
        }

    def evaluate(self, root_values: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate the DAG, requiring exactly one value for every root."""

        supplied = set(root_values)
        expected = set(self.roots)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            raise ValueError(
                f"Root inputs differ; missing={missing}, unexpected={unexpected}"
            )
        states: dict[str, Any] = {}
        for node in self.nodes:
            if not node.parents:
                states[node.name] = root_values[node.name]
                continue
            assert node.function is not None
            states[node.name] = node.function(
                {parent: states[parent] for parent in node.parents}
            )
        return states

    def _validate_path_spec(
        self,
        *,
        source_overrides: Mapping[str, Any],
        selected_edges: frozenset[Edge],
        release_nodes: frozenset[str],
    ) -> None:
        if not source_overrides:
            raise ValueError("A path patch requires at least one source override")
        unknown_sources = sorted(set(source_overrides) - set(self._by_name))
        if unknown_sources:
            raise ValueError(f"Unknown source override nodes: {unknown_sources}")
        unknown_edges = sorted(selected_edges - self.edges)
        if unknown_edges:
            raise ValueError(f"Selected edges are absent from the DAG: {unknown_edges}")
        unknown_release = sorted(release_nodes - set(self._by_name))
        if unknown_release:
            raise ValueError(f"Unknown release nodes: {unknown_release}")
        if not release_nodes:
            raise ValueError("A path patch requires at least one release node")

        selected_children: dict[str, list[str]] = defaultdict(list)
        selected_parents: dict[str, list[str]] = defaultdict(list)
        for source, target in selected_edges:
            selected_children[source].append(target)
            selected_parents[target].append(source)

        forward = set(source_overrides)
        queue = deque(source_overrides)
        while queue:
            source = queue.popleft()
            for target in selected_children.get(source, ()):
                if target not in forward:
                    forward.add(target)
                    queue.append(target)
        unreachable_release = sorted(release_nodes - forward)
        if unreachable_release:
            raise ValueError(
                "Release nodes are not reachable from a source override through selected "
                f"edges: {unreachable_release}"
            )

        backward = set(release_nodes)
        queue = deque(release_nodes)
        while queue:
            target = queue.popleft()
            for source in selected_parents.get(target, ()):
                if source not in backward:
                    backward.add(source)
                    queue.append(source)
        unused = sorted(
            edge
            for edge in selected_edges
            if edge[0] not in forward or edge[1] not in backward
        )
        if unused:
            raise ValueError(
                "Every selected edge must lie on a selected source-to-release route: "
                f"{unused}"
            )

    def path_patch(
        self,
        *,
        reference_roots: Mapping[str, Any],
        counterfactual_roots: Mapping[str, Any],
        source_nodes: Sequence[str],
        selected_edges: Sequence[Edge],
        release_nodes: Sequence[str],
    ) -> PathPatchResult:
        """Run an exact treeified path patch on the declared hook DAG.

        A source node may be a root or an internal hook.  Its replacement value
        is taken from the complete counterfactual cache.  Before a release node,
        a mixed value travels only on ``selected_edges``.  A release node and all
        ordinary descendants are then recomputed, while unaffected parent inputs
        retain their reference values.
        """

        if len(source_nodes) != len(set(source_nodes)):
            raise ValueError("source_nodes must be unique")
        selected = frozenset(tuple(edge) for edge in selected_edges)
        if len(selected) != len(selected_edges):
            raise ValueError("selected_edges must be unique")
        releases = frozenset(release_nodes)
        if len(releases) != len(release_nodes):
            raise ValueError("release_nodes must be unique")

        reference = self.evaluate(reference_roots)
        counterfactual = self.evaluate(counterfactual_roots)
        overrides = {name: counterfactual.get(name) for name in source_nodes}
        self._validate_path_spec(
            source_overrides=overrides,
            selected_edges=selected,
            release_nodes=releases,
        )

        patched: dict[str, Any] = {}
        open_nodes: set[str] = set()
        override_names = set(overrides)
        for node in self.nodes:
            if node.name in override_names:
                patched[node.name] = overrides[node.name]
            elif not node.parents:
                patched[node.name] = reference[node.name]
            else:
                assert node.function is not None
                inputs: dict[str, Any] = {}
                for parent in node.parents:
                    use_mixed = (parent, node.name) in selected or parent in open_nodes
                    inputs[parent] = patched[parent] if use_mixed else reference[parent]
                patched[node.name] = node.function(inputs)

            if node.name in releases or any(parent in open_nodes for parent in node.parents):
                open_nodes.add(node.name)

        return PathPatchResult(
            reference=reference,
            counterfactual=counterfactual,
            patched=patched,
            selected_edges=tuple(sorted(selected, key=lambda edge: (self._position[edge[0]], self._position[edge[1]]))),
            source_overrides=tuple(sorted(override_names, key=self._position.__getitem__)),
            release_nodes=tuple(sorted(releases, key=self._position.__getitem__)),
            open_nodes=tuple(sorted(open_nodes, key=self._position.__getitem__)),
        )

