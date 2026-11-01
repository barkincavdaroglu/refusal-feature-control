"""agtk -- classical graph algorithms for attribution graphs."""
from .core import AGraph, FEATURE, ERROR, EMBED, TARGET  # noqa: F401
from .io import load_json, load_pt, set_circuit_tracer_path  # noqa: F401
from .mass import MassEngine, backward_mass  # noqa: F401
from .redundancy_calculus import (  # noqa: F401
    ReceiverRootedMass,
    analyze_candidate_redundancy,
    candidate_excluded_c_k,
    candidate_interception_fraction,
    candidate_redundancy_across_thresholds,
    internal_vertex_disjoint_route_count,
    prune_for_receiver,
    prune_for_receiver_across_thresholds,
)

__version__ = "0.1.0"
