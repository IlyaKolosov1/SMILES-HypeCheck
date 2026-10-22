"""Memory-bounded reproduction of CausalPFN's adaptive 1D retrieval batches.

This mirrors causalpfn.CausalEstimator._predict_cepo from the pinned runtime.
It yields the exact query indices and context-row indices for a full, globally
sorted query array without materializing neighbour matrices for every query.
"""

from __future__ import annotations

import numpy as np
from causalpfn._nearest_neighbors import nearest_indices_1d


def iter_native_batches(
    context_t: np.ndarray,
    context_effect: np.ndarray,
    query_effect: np.ndarray,
    *,
    max_context_length: int,
    max_query_length: int,
    num_neighbours: int,
):
    """Yield ``(query_indices, train_context_indices)`` exactly as CausalPFN.

    Context order is control rows followed by treated rows, matching the
    installed ``causalpfn`` implementation. Query order is its native
    ``np.argsort`` order; adaptive windows are allowed to overlap in their
    context-neighbour search exactly as in the reference implementation.
    """
    context_t = np.asarray(context_t)
    context_effect = np.asarray(context_effect)
    query_effect = np.asarray(query_effect)
    if context_t.ndim != 1 or context_effect.shape != context_t.shape:
        raise ValueError("Context treatment and weak scores must be aligned vectors")
    if query_effect.ndim != 1:
        raise ValueError("Query weak scores must be a vector")
    if not np.isfinite(context_effect).all() or not np.isfinite(query_effect).all():
        raise ValueError("Weak effects must be finite")
    if max_context_length < 2 or max_context_length % 2:
        raise ValueError("max_context_length must be positive and even")
    if max_query_length < 1 or num_neighbours < 1:
        raise ValueError("Query and neighbour limits must be positive")

    treatment_rows = np.where(np.isclose(context_t, 1))[0]
    control_rows = np.where(np.isclose(context_t, 0))[0]
    if not len(treatment_rows) or not len(control_rows):
        raise ValueError("Both treatment arms are required in context")
    treatment_effect = context_effect[treatment_rows]
    control_effect = context_effect[control_rows]
    query_order = np.argsort(query_effect)
    k = min(int(num_neighbours), max_context_length // 2)

    start = 0
    while start < len(query_effect):
        window_end = min(start + max_query_length, len(query_effect))
        window_queries = query_order[start:window_end]
        window_effects = query_effect[window_queries]
        neighbour_control = nearest_indices_1d(control_effect, window_effects, k)
        neighbour_treatment = nearest_indices_1d(treatment_effect, window_effects, k)

        def stratum_end(neighbour_indices: np.ndarray) -> int:
            # Same binary search and conservative boundary choice as
            # CausalEstimator._predict_cepo in the pinned package.
            left = start + 1
            right = window_end
            while right > left + 1:
                middle = (left + right) // 2
                prefix = middle - start
                unique = np.unique(neighbour_indices[:prefix].ravel())
                if len(unique) > max_context_length // 2:
                    right = middle - 1
                else:
                    left = middle
            return min(left, len(query_effect))

        control_end = stratum_end(neighbour_control)
        treatment_end = stratum_end(neighbour_treatment)
        end = min(control_end, treatment_end)
        control_neighbours = np.unique(
            neighbour_control[:control_end - start].ravel())
        treatment_neighbours = np.unique(
            neighbour_treatment[:treatment_end - start].ravel())
        context_rows = np.concatenate((
            control_rows[control_neighbours],
            treatment_rows[treatment_neighbours],
        ))
        query_rows = query_order[start:end]
        if not len(query_rows):
            raise RuntimeError("Native batching made no progress")
        if len(context_rows) > max_context_length:
            raise AssertionError("Native retrieval exceeded context limit")
        yield query_rows, context_rows
        start = end
