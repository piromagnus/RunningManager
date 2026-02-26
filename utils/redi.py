"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import numpy as np


def _as_1d_float_array(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim > 1:
        return array.reshape(-1)
    return array


def compute_redi(workloads: np.ndarray | list[float], lam: float) -> np.ndarray:
    """Compute REDI cumulative workload with exponential decay.

    REDI follows the weighted-mean structure:
    REDI[t] = sum(exp(-lam * i) * workload[t-i]) / sum(exp(-lam * i))

    Missing values (NaN) are treated as absent observations: they contribute
    no load and no weight to the denominator.
    """
    if lam < 0:
        raise ValueError("lam must be >= 0 for REDI")

    values = _as_1d_float_array(workloads)
    if values.size == 0:
        return values.copy()

    decay = float(np.exp(-lam))
    observed = np.isfinite(values).astype(float)
    load_values = np.where(np.isfinite(values), values, 0.0)

    numerator = np.zeros_like(load_values, dtype=float)
    denominator = np.zeros_like(load_values, dtype=float)

    for idx in range(load_values.size):
        if idx == 0:
            numerator[idx] = load_values[idx]
            denominator[idx] = observed[idx]
        else:
            numerator[idx] = load_values[idx] + decay * numerator[idx - 1]
            denominator[idx] = observed[idx] + decay * denominator[idx - 1]

    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def compute_ewma(workloads: np.ndarray | list[float], lam: float) -> np.ndarray:
    """Compute EWMA cumulative workload.

    EWMA[t] = lam * workload[t] + (1 - lam) * EWMA[t-1]

    Missing values (NaN) keep the previous EWMA value unchanged.
    """
    if lam < 0 or lam > 1:
        raise ValueError("lam must be between 0 and 1 for EWMA")

    values = _as_1d_float_array(workloads)
    if values.size == 0:
        return values.copy()

    ewma = np.full_like(values, np.nan, dtype=float)

    first_valid_idx = np.flatnonzero(np.isfinite(values))
    if first_valid_idx.size == 0:
        return ewma

    start_idx = int(first_valid_idx[0])
    ewma[start_idx] = float(values[start_idx])

    for idx in range(start_idx + 1, values.size):
        if np.isfinite(values[idx]):
            ewma[idx] = lam * float(values[idx]) + (1.0 - lam) * ewma[idx - 1]
        else:
            ewma[idx] = ewma[idx - 1]

    return ewma
