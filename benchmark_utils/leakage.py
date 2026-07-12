"""Leakage detection for forecasting adapters.

A forecasting adapter receives the *full* series ``x[i]`` of shape
``(T_i, C)`` plus per-series cutoff indexes, and is contractually bound
to use only ``x[i][:cutoff]`` as history (see
:mod:`benchmark_utils.adapters.base`). A model that peeks at
``x[i][cutoff:]`` — the future, which contains the evaluation target —
has *leakage*, and any score it earns is invalid.

This module probes for leakage *behaviourally*, without inspecting model
internals: it perturbs the future segment of the target series and
checks that the forecasts are unchanged. Future *covariates* are
legitimately known ahead of time, so they are NOT perturbed — only the
future target values in ``x`` are.

Soundness
---------
For a series with cutoffs ``C_i``, we perturb only indices
``>= max(C_i)``. The legitimate history for every cutoff ``c`` in
``C_i`` is ``x[:c]``, a subset of ``x[:max(C_i)]``, which is left
untouched. Therefore any change in the forecast is caused solely by the
model reading future target values — i.e. leakage. There are no false
positives.

Coverage caveat: leakage that only reads the gap ``[c, max(C_i))`` of an
*earlier* cutoff is not exercised. With the default single-window
evaluation (:func:`benchmark_utils.windowing.make_forecasting_splits`
with ``n_windows=1``) each series has exactly one cutoff, so coverage is
complete.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput


@dataclass(frozen=True)
class LeakageReport:
    """Result of a behavioural leakage probe.

    Attributes
    ----------
    leaked : bool
        True if perturbing the future target changed any forecast.
    max_abs_diff : float
        Largest absolute change in any quantile across all tested series.
        ``0.0`` when nothing was tested.
    n_series_tested : int
        Series that had a perturbable future segment.
    n_series_skipped : int
        Series with no future to perturb (``max(cutoff) >= T``) or no
        cutoffs — leakage cannot be exercised for these.
    offending_series : list of int
        Indexes (into the original ``x``) of series whose forecast
        changed under perturbation.
    """

    leaked: bool
    max_abs_diff: float
    n_series_tested: int
    n_series_skipped: int
    offending_series: List[int] = field(default_factory=list)


def _perturb_future(
    series: np.ndarray, cut: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a copy of ``series`` with indices ``>= cut`` replaced.

    The replacement is a high-magnitude random signal that is independent
    of the original values, so a leaky model is caught whether it *reads*,
    *adds*, or *scales by* the future — there is no perturbation a
    non-degenerate model could be invariant to by coincidence.
    """
    out = np.array(series, dtype=np.float64, copy=True)
    future_shape = out[cut:].shape
    # Scale relative to the history so the perturbation dominates any
    # plausible signal the model might be reading, then shift far off.
    hist = out[:cut]
    scale = 10.0 * (float(np.nanstd(hist)) + 1.0) if hist.size else 100.0
    out[cut:] = rng.standard_normal(future_shape) * scale + 1e4
    return out


def detect_forecast_leakage(
    adapter,
    forecast_input: ForecastInput,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    seed: int = 0,
    max_series: Optional[int] = None,
) -> LeakageReport:
    """Probe a fitted forecasting adapter for target leakage.

    Runs ``adapter.predict`` twice — once on ``forecast_input`` and once
    on a copy whose future target values (beyond each series' last
    cutoff) are perturbed — and compares the quantile forecasts. A
    leakage-free model produces identical forecasts; any difference
    beyond ``rtol``/``atol`` is flagged.

    Parameters
    ----------
    adapter : BaseTSFMAdapter
        A fitted forecasting adapter (``predict(ForecastInput) ->
        ForecastOutput``).
    forecast_input : ForecastInput
        The input to probe. Only the target ``x`` is perturbed;
        ``covariates`` (including ``future_covars``) are passed through
        unchanged, as future covariates are legitimately known.
    rtol, atol : float
        Tolerances forwarded to :func:`numpy.allclose`.
    seed : int
        Seed for the perturbation RNG (deterministic).
    max_series : int or None
        If given, only the first ``max_series`` series are probed — useful
        for a cheap in-loop check on large test sets.

    Returns
    -------
    LeakageReport
    """
    rng = np.random.default_rng(seed)

    n = len(forecast_input.x)
    if max_series is not None:
        n = min(n, max_series)
    probe_idx = list(range(n))

    # Build a perturbed input over the SAME series order/shapes, touching
    # only the probed series' future. Untouched series are passed through
    # so the batch shape the model sees is identical.
    perturbed_x = [np.asarray(s) for s in forecast_input.x]
    tested, skipped = [], 0
    for i in probe_idx:
        series = np.asarray(forecast_input.x[i])
        cutoffs = forecast_input.cutoff_indexes[i]
        if len(cutoffs) == 0:
            skipped += 1
            continue
        cut = int(max(cutoffs))
        if cut >= series.shape[0]:
            skipped += 1
            continue
        perturbed_x[i] = _perturb_future(series, cut, rng)
        tested.append(i)

    if not tested:
        return LeakageReport(
            leaked=False,
            max_abs_diff=0.0,
            n_series_tested=0,
            n_series_skipped=skipped,
        )

    baseline = adapter.predict(forecast_input)
    perturbed = adapter.predict(
        ForecastInput(
            x=perturbed_x,
            cutoff_indexes=forecast_input.cutoff_indexes,
            covariates=forecast_input.covariates,
        )
    )
    _validate_output(baseline)
    _validate_output(perturbed)

    offending: List[int] = []
    max_abs_diff = 0.0
    for i in tested:
        a = np.asarray(baseline.quantiles[i], dtype=np.float64)
        b = np.asarray(perturbed.quantiles[i], dtype=np.float64)
        if a.shape != b.shape:
            # Output shape itself depends on the future — unambiguous leakage.
            offending.append(i)
            max_abs_diff = float("inf")
            continue
        diff = np.abs(a - b)
        finite = diff[np.isfinite(diff)]
        if finite.size:
            max_abs_diff = max(max_abs_diff, float(finite.max()))
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            offending.append(i)

    return LeakageReport(
        leaked=bool(offending),
        max_abs_diff=max_abs_diff,
        n_series_tested=len(tested),
        n_series_skipped=skipped,
        offending_series=offending,
    )


def _validate_output(output) -> None:
    if not isinstance(output, ForecastOutput):
        raise TypeError(
            "leakage probe expects predict() to return a ForecastOutput, "
            f"got {type(output).__name__}"
        )
