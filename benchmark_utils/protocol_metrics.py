"""Upstream-exact forecasting metrics of public benchmarks, from one core.

Public benchmarks agree on what a metric *is* (MASE, weighted quantile
loss, ...) but not on how they average it, treat missing values or
handle degenerate scales. A number scored "the GIFT-Eval way" can differ
from the same model's number scored "the fev way" by more than the gap
between models. This module therefore implements every metric once and
reproduces each benchmark through a :class:`Protocol`: a handful of knobs
that encode where the benchmarks really differ. The metrics are
registered as ``<protocol>_<metric>`` (e.g. ``gluonts_mase``,
``fev_sql``, ``time_crps``). Adding a benchmark means adding one
:class:`Protocol` to :data:`PROTOCOLS`.

Protocols
---------
``gluonts``
    GIFT-Eval and BOOM. Both call gluonts 0.15.1 ``evaluate_model(...,
    axis=None, mask_invalid_label=True, allow_nan_forecast=False,
    seasonality=get_seasonality(freq))`` (src/gluonts/model/evaluation.py,
    src/gluonts/ev/{metrics,stats,aggregations,ts_stats}.py).
``fev``
    fev-bench, github.com/autogluon/fev at
    edcc51ad3c1f5db6148c31a5721788afdfb703dc (src/fev/metrics.py,
    src/fev/task.py), with the ``epsilon: 1.0`` that fev-bench's
    tasks.yaml sets on WAPE and WQL.
``time``
    TIME, github.com/zqiao11/TIME at
    6d3ba1150dfe8e551bc280cba2971bdb276f4f6b
    (``compute_per_window_metrics_from_quantiles`` in
    src/timebench/evaluation/metrics.py, aggregated as in
    scripts/compute_local_leaderboard.py).

What they share
---------------
The point forecast is the 0.5 quantile. Quantile metrics use the nine
levels 0.1, ..., 0.9 (see :func:`_level_columns` for how they are picked
from the forecast) and the loss ``QL_q = 2 |(y - q̂) (1{q̂ >= y} - q)|``.
Scaled metrics divide by the seasonal-naive MAE ``s`` of each row's own
history ``x[:cutoff]``, one per (row, variate), where a row is one
(series, window) pair. A zero or undefined scale drops the affected
values in all three benchmarks. Missing ground-truth steps never count.
Everything is computed in float64.

Where they differ (the :class:`Protocol` knobs)
------------------------------------------------
=====================  ==============  ==============  ==============
knob                   gluonts         fev             time
=====================  ==============  ==============  ==============
``unit``               all             window          cell
``missing``            nonfinite       nan             nan
``history_nan``        pairwise        pairwise        collapse
``short_history``      lag1            undefined       undefined
``invalid_forecast``   nan             nan             propagate
``scaled_nonfinite``   drop            drop            propagate
``ratio_eps``          None            1.0             None
``zero_denominator``   ieee            ieee            nan
=====================  ==============  ==============  ==============

* ``unit``: what a value is computed over before averaging.
  ``all`` pools every (row, variate, step) into one value (gluonts
  ``axis=None``). ``window`` computes one value per (window, variate)
  over all the window's rows and steps, then takes ``np.mean`` over
  variates and then over windows, NaN propagating (fev). ``cell``
  computes one value per (row, variate), then ``np.nanmean`` over cells
  (TIME).
* ``missing``: which values count as missing in labels and history:
  NaN and ±inf (gluonts ``np.ma.masked_invalid``) or NaN only.
* ``history_nan``: ``pairwise`` skips lag pairs with a missing end, so
  the lag never spans a gap (gluonts masked arrays, fev ``nanmean`` of
  the differences). ``collapse`` removes missing values first and
  differences what is left, so the lag spans gaps (TIME).
* ``short_history``: with ``m >= len(history)``, ``lag1`` falls back to
  lag 1 when ``m > len(history)`` (gluonts ``seasonal_error``; with
  ``m == len(history)`` there is no pair and the scale is undefined),
  ``undefined`` leaves the scale undefined.
* ``invalid_forecast``: gluonts and fev raise on a NaN forecast. Here,
  ``nan`` returns NaN for every metric of the protocol and warns, so a
  broken model never gets a better number. ``propagate`` lets NaNs flow
  through the arithmetic as TIME does: a NaN step makes its cell NaN,
  which then drops out of the task mean (MAPE and sMAPE drop only the
  step).
* ``scaled_nonfinite``: ``drop`` skips non-finite scaled values (gluonts
  masked division, fev ``_safemean``). ``propagate`` is TIME's
  ``MAE / s``.
* ``ratio_eps`` and ``zero_denominator``: ND / WAPE and the weighted
  quantile loss divide by ``|y|``. ``ratio_eps=None`` divides sums
  (gluonts, TIME). A float divides means, flooring the denominator at
  ``eps`` (fev's ``np.maximum(mean |y|, epsilon)``). A zero denominator
  gives inf / NaN (``ieee``, gluonts) or NaN (``nan``, TIME).

Metrics
-------
The core metrics, per step and before the protocol's averaging, are
``mae`` (``|e|``), ``mse`` (``e²``), ``rmse`` (root of ``mse``, taken
per unit), ``mape`` (``|e| / |y|``), ``smape``
(``2 |e| / (|y| + |ŷ|)``), ``mase`` (``|e| / s``), ``nd`` (``Σ|e| /
Σ|y|``, fev's WAPE), ``wql`` (mean over levels of ``ΣQL_q / Σ|y|``,
GIFT-Eval's and TIME's CRPS, fev's WQL) and ``sql`` (mean over levels
of ``QL_q / s``, fev's SQL). ``mape`` and ``smape`` skip steps where
they are not finite (zero label), as gluonts and TIME do.

Windows (fev)
-------------
fev windows are global cutoffs shared by all series, scored one at a
time. Rows reach the metrics as (series, cutoff) pairs, so the
objective passes ``row_index``: per row, the series index and the
window's position **counted from the series end** (0 = last window).
Rows with the same position form one fev window. This matches fev
whenever the series end together, which is how the repo's loaders lay
windows out; a series with fewer windows lost its *early* ones, so it
still lines up with the others from the end.

Known deviations
----------------
* NaN or ±inf forecasts: gluonts and fev raise, we return NaN and warn
  (``invalid_forecast="nan"``). gluonts does not reject ±inf forecasts
  and drops them from scaled metrics; we treat them as invalid too.
* Precision: gluonts computes in the dtype of its inputs (often
  float32). We always use float64, so values can differ from a float32
  upstream run at the float32 rounding level (~1e-7 relative).
"""

import re
import warnings
from dataclasses import dataclass
from functools import partial

import numpy as np

from benchmark_utils.outputs import ForecastOutput

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_MEDIAN = QUANTILE_LEVELS.index(0.5)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Protocol:
    """How one benchmark scores forecasts. See the module docstring."""

    name: str  # metric prefix
    label: str  # benchmark name, for messages
    # exposed metric name → core metric (see ``_METRICS``); the first one
    # is the protocol's primary metric.
    metrics: tuple
    unit: str  # "all" | "window" | "cell"
    missing: str  # "nonfinite" | "nan"
    history_nan: str  # "pairwise" | "collapse"
    short_history: str  # "lag1" | "undefined"
    invalid_forecast: str  # "nan" | "propagate"
    scaled_nonfinite: str  # "drop" | "propagate"
    ratio_eps: "float | None"
    zero_denominator: str  # "ieee" | "nan"


PROTOCOLS = {
    p.name: p
    for p in (
        Protocol(
            name="gluonts",
            label="gluonts (GIFT-Eval / BOOM)",
            metrics=(
                ("mase", "mase"),
                ("crps", "wql"),
                ("mae", "mae"),
                ("mse", "mse"),
                ("rmse", "rmse"),
                ("nd", "nd"),
                ("smape", "smape"),
                ("mape", "mape"),
            ),
            unit="all",
            missing="nonfinite",
            history_nan="pairwise",
            short_history="lag1",
            invalid_forecast="nan",
            scaled_nonfinite="drop",
            ratio_eps=None,
            zero_denominator="ieee",
        ),
        Protocol(
            name="fev",
            label="fev",
            metrics=(
                ("sql", "sql"),
                ("mase", "mase"),
                ("wql", "wql"),
                ("wape", "nd"),
            ),
            unit="window",
            missing="nan",
            history_nan="pairwise",
            short_history="undefined",
            invalid_forecast="nan",
            scaled_nonfinite="drop",
            ratio_eps=1.0,
            zero_denominator="ieee",
        ),
        Protocol(
            name="time",
            label="TIME",
            metrics=(
                ("mase", "mase"),
                ("crps", "wql"),
                ("mae", "mae"),
                ("mse", "mse"),
                ("rmse", "rmse"),
                ("mape", "mape"),
                ("smape", "smape"),
                ("nd", "nd"),
            ),
            unit="cell",
            missing="nan",
            history_nan="collapse",
            short_history="undefined",
            invalid_forecast="propagate",
            scaled_nonfinite="propagate",
            ratio_eps=None,
            zero_denominator="nan",
        ),
    )
}


def protocol_metric_names(protocol: str) -> "tuple[str, ...]":
    """Registered metric names of ``protocol``, primary metric first."""
    return tuple(f"{protocol}_{name}" for name, _ in PROTOCOLS[protocol].metrics)


# ---------------------------------------------------------------------------
# Seasonality — gluonts 0.15.1 ``get_seasonality``, used by GIFT-Eval, BOOM
# and TIME for the MASE lag. gluonts isn't a dependency here, and pandas 3
# no longer parses legacy aliases ("5T", "H", "M", "Q"), so this is a port
# of ``get_seasonality`` + ``norm_freq_str`` on top of pandas 2.2+ offset
# names. It differs from ``forecasting_constants.from_pandas``: D, W → 1.
# ---------------------------------------------------------------------------
_GLUONTS_SEASONALITIES = {
    "S": 3600, "s": 3600,
    "T": 1440, "min": 1440,
    "H": 24, "h": 24,
    "D": 1,
    "W": 1,
    "M": 12, "ME": 12,
    "B": 5,
    "Q": 4, "QE": 4,
}  # fmt: skip

# "<multiple><unit>[-<anchor>]", e.g. "15T", "W-SUN", "QS-OCT".
_FREQ_RE = re.compile(r"^(\d*)([A-Za-z]+)(?:-\w+)?$")

# Legacy / alternative spellings → the pandas >= 2.2 ``offset.name`` base.
_PANDAS_OFFSET_NAME = {
    "T": "min", "H": "h", "S": "s", "L": "ms", "U": "us", "N": "ns",
    "M": "ME", "Q": "QE", "Y": "YE", "A": "YE",
    "BM": "BME", "BQ": "BQE", "BY": "BYE", "BA": "BYE",
    "AS": "YS", "BAS": "BYS",
}  # fmt: skip


def gluonts_seasonality(freq: str) -> int:
    """``gluonts.time_feature.get_seasonality(freq)`` (gluonts 0.15.1).

    The base seasonality of the frequency's unit is divided by its
    multiple ("15T" → 1440 // 15 = 96). If the multiple doesn't divide
    it, or the unit is not in gluonts' table (e.g. yearly), it is 1.
    """
    m = _FREQ_RE.match(freq)
    if not m:
        raise ValueError(f"Cannot parse frequency {freq!r}.")
    n = int(m.group(1)) if m.group(1) else 1
    name = _PANDAS_OFFSET_NAME.get(m.group(2), m.group(2))
    # norm_freq_str: the anchor is already dropped ("W-SUN" → "W"); map
    # start-anchored units to their end-anchored twin ("MS" → "ME").
    if len(name) >= 2 and name.endswith("S"):
        name = name[:-1] + "E"
    seasonality, remainder = divmod(_GLUONTS_SEASONALITIES.get(name, 1), n)
    return 1 if remainder else seasonality


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _level_columns(forecast: ForecastOutput, label: str):
    """Stacked (M, H, C, Q) quantiles and the column of each level 0.1..0.9.

    Levels are matched with a 1e-6 tolerance, so a superset (Chronos-2)
    works. A median-only point forecast (``quantile_levels=(0.5,)``) maps
    all nine levels to its single column, i.e. it is scored as a
    degenerate distribution (what upstream's savers compute when every
    quantile equals the point forecast). Any other missing level is an
    error: we never interpolate quantiles.
    """
    if len(forecast.quantiles) != 1:
        forecast = forecast.flatten()
    if not forecast.quantiles:
        raise ValueError("ForecastOutput is empty — no predictions to score")
    quants = forecast.quantiles[0]
    levels = np.asarray(forecast.quantile_levels, dtype=np.float64)
    if levels.shape == (1,) and abs(levels[0] - 0.5) <= 1e-6:
        return quants, [0] * len(QUANTILE_LEVELS)
    cols = []
    for q in QUANTILE_LEVELS:
        match = np.flatnonzero(np.abs(levels - q) <= 1e-6)
        if match.size != 1:
            raise ValueError(
                f"{label} metrics need the quantile levels "
                f"{QUANTILE_LEVELS} (or a median-only point forecast); "
                f"level {q} is missing from the forecast's {levels.tolist()}."
            )
        cols.append(int(match[0]))
    return quants, cols


class _Inputs:
    """Truth, forecast and validity, shared by every metric of one call."""

    def __init__(self, protocol: Protocol, y_true, forecast: ForecastOutput):
        quants, cols = _level_columns(forecast, protocol.label)
        self.gt = np.asarray(y_true, dtype=np.float64)  # (M, H, C)
        self.quants, self.cols = quants, cols
        self.median = self.quantile(_MEDIAN)
        if protocol.missing == "nonfinite":
            self.valid = np.isfinite(self.gt)
        else:
            self.valid = ~np.isnan(self.gt)

    def quantile(self, k: int) -> np.ndarray:
        """(M, H, C) float64 forecast of level ``QUANTILE_LEVELS[k]``."""
        return self.quants[..., self.cols[k]].astype(np.float64)

    def forecast_is_finite(self) -> bool:
        return bool(np.isfinite(self.quants[..., sorted(set(self.cols))]).all())


# ---------------------------------------------------------------------------
# Seasonal-naive scale
# ---------------------------------------------------------------------------


def seasonal_scale(contexts, n_variates: int, seasonality: int, protocol) -> np.ndarray:
    """(M, C) seasonal-naive MAE of each row's history; NaN where undefined.

    ``contexts[m]`` is the ``(cutoff, C)`` history of row ``m``. The
    scale is undefined (NaN) when there is no usable lag pair and when
    it is zero, so the values it would divide drop out.
    """
    if isinstance(protocol, str):
        protocol = PROTOCOLS[protocol]
    m = int(seasonality)
    scale = np.full((len(contexts), n_variates), np.nan)
    for row, ctx in enumerate(contexts):
        ctx = np.asarray(ctx, dtype=np.float64).reshape(len(ctx), -1)
        if protocol.history_nan == "collapse":
            for v in range(n_variates):
                c = ctx[:, v]
                c = c[
                    np.isfinite(c) if protocol.missing == "nonfinite" else ~np.isnan(c)
                ]
                if len(c) > m:
                    s = np.mean(np.abs(c[m:] - c[:-m]))
                    if s > 0:
                        scale[row, v] = s
            continue
        lag = 1 if protocol.short_history == "lag1" and m > len(ctx) else m
        if len(ctx) <= lag:
            continue
        head, tail = ctx[lag:], ctx[:-lag]
        diff = np.abs(head - tail)
        if protocol.missing == "nonfinite":
            keep = np.isfinite(head) & np.isfinite(tail)
        else:
            keep = ~np.isnan(diff)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(keep, diff, 0.0).sum(axis=0) / keep.sum(axis=0)
        scale[row] = np.where(s > 0, s, np.nan)
    return scale


# ---------------------------------------------------------------------------
# Core metrics: per-step statistics
# ---------------------------------------------------------------------------


def _abs_error(inp):
    return np.abs(inp.gt - inp.median)


def _squared_error(inp):
    return (inp.gt - inp.median) ** 2


def _ape(inp):
    return _abs_error(inp) / np.abs(inp.gt)


def _sape(inp):
    return 2 * _abs_error(inp) / (np.abs(inp.gt) + np.abs(inp.median))


def _quantile_losses(inp):
    """One (M, H, C) array of ``QL_q`` per level."""
    losses = []
    for k, q in enumerate(QUANTILE_LEVELS):
        q_pred = inp.quantile(k)
        indicator = (q_pred >= inp.gt).astype(float)
        losses.append(2 * np.abs((inp.gt - q_pred) * (indicator - q)))
    return losses


@dataclass(frozen=True)
class _Metric:
    """``kind="mean"``: mean of ``stat`` (``/ s`` if ``scaled``).
    ``kind="ratio"``: ``stat`` over ``|y|``. ``per_level`` metrics are
    averaged over the nine quantile levels; ``post`` maps each unit's
    value (e.g. sqrt for RMSE)."""

    kind: str
    stat: object
    scaled: bool = False
    per_level: bool = False
    drop_nonfinite: bool = False
    post: object = None


_METRICS = {
    "mae": _Metric("mean", _abs_error),
    "mse": _Metric("mean", _squared_error),
    "rmse": _Metric("mean", _squared_error, post=np.sqrt),
    "mape": _Metric("mean", _ape, drop_nonfinite=True),
    "smape": _Metric("mean", _sape, drop_nonfinite=True),
    "mase": _Metric("mean", _abs_error, scaled=True),
    "nd": _Metric("ratio", _abs_error),
    "wql": _Metric("ratio", _quantile_losses, per_level=True),
    "sql": _Metric("mean", _quantile_losses, scaled=True, per_level=True),
}


# ---------------------------------------------------------------------------
# Reduction over units
# ---------------------------------------------------------------------------


class _Units:
    """Per-unit sums, as a (G, C) array: G is 1 (``unit="all"``, pooled
    over variates too), the number of windows (``"window"``) or of rows
    (``"cell"``)."""

    def __init__(self, unit: str, n_rows: int, row_index, name: str):
        self.unit = unit
        if unit != "window":
            return
        if row_index is None:
            raise ValueError(
                f"{name} needs each row's window: the objective passes it as "
                "the ``row_index`` keyword ((series, window from the end) "
                "per row of y_true)."
            )
        row_index = np.asarray(row_index).reshape(-1, 2)
        if len(row_index) != n_rows:
            raise ValueError(
                f"Got a row_index of {len(row_index)} rows for {n_rows} "
                "forecast windows."
            )
        # fev scores windows in time order: farthest from the end first.
        from_end = row_index[:, 1]
        self.windows = [
            np.flatnonzero(from_end == k) for k in np.unique(from_end)[::-1]
        ]

    def sum(self, values, keep):
        """Per-unit sum of ``values`` over the kept entries, and their count."""
        values = np.where(keep, values, 0.0)
        if self.unit == "cell":
            return values.sum(axis=1), keep.sum(axis=1)
        if self.unit == "all":
            return values.sum().reshape(1, 1), keep.sum().reshape(1, 1)
        totals = [values[rows].sum(axis=(0, 1)) for rows in self.windows]
        counts = [keep[rows].sum(axis=(0, 1)) for rows in self.windows]
        return np.stack(totals), np.stack(counts)


def _combine(unit: str, values) -> float:
    """Average (G, C) per-unit values into the task value."""
    if unit == "all":
        return float(values[0, 0])
    if unit == "window":
        # fev: np.mean over target dims per window, then over windows.
        return float(np.mean([float(np.mean(per_dim)) for per_dim in values]))
    if np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def _unit_values(p, metric, name, inp, contexts, seasonality, row_index):
    """(G, C) value of a core ``metric`` per unit (see :class:`_Units`)."""
    units = _Units(p.unit, inp.gt.shape[0], row_index, name)
    scale = None
    if metric.scaled:
        if contexts is None:
            raise ValueError(
                f"{name} needs the per-window history: the objective passes "
                "it as the ``contexts`` keyword (one array per row of y_true)."
            )
        if len(contexts) != inp.gt.shape[0]:
            raise ValueError(
                f"Got {len(contexts)} contexts for {inp.gt.shape[0]} forecast windows."
            )
        scale = seasonal_scale(contexts, inp.gt.shape[2], seasonality, p)
    # The scale is constant over a cell: there, divide the cell's mean
    # once, as TIME computes ``MAE / s``. Elsewhere, scale each step.
    scale_after = metric.scaled and p.unit == "cell"
    drop = metric.drop_nonfinite or (metric.scaled and p.scaled_nonfinite == "drop")
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        stats = metric.stat(inp)
        if metric.kind == "ratio":
            den, n_den = units.sum(np.abs(inp.gt), inp.valid)
        values = []
        for x in stats if metric.per_level else [stats]:
            if metric.scaled and not scale_after:
                x = x / scale[:, None, :]
            keep = inp.valid & np.isfinite(x) if drop else inp.valid
            total, n = units.sum(x, keep)
            if metric.kind == "mean":
                value = total / n
                if scale_after:
                    value = value / scale
            elif p.ratio_eps is not None:
                value = (total / n) / np.maximum(den / n_den, p.ratio_eps)
            elif p.zero_denominator == "nan":
                value = np.where(den > 0, total / den, np.nan)
            else:
                value = total / den
            values.append(value)
        if metric.per_level:
            value = np.stack(values, axis=-1).mean(axis=-1)
        else:
            value = values[0]
        return metric.post(value) if metric.post is not None else value


def protocol_unit_values(
    protocol: str,
    metric: str,
    y_true,
    forecast: ForecastOutput,
    contexts=None,
    seasonality=1,
    row_index=None,
):
    """Per-unit (G, C) values of a core ``metric`` under ``protocol``.

    For ``unit="cell"`` (TIME) these are the per-(row, variate) cells
    upstream stores. The registered metric averages them; the
    invalid-forecast rule is not applied here.
    """
    p = PROTOCOLS[protocol]
    return _unit_values(
        p,
        _METRICS[metric],
        f"{protocol}_{metric}",
        _Inputs(p, y_true, forecast),
        contexts,
        seasonality,
        row_index,
    )


def _make_metric(p: Protocol, exposed: str, core: str):
    name = f"{p.name}_{exposed}"

    def metric(
        y_true,
        forecast: ForecastOutput,
        contexts=None,
        seasonality=1,
        row_index=None,
        **_,
    ):
        inp = _Inputs(p, y_true, forecast)
        if p.invalid_forecast == "nan" and not inp.forecast_is_finite():
            warnings.warn(
                f"{name}: the forecast has NaN or inf values, which {p.label} "
                "rejects; reporting NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
        values = _unit_values(
            p, _METRICS[core], name, inp, contexts, seasonality, row_index
        )
        return _combine(p.unit, values)

    metric.__name__ = metric.__qualname__ = name
    metric.__doc__ = (
        f"``{core}`` scored as {p.label} does (see benchmark_utils.protocol_metrics)."
    )
    return metric


# ``<protocol>_<metric>`` → metric(y_true, forecast, contexts=None,
# seasonality=1, row_index=None, **_), generated from the protocols.
PROTOCOL_METRICS = {
    f"{p.name}_{exposed}": _make_metric(p, exposed, core)
    for p in PROTOCOLS.values()
    for exposed, core in p.metrics
}

# Per-cell values of the cell-unit protocols (TIME), as upstream stores
# them: ``<protocol>_<metric>_cells(y_true, forecast, contexts=None,
# seasonality=1)`` → (rows, variates).
PROTOCOL_CELLS = {
    f"{p.name}_{exposed}_cells": partial(protocol_unit_values, p.name, core)
    for p in PROTOCOLS.values()
    if p.unit == "cell"
    for exposed, core in p.metrics
}
