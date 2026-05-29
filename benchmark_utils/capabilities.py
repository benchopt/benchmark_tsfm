"""Model capability flags and covariate masking.

Vocabulary
----------
A forecasting solver declares a ``capabilities`` set drawn from:

- :data:`MULTIVARIATE`      — the model treats target channels jointly.
  *Declarative only*: targets are always passed whole (no channel
  splitting), so there is no behavioural toggle for this yet — it exists to
  describe the model until a multivariate-*target* dataset and the matching
  masking land.
- :data:`HIST_COVARIATES`   — the model consumes history-only (past) covariates.
- :data:`FUTURE_COVARIATES` — the model consumes known-ahead (future) covariates.

``univariate`` is deliberately **not** a flag — it is the floor every model
gets. A model that declares (or has enabled) none of the covariate
capabilities runs univariate.

Deactivation / lift
-------------------
The covariate capabilities are independently switchable per run (exposed as
benchopt parameters by the consuming solver), so the lift each one provides
can be benchmarked. Enforcement is central: the objective masks the
:class:`~benchmark_utils.covariates.Covariates` payload down to the adapter's
*effective* active set (``BaseTSFMAdapter.covariate_capabilities``) via
:func:`mask_covariates` before calling ``predict``. A model therefore only
ever sees covariates it both declares and has enabled. Targets are never
masked.
"""

from benchmark_utils.covariates import Covariates

MULTIVARIATE = "multivariate"
HIST_COVARIATES = "hist_covariates"
FUTURE_COVARIATES = "future_covariates"

#: Capabilities whose covariate payload :func:`mask_covariates` acts on.
COVARIATE_CAPABILITIES = frozenset({HIST_COVARIATES, FUTURE_COVARIATES})

#: Every capability in the vocabulary.
ALL_CAPABILITIES = frozenset({MULTIVARIATE, HIST_COVARIATES, FUTURE_COVARIATES})


def mask_covariates(covariates: Covariates, active) -> Covariates:
    """Return a copy of ``covariates`` with disabled covariate fields emptied.

    ``hist_covars`` is cleared unless :data:`HIST_COVARIATES` is in ``active``,
    and ``future_covars`` unless :data:`FUTURE_COVARIATES` is in ``active``.
    ``static_covars`` is passed through unchanged — it is not yet part of the
    capability vocabulary. Targets live in ``ForecastInput.x`` and are never
    touched here.

    Parameters
    ----------
    covariates : Covariates
        The dataset's full covariate payload.
    active : Iterable[str]
        The effective active capability names (typically an adapter's
        ``covariate_capabilities``).

    Returns
    -------
    Covariates
        A new (frozen) instance; the input is not mutated.
    """
    active = frozenset(active)
    return Covariates(
        static_covars=covariates.static_covars,
        hist_covars=(
            covariates.hist_covars if HIST_COVARIATES in active else []
        ),
        future_covars=(
            covariates.future_covars if FUTURE_COVARIATES in active else []
        ),
    )
