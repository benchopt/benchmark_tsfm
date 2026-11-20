Adding a model
==============

A model is a solver file in ``solvers/`` that subclasses ``BaseTSFMSolver``
from ``benchmark_utils/base_solver.py``. The base class handles the device and
dtype, loads the model outside the timed part, and evaluates every task through
an adapter. Most models only call their network on a batch of series, and the
base class does the rest.


What a subclass implements
--------------------------

- ``supported_tasks``: the set of tasks the model can run. The solver is
  skipped on the other ones.
- ``model_id``: a string identifying the model variant. The model is reloaded
  only when it changes.
- ``load_model(device, dtype)``: returns the loaded model. It is called in
  ``set_objective``, so it is not timed.
- At least one of the three capabilities below.

==================== ====================== ===============================
Capability           Input, one per series  Output, one per series
==================== ====================== ===============================
``forecast_batch``   history ``(T, C)``     quantiles ``(H, C, Q)``
``embed_batch``      series ``(T, C)``      embedding ``(D,)``
``time_embed_batch`` series ``(T, C)``      embeddings ``(T', D)``
==================== ====================== ===============================

``forecast_batch`` also receives the covariates of each window and the
prediction length ``H``. ``T'`` is set by the model, e.g. by its patch size.


Adapter API
-----------

The objective never calls the solver directly. ``run()`` calls
``build_adapter(task, model)`` and stores the result. ``get_result()`` returns
it as ``{"model": adapter}``, and the objective calls ``adapter.predict(...)``
to evaluate it.

The default ``build_adapter`` picks a strategy from the capabilities that the
subclass overrides. It detects them with ``can_forecast``, ``can_embed`` and
``can_time_embed``. For each task, the first capability available is used:

====================== ==================== ===================================
Task                   Capability           Strategy
====================== ==================== ===================================
forecasting            ``forecast_batch``   zero-shot forecast
forecasting            ``embed_batch``      ridge regression from the embedding
                                            of the last window to the horizon
classification         ``embed_batch``      linear probe on the embedding
classification         ``time_embed_batch`` linear probe on the mean-pooled
                                            temporal embedding
anomaly_detection      ``embed_batch``      distance to the mean embedding
anomaly_detection      ``forecast_batch``   forecast error
event_detection        ``time_embed_batch`` logistic regression per position
event_detection        ``embed_batch``      logistic regression on causal
                                            windows
====================== ==================== ===================================

So most models never deal with adapters. They implement one of the three
capabilities, and the matching adapter is built for each task.

Override ``build_adapter`` only when the model is itself the predictor and
needs no adapter. For example, a tabular foundation model classifies by
in-context learning on the training labels, so its classification fits the
model directly instead of a linear probe on embeddings.
