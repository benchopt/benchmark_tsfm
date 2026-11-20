Tabular foundation models
=========================

Tabular foundation models, such as TabICL and TabPFN, are transformers
pre-trained on many synthetic tables. They learn *in context*. ``fit(X, y)``
stores the training rows, and ``predict(X_test)`` runs one forward pass over
the training rows and the test rows together. No weight is updated. A single
pre-trained checkpoint thus serves any new table, as long as the problem is
cast as rows of features with a target.

This page explains how the benchmark casts time series tasks to tables, and
how this maps to the API of ``benchmark_utils/tabular_fm.py``.


Core concept: time series as tables
-----------------------------------

Classification
~~~~~~~~~~~~~~

Each sample is a ``(T, C)`` series with one label. It is flattened into a row
of ``T*C`` features. The training samples form the context table, and each test
sample is a query row. The model behaves as a scikit-learn classifier on these
rows.

::

    sample i, shape (T, C)  --flatten-->  row i: [x_1,1 ... x_T,C] | label y_i

The model sees the time steps as unordered features. The temporal structure is
only learned from the context rows.

Forecasting
~~~~~~~~~~~

Forecasting has no labelled table to start from. Each series becomes its own
regression problem instead. Each time step is a row, its value is the target,
and the features are computed from its timestamp:

- ``index``: the position of the step in the series,
- ``datetime``: calendar features, such as the day of week or the month,
- ``periodic``: sine and cosine features at the seasonal periods detected by an
  FFT on the history.

The history rows form the context. The future steps are query rows, whose
features are known in advance from their timestamps. The model predicts
quantiles of the target for each of them.

::

    history step t:  [index(t), calendar(t), periodic(t)] | value_t   (context)
    future  step h:  [index(h), calendar(h), periodic(h)] | ?         (query)

One regression model is fitted per series and per channel, with no information
shared across series. A multivariate series is forecast channel by channel.

This recipe is the forecasting pipeline of tabicl (``tabicl.forecast``). The
benchmark reuses it for every tabular model, and only swaps the regressor
called on each table.


Mapping to the API
------------------

A new model is a solver file that subclasses ``TabularFMSolver``. It provides
two estimators, and the base class wires them to the tasks.

=========================== ============================================
Subclass provides           Used for
=========================== ============================================
``make_classifier(device)`` classification, on the flattened rows
``make_regressor()``        forecasting, on the per-series tables
``quantiles_kwarg``         the name of the regressor's quantile argument
=========================== ============================================

Both estimators are returned unfitted. The base class fits them when it
receives the data.

Classification
~~~~~~~~~~~~~~

1. ``load_model`` builds ``make_pipeline(FunctionTransformer(to_tabular),
   classifier)``, where ``to_tabular`` does the flattening.
2. ``run`` calls ``build_adapter``, which fits this pipeline on the training
   samples. The fitted pipeline is the result.
3. The objective calls ``predict`` on the test samples.

Forecasting
~~~~~~~~~~~

1. ``load_model`` builds a ``TabICLForecaster`` and replaces its predictor with
   ``_RegressorEngine(regressor, quantiles_kwarg)``. The forecaster computes the
   features, and the engine fits and queries the regressor.
2. ``run`` returns the generic forecasting adapter of ``BaseTSFMSolver``, with
   no training.
3. The objective asks for forecasts at several cutoffs of each test series.
   ``BaseTSFMSolver.forecast`` cuts the history before each cutoff and passes
   batches of windows to ``forecast_batch``.
4. ``forecast_batch`` calls ``forecast_windows``. It trims each window to
   ``max_context_length`` steps, builds timestamps for the history and the
   horizon, and calls ``predict_df`` on the forecaster.
5. tabicl computes the features of each series and dispatches the series to
   worker processes, one per GPU or up to 8 on CPU. In each worker,
   ``_RegressorEngine._inference_routine`` fits a copy of the regressor on the
   history rows and predicts the quantiles of the future rows:

   .. code-block:: python

       model = copy.deepcopy(regressor).fit(train_X, train_y)
       q = model.predict(test_X, output_type="quantiles", **{quantiles_kwarg: levels})

   ``q`` is an ``(n_test, Q)`` array, or a list of ``Q`` arrays.

The regressor is sent to the workers by pickling, so it must be a plain
library estimator, not a closure or a fitted model.

Timing
~~~~~~

benchopt only times ``run``. These models load their weights in their first
``fit``, so ``load_model`` also fits a copy of each estimator on a toy table.
The checkpoint is then downloaded and cached outside the timed part. Forecasts
are computed when the objective evaluates the result, so they are not timed
either.


Adding a model
--------------

Write ``solvers/<model>.py`` as follows:

.. code-block:: python

    from benchmark_utils.tabular_fm import TabularFMSolver
    from my_model import MyClassifier, MyRegressor


    class Solver(TabularFMSolver):
        name = "MyModel"
        requirements = ["pip::my-model", "pip::tabicl[forecast]"]
        parameters = {
            "max_context_length": [512],
            "inference_batch_size": [4096],
        }
        quantiles_kwarg = "quantiles"

        def make_classifier(self, device):
            return MyClassifier(device=device)

        def make_regressor(self):
            return MyRegressor()

``name``, ``requirements`` and ``parameters`` must be literals in the solver
file. benchopt reads them before the dependencies are installed, so they are
not inherited from the base class. ``tabicl[forecast]`` is always required,
since it provides the forecasting pipeline.

A model with no quantile predictions sets
``supported_tasks = {"classification"}``, and its ``make_regressor``
raises ``NotImplementedError``. A model with limits on the input overrides ``skip``, as
``solvers/tabpfn.py`` does for more than 10 classes.
