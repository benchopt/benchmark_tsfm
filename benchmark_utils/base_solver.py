"""Abstract base solver for Time Series Foundation Models (TSFM)."""

from abc import abstractmethod
from typing import Any, Literal, Sequence

import numpy as np
import torch
from benchopt import BaseSolver

from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.adapters.forecast_residual import ForecastResidualAdapter
from benchmark_utils.adapters.linear_probe import LinearProbeAdapter
from benchmark_utils.covariates import Covariates
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput

TaskType = Literal[
    "forecasting", "classification", "anomaly_detection", "event_detection"
]


# ---------------------------------------------------------------------------
# Private adapter / encoder helpers used by build_adapter defaults
# ---------------------------------------------------------------------------


class _SolverForecastAdapter(BaseTSFMAdapter):
    """Wraps BaseTSFMSolver.forecast() as a BaseTSFMAdapter."""

    _DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, solver, quantile_levels=None):
        self.solver = solver
        self.quantile_levels = quantile_levels or self._DEFAULT_QUANTILE_LEVELS

    def predict(self, x: ForecastInput, prediction_length=None) -> ForecastOutput:
        horizon = prediction_length or self.solver.meta.get("prediction_length", 1)
        return self.solver.forecast(x, horizon, self.quantile_levels)


class _SolverEmbedEncoder:
    """Wraps BaseTSFMSolver.embed() as a flat encoder for LinearProbeAdapter."""

    def __init__(self, solver):
        self.solver = solver

    def encode(self, X: np.ndarray) -> np.ndarray:
        # X: (B, T, C) or (T, C); returns (B, D)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            return self.solver.embed([X])
        return self.solver.embed(list(X))


class _SolverTimeEmbedPooledEncoder:
    """Wraps BaseTSFMSolver.time_embed() with mean pooling for LinearProbeAdapter."""

    def __init__(self, solver):
        self.solver = solver

    def encode(self, X: np.ndarray) -> np.ndarray:
        # X: (B, T, C) or (T, C); returns (B, D)
        X = np.asarray(X, dtype=np.float32)
        series_list = [X] if X.ndim == 2 else list(X)
        time_embs = self.solver.time_embed(series_list)  # list of (T'_i, D)
        return np.stack([emb.mean(axis=0) for emb in time_embs], axis=0)  # (B, D)


class _WindowedForecastAdapter(BaseTSFMAdapter):
    """Point forecast via embed on sliding windows + ridge regression.

    Builds (window_embedding → next_H_values) training pairs from the
    training series, then at inference embeds the last ``window_size``
    timesteps before each cutoff and predicts the next ``prediction_length``
    values. Always outputs a single quantile at 0.5 (point forecast).
    """

    def __init__(self, solver, window_size: int, prediction_length: int):
        self.solver = solver
        self.window_size = window_size
        self.prediction_length = prediction_length
        self._head = None

    def fit(self, X_train, y_train=None):
        from sklearn.linear_model import Ridge

        windows, targets = [], []
        for series in X_train:
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            T, C = series.shape
            for t in range(self.window_size, T - self.prediction_length + 1):
                windows.append(series[t - self.window_size: t])
                targets.append(series[t: t + self.prediction_length].flatten())

        if not windows:
            return self
        embs = self.solver.embed(windows)                    # (N, D)
        self._head = Ridge().fit(embs, np.stack(targets))    # targets: (N, H*C)
        return self

    def predict(self, x: ForecastInput, prediction_length=None) -> ForecastOutput:
        # Ignore prediction_length override — trained for a fixed horizon.
        H = self.prediction_length
        windows, layout, per_series_shape = [], [], []

        for series_idx, (series, cutoffs) in enumerate(zip(x.x, x.cutoff_indexes)):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            _, C = series.shape
            per_series_shape.append((C, len(cutoffs)))
            for cutoff_idx, cutoff in enumerate(cutoffs):
                hist = series[:cutoff]
                if len(hist) >= self.window_size:
                    window = hist[-self.window_size:]
                else:
                    pad = np.zeros(
                        (self.window_size - len(hist), hist.shape[1]), dtype=np.float32
                    )
                    window = np.concatenate([pad, hist], axis=0)
                windows.append(window)
                layout.append((series_idx, cutoff_idx))

        if not windows or self._head is None:
            return ForecastOutput(quantiles=[], quantile_levels=(0.5,))

        embs = self.solver.embed(windows)   # (N, D)
        preds = self._head.predict(embs)    # (N, H*C)

        per_series = [
            np.empty((n_cutoffs, H, C, 1), dtype=np.float32)
            for C, n_cutoffs in per_series_shape
        ]
        for i, (series_idx, cutoff_idx) in enumerate(layout):
            C = per_series_shape[series_idx][0]
            per_series[series_idx][cutoff_idx, :, :, 0] = preds[i].reshape(H, C)

        return ForecastOutput(quantiles=per_series, quantile_levels=(0.5,))


class _TimeEmbedEventAdapter(BaseTSFMAdapter):
    """Event detection via temporal embeddings + per-position logistic regression.

    The temporal embedding may be at a coarser stride than the original
    series; it is resampled back to the original length T via nearest-
    neighbour indexing before fitting and scoring.
    """

    def __init__(self, solver):
        self.solver = solver

    def _align(self, emb: np.ndarray, T: int) -> np.ndarray:
        """Resample emb (T', D) to length T via nearest-neighbour indices."""
        T_prime = emb.shape[0]
        if T_prime == T:
            return emb
        idx = np.round(np.linspace(0, T_prime - 1, T)).astype(int)
        return emb[idx]

    def fit(self, X_train, y_train):
        from sklearn.linear_model import LogisticRegression

        series_list = [np.asarray(s, dtype=np.float32) for s in X_train]
        time_embs = self.solver.time_embed(series_list)  # list of (T'_i, D)
        embs_all, labels_all = [], []
        for emb, labels in zip(time_embs, y_train):
            T = len(labels)
            embs_all.append(self._align(emb, T))
            labels_all.append(np.asarray(labels))
        X = np.concatenate(embs_all, axis=0)     # (sum T_i, D)
        y = np.concatenate(labels_all, axis=0)   # (sum T_i,)
        self._head = LogisticRegression(max_iter=1000).fit(X, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: (T, C) → scores: (T,) probabilities of the positive class
        x = np.asarray(x, dtype=np.float32)
        T = x.shape[0]
        emb = self.solver.time_embed([x])[0]                # (T', D)
        return self._head.predict_proba(self._align(emb, T))[:, 1]


class _WindowedEventAdapter(BaseTSFMAdapter):
    """Event detection via causal windowed embedding + per-position logistic regression.

    At each timestep, a window of size ``window_size`` ending at that
    position is embedded. Positions near the start of the series are zero-
    padded so that every timestep receives a score.
    """

    def __init__(self, solver, window_size: int):
        self.solver = solver
        self.window_size = window_size

    def _causal_windows(self, series: np.ndarray) -> list[np.ndarray]:
        """One zero-padded causal window per timestep."""
        T, C = series.shape
        padded = np.zeros((self.window_size - 1 + T, C), dtype=np.float32)
        padded[self.window_size - 1:] = series
        return [padded[t: t + self.window_size] for t in range(T)]

    def fit(self, X_train, y_train):
        from sklearn.linear_model import LogisticRegression

        all_windows, all_labels = [], []
        for series, labels in zip(X_train, y_train):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            for window, label in zip(self._causal_windows(series), labels):
                all_windows.append(window)
                all_labels.append(label)

        embs = self.solver.embed(all_windows)                       # (N, D)
        self._head = LogisticRegression(max_iter=1000).fit(embs, np.array(all_labels))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: (T, C) → scores: (T,) probabilities of the positive class
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        embs = self.solver.embed(self._causal_windows(x))   # (T, D)
        return self._head.predict_proba(embs)[:, 1]          # (T,)


# ---------------------------------------------------------------------------
# Base solver
# ---------------------------------------------------------------------------


class BaseTSFMSolver(BaseSolver):
    """Template solver for Time Series Foundation Models.

    It handles common boilerplate such as:
    - Device management (CUDA vs CPU, dtype selection)
    - Model loading and caching in set_objective (untimed)
    - Adapter setup based on task type
    - Metadata and data storage
    - Forecast batching for multi-series, multi-cutoff predictions

    Subclasses only need to implement:
    - supported_tasks: set of task names the model supports
    - model_id: unique string identifying the current model variant
    - load_model(): load/initialize the model for the given device
    - At least one of forecast_batch / embed_batch / time_embed_batch

    Attributes
    ----------
    supported_tasks
        Subset of TaskType that this solver supports. Must be set by subclass.

    task
        Current task being solved (set in set_objective).

    X_train, y_train : array-like
        Training data (set in set_objective).

    meta
        Task metadata like prediction_length, n_classes, etc.

    model
        The loaded TSFM model (cached by model_id across set_objective calls).

    device
        "cuda" or "cpu", automatically selected in set_objective.

    dtype
        The data type of both data and model.
        Default to bfloat16 on CUDA, float32 elsewhere.
    """

    supported_tasks: set[TaskType]
    task: TaskType

    X_train: Sequence[np.ndarray]
    y_train: Sequence[np.ndarray]
    meta: dict[str, Any]

    model: Any
    device: str | torch.device
    dtype: str | torch.dtype

    sampling_strategy = "run_once"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize solver with model-specific setup.

        Subclasses can override this method to perform model-specific
        initialization. If overriding, call super().__init__(**kwargs).
        """
        super().__init__()
        self._loaded_model_id = None
        self.model = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def supported_tasks(self) -> set[TaskType]:
        """Return a set of supported task names."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return a unique string identifying the current model variant.

        Used to cache the loaded model across set_objective calls. Should
        encode all parameters that affect model loading (e.g. model size,
        architecture variant). When model_id changes, load_model is called
        again.
        """

    @abstractmethod
    def load_model(self, device: str | torch.device, dtype: torch.dtype) -> Any:
        """Load and return the TSFM model.

        Called once per model_id (cached by base class). This method is
        called inside set_objective and is NOT timed.

        Parameters
        ----------
        device
            "cuda" or "cpu"
        dtype
            torch.bfloat16 or torch.float32

        Returns
        -------
        model : object
            The loaded TSFM model
        """

    def build_adapter(self, task: TaskType, model: Any) -> Any:
        """Create and optionally fit a task-specific adapter.

        Default strategies — the first capability the solver implements is used:

        forecasting
            1. forecast_batch  (zero-shot, via _SolverForecastAdapter)
            2. embed_batch     (windowed ridge regression, via _WindowedForecastAdapter)
        classification
            1. embed_batch     (flat embedding + LinearProbeAdapter)
            2. time_embed_batch (mean-pooled temporal embedding + LinearProbeAdapter)
        anomaly_detection
            1. embed_batch     (distance-from-mean score, via LinearProbeAdapter)
            2. forecast_batch  (forecast-error score, via ForecastResidualAdapter)
        event_detection
            1. time_embed_batch (per-position LogReg, via _TimeEmbedEventAdapter)
            2. embed_batch      (causal-windowed LogReg, via _WindowedEventAdapter)

        Override this method if the model requires custom adapter logic
        (e.g. specialised quantile computation, layer-specific embeddings).

        Parameters
        ----------
        task
            One of the supported tasks
        model
            The loaded TSFM model

        Returns
        -------
        adapter : BaseTSFMAdapter
            A fitted (or zero-shot) adapter ready for prediction
        """
        pred_len = self.meta.get("prediction_length", 1)
        window_size = getattr(self, "window_size", max(pred_len * 2, 64))

        if task == "forecasting":
            if self.can_forecast:
                return _SolverForecastAdapter(self)
            if self.can_embed:
                adapter = _WindowedForecastAdapter(self, window_size, pred_len)
                adapter.fit(self.X_train)
                return adapter
            raise NotImplementedError(
                f"{self.name} must implement forecast_batch or embed_batch "
                "for task='forecasting'"
            )

        if task == "classification":
            if self.can_embed:
                encoder = _SolverEmbedEncoder(self)
            elif self.can_time_embed:
                encoder = _SolverTimeEmbedPooledEncoder(self)
            else:
                raise NotImplementedError(
                    f"{self.name} must implement embed_batch or time_embed_batch "
                    "for task='classification'"
                )
            adapter = LinearProbeAdapter(
                encoder,
                task="classification",
                n_classes=self.meta.get("n_classes"),
                classifier=getattr(self, "classifier", "log_reg"),
                penalty=getattr(self, "penalty", "l2"),
                C=getattr(self, "C", 1.0),
                alpha=getattr(self, "alpha", 1.0),
                n_estimators=getattr(self, "n_estimators", 100),
            )
            adapter.fit(self.X_train, self.y_train)
            return adapter

        if task == "anomaly_detection":
            if self.can_embed:
                encoder = _SolverEmbedEncoder(self)
                adapter = LinearProbeAdapter(encoder, task="anomaly_detection")
                adapter.fit(self.X_train, self.y_train)
                return adapter
            if self.can_forecast:
                return ForecastResidualAdapter(_SolverForecastAdapter(self))
            raise NotImplementedError(
                f"{self.name} must implement embed_batch or forecast_batch "
                "for task='anomaly_detection'"
            )

        if task == "event_detection":
            if self.can_time_embed:
                adapter = _TimeEmbedEventAdapter(self)
                adapter.fit(self.X_train, self.y_train)
                return adapter
            if self.can_embed:
                adapter = _WindowedEventAdapter(self, window_size)
                adapter.fit(self.X_train, self.y_train)
                return adapter
            raise NotImplementedError(
                f"{self.name} must implement time_embed_batch or embed_batch "
                "for task='event_detection'"
            )

        raise NotImplementedError(f"Unknown task: {task!r}")

    def skip(self, task: TaskType, **_) -> tuple[bool, str | None]:
        """Skip unsupported tasks."""
        if task not in self.supported_tasks:
            return True, f"{self.name} solver does not support task={task!r}"
        return False, None

    def set_objective(
        self,
        X_train: Sequence[np.ndarray],
        y_train: Sequence[np.ndarray] | None,
        task: TaskType,
        **meta: Any,
    ) -> None:
        """Initialize solver for a task.

        Automatically handles device selection, dtype choice, and model
        loading/caching. Subclasses can override to add custom logic
        *after* calling super().set_objective(...).

        Parameters
        ----------
        X_train
            Training data
        y_train
            Training labels (may be None for unsupervised tasks)
        task
            Task name
        **meta
            Task metadata (prediction_length, n_classes, etc.)
        """
        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta

        # bfloat16 is well-supported on CUDA but poorly on CPU/MPS;
        # fall back to float32 elsewhere.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Load model, caching by model_id across set_objective calls.
        current_id = self.model_id
        if self._loaded_model_id != current_id:
            self.model = self.load_model(self.device, self.dtype)
            self._loaded_model_id = current_id

    def run(self, _: Any) -> None:
        """Build and fit task-specific adapter."""
        self._adapter = self.build_adapter(self.task, self.model)

    def get_result(self) -> dict[str, Any]:
        """Return the fitted adapter."""
        return {"model": self._adapter}

    @property
    def can_forecast(self):
        # A model is supposed to have forecast capabilities iff
        # it overrides the `forecast_batch` method
        return type(self).forecast_batch is not BaseTSFMSolver.forecast_batch

    def forecast_batch(
        self, inputs: list[torch.Tensor], covariates: Sequence[Covariates]
    ) -> list[torch.Tensor]:
        """Forecast on a batch of prepared inputs.

        Parameters
        ----------
        inputs
            Prepared input tensors of shape: (lookback, channel)
        covariates
            Corresponding covariates for each input

        Returns
        -------
        list of torch.Tensor
            Model output tensors of shape: (horizon, channel, quantiles)
        """
        raise NotImplementedError(
            "Subclasses must implement forecast_batch to call their model"
        )

    def forecast(
        self,
        x: ForecastInput,
        prediction_length: int,
        quantile_levels: tuple[float, ...],
    ) -> ForecastOutput:
        """Generic per-series, per-cutoff forecast batching.

        Handles input preparation, batching, and output reconstruction.
        Calls forecast_batch() for actual model inference.

        Parameters
        ----------
        x
            Input with series list and per-series cutoff indexes
        prediction_length
            Forecast horizon
        quantile_levels
            Quantile levels in outputs

        Returns
        -------
        ForecastOutput
            With per-series quantile arrays
        """
        inputs = []
        covariates = []
        layout = []  # (series_idx, cutoff_idx) per input
        per_series_shape = []  # (C, n_cutoffs) per series

        for series_idx, (series, cutoffs) in enumerate(zip(x.x, x.cutoff_indexes)):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            _, C = series.shape
            per_series_shape.append((C, len(cutoffs)))

            for cutoff_idx, cutoff in enumerate(cutoffs):
                hist = series[:cutoff]  # (T_cutoff, C)
                inputs.append(torch.from_numpy(hist))
                covariates.append(x.covariates.slice(cutoff, prediction_length))
                layout.append((series_idx, cutoff_idx))

        if not inputs:
            return ForecastOutput(quantiles=[], quantile_levels=quantile_levels)

        # TODO We still do this in batches in case data is very large

        # Get a list of model outputs aligned with inputs
        raw = self.forecast_batch(inputs, covariates)

        per_series_preds = [
            [None] * n_cutoffs for _, n_cutoffs in per_series_shape
        ]
        for (series_idx, cutoff_idx), pred in zip(layout, raw):
            per_series_preds[series_idx][cutoff_idx] = pred.float().cpu().numpy()

        per_series = [np.stack(preds) for preds in per_series_preds]

        return ForecastOutput(quantiles=per_series, quantile_levels=quantile_levels)

    @property
    def can_embed(self):
        # A model is supposed to have static embed capabilities iff
        # it overrides the `embed_batch` method
        return type(self).embed_batch is not BaseTSFMSolver.embed_batch

    def embed_batch(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Compute static embeddings on a batch of series.

        Parameters
        ----------
        inputs : list of torch.Tensor, shape (T, C)
            One tensor per series (full series, not windowed).

        Returns
        -------
        list of torch.Tensor, shape (D,)
            One flat embedding vector per input series.
        """
        raise NotImplementedError(
            "Subclasses must implement embed_batch to call their model"
        )

    def embed(self, x: list[np.ndarray]) -> np.ndarray:
        """Compute static embeddings for a list of series.

        Calls embed_batch() and stacks results into a 2-D array.

        Parameters
        ----------
        x : list of np.ndarray, shape (T_i, C)
            Input series (variable length).

        Returns
        -------
        np.ndarray, shape (N, D)
            One flat embedding per input series.
        """
        inputs = [
            torch.from_numpy(np.asarray(series, dtype=np.float32))
            for series in x
        ]
        results = self.embed_batch(inputs)
        return np.stack([r.float().cpu().numpy() for r in results], axis=0)

    @property
    def can_time_embed(self):
        # A model is supposed to have temporal embed capabilities iff
        # it overrides the `time_embed_batch` method
        return type(self).time_embed_batch is not BaseTSFMSolver.time_embed_batch

    def time_embed_batch(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Compute temporal embeddings on a batch of series.

        Parameters
        ----------
        inputs : list of torch.Tensor, shape (T, C)
            One tensor per series (full series, not windowed).

        Returns
        -------
        list of torch.Tensor, shape (T', D)
            One temporal embedding per input series; T' is model-determined
            (depends on stride and windowing).
        """
        raise NotImplementedError(
            "Subclasses must implement time_embed_batch to call their model"
        )

    def time_embed(self, x: list[np.ndarray]) -> list[np.ndarray]:
        """Compute temporal embeddings for a list of series.

        Calls time_embed_batch() and converts results to numpy.

        Parameters
        ----------
        x : list of np.ndarray, shape (T_i, C)
            Input series (variable length).

        Returns
        -------
        list of np.ndarray, shape (T'_i, D)
            Temporal embeddings; T'_i is model-determined per series.
        """
        inputs = [
            torch.from_numpy(np.asarray(series, dtype=np.float32))
            for series in x
        ]
        results = self.time_embed_batch(inputs)
        return [r.float().cpu().numpy() for r in results]
