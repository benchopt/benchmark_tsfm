"""Abstract base solver for Time Series Foundation Models (TSFM)."""

from abc import abstractmethod
from typing import Any, Literal, Sequence

import numpy as np
import torch
from benchopt import BaseSolver

from benchmark_utils.covariates import Covariates
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput

TaskType = Literal["forecasting", "classification", "anomaly_detection"]


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
    - load_model(): load/initialize the model for the given device
    - build_adapter(): create task-specific adapters

    Attributes
    ----------
    supported_tasks
        Subset of {"forecasting", "classification", "anomaly_detection"}
        that this solver supports. Must be set by subclass.

    task
        Current task being solved (set in set_objective).

    X_train, y_train : array-like
        Training data (set in set_objective).

    meta
        Task metadata like prediction_length, n_classes, etc.

    model
        The loaded TSFM model (cached across multiple set_objective calls).

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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize solver with model-specific setup.

        Subclasses can override this method to perform model-specific
        initialization. If overriding, call super().__init__(**kwargs).
        """
        super().__init__()

        # Initialize cached model state
        self._loaded_model = None
        self.model = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def supported_tasks(self) -> set[TaskType]:
        """Return a set of supported task names.

        Returns
        -------
        set of str
            Subset of {"forecasting", "classification", "anomaly_detection"}
        """

    @abstractmethod
    def load_model(self, device: str | torch.device, dtype: torch.dtype) -> Any:
        """Load and return the TSFM model.

        Called once per model variant (cached by subclass if needed).
        This method is called inside set_objective and is NOT timed.

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

    @abstractmethod
    def build_adapter(self, task: TaskType, model: Any) -> Any:
        """Create and optionally fit a task-specific adapter.

        Called from run() for the current task and model.
        If the adapter requires fitting (e.g., LinearProbe), do it here.

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

    def skip(self, task: str, **_) -> tuple[bool, str | None]:
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

        # Select device and dtype
        # bfloat16 is well-supported on CUDA but poorly on CPU/MPS;
        # fall back to float32 elsewhere.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Load model (caching is subclass responsibility via load_model)
        self.model = self.load_model(self.device, self.dtype)

    def run(self, _: Any) -> None:
        """Build and fit task-specific adapter.

        Calls build_adapter() to create the adapter for the current task.
        Subclasses can override to add custom logic, but should typically
        just call super().run(_) to set up self._adapter.
        """
        self._adapter = self.build_adapter(self.task, self.model)

    def get_result(self) -> dict[str, Any]:
        """Return the fitted adapter."""
        return {"model": self._adapter}

    def forecast_batch(
        self, inputs: list[torch.Tensor], covariates: Sequence[Covariates]
    ) -> list[torch.Tensor]:
        """Forecast on a batch of prepared inputs.

        Subclasses must implement this to call their model's inference.

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

        per_series_preds: list[list] = [
            [None] * n_cutoffs for _, n_cutoffs in per_series_shape
        ]
        for (series_idx, cutoff_idx), pred in zip(layout, raw):
            per_series_preds[series_idx][cutoff_idx] = pred.float().cpu().numpy()

        per_series = [np.stack(preds) for preds in per_series_preds]

        return ForecastOutput(quantiles=per_series, quantile_levels=quantile_levels)
