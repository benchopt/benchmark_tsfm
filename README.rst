Benchmark for Time Series Foundation Models (TSFM)
==================================================
|Build Status| |Python 3.10+|

This benchmark evaluates the performance of various time series foundation models (TSFM) on a range of datasets and tasks. It is built using the `benchopt` framework, which provides a standardized way to compare different solvers and algorithms on a common set of problems.

The goal is to provide a benchmark that evaluate the models on:

- Classification
- Forecasting
- Anomaly Detection
- Event Detection

With diverse modalities (univariate, multivariate, EEG, etc.) and varying sequence lengths.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_tsfm
   $ benchopt run benchmark_tsfm

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_tsfm -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_tsfm/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_tsfm/actions
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-310/
