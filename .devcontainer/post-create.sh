#!/usr/bin/env bash
set -euo pipefail

# VS Code runs onCreateCommand as `sh -c bash post-create.sh`, a
# non-login, non-interactive shell that never sources the conda init
# hooks in ~/.bashrc. Without CONDA_PREFIX set, `benchopt install`
# crashes outright (list_conda_envs() calls Path(None)). Activate
# explicitly so this works regardless of how the script is invoked.
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# pip install benchopt
pip install git+https://github.com/benchopt/benchopt.git

# Install AI agent skills
benchopt sync-skills

# `benchopt install` can report freshly-installed packages as missing on its
# first pass: it checks importability in the same long-lived process that
# ran the installs, and Python's import machinery caches negative lookups
# from before those installs landed on disk. A second (fresh-process)
# invocation reliably sees everything that's actually on disk.
benchopt install -y || benchopt install -y

# Installing `aeon` (Monash dataset requirement) pulls in numpy>=2, which
# is ABI-incompatible with this image's conda-shipped scipy/pandas/
# matplotlib/pyarrow/numexpr/bottleneck (built against numpy 1.x). Realign
# them with pip wheels built for whatever numpy version ends up installed.
pip install --upgrade --force-reinstall --no-deps \
    scipy pandas matplotlib pyarrow numexpr bottleneck

pip install pre-commit
pre-commit install
