"""Deprecated: import from ``config`` instead of ``config.config``.

Kept for one release as a backward-compatibility shim. All public names are
re-exported from the :mod:`config` package.
"""

import warnings

warnings.warn(
    "config.config is deprecated; import from `config` instead "
    "(e.g. `from config import get_cfg`).",
    DeprecationWarning,
    stacklevel=2,
)

from config import *  # noqa: F401, F403
