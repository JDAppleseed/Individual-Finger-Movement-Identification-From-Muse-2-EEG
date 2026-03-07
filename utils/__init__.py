"""
Utils package initializer.

This project previously used filenames prefixed with `utils.` inside the
`utils/` directory (e.g. `utils.per_subject_calibration.py`). Those files
have been renamed to conventional module names (e.g. `per_subject_calibration.py`).
Importing submodules directly (for example `from utils.per_subject_calibration import ...`)
now works as usual.
"""

import importlib

__all__ = [
    "experiment_logger",
    "per_subject_calibration",
    "report_generator",
    "sequence_data",
    "timebase",
    "eval_utils",
    "stream_timebase",
]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
