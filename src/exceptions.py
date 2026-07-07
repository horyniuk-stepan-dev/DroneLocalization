"""Domain exception hierarchy for DroneLocalization.

A single base (:class:`DroneLocError`) lets callers catch any project-specific
error, while the subclasses let optional-feature and required-path code narrow
to the exact failure instead of a blanket ``except Exception``.
"""


class DroneLocError(Exception):
    """Base class for all DroneLocalization domain errors."""


class ModelLoadError(DroneLocError):
    """A model (weights, TensorRT/TorchScript engine, HF repo) failed to load or init."""


class DatabaseFormatError(DroneLocError):
    """A features database is missing, corrupt, or has an unexpected/old schema."""


class CalibrationError(DroneLocError):
    """Calibration or propagation could not be computed (no anchors, singular graph, ...)."""


class VideoDecodeError(DroneLocError):
    """A video source could not be opened or a frame could not be decoded."""
