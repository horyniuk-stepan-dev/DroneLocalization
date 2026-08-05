"""The domain exception hierarchy: subclasses are catchable via the base."""

import pytest

from src.exceptions import (
    CalibrationError,
    DatabaseFormatError,
    DroneLocError,
    ModelLoadError,
    VideoDecodeError,
)

_SUBCLASSES = [ModelLoadError, DatabaseFormatError, CalibrationError, VideoDecodeError]


@pytest.mark.parametrize("exc", _SUBCLASSES)
def test_subclass_of_base_and_exception(exc):
    assert issubclass(exc, DroneLocError)
    assert issubclass(exc, Exception)


@pytest.mark.parametrize("exc", _SUBCLASSES)
def test_base_catches_subclass(exc):
    with pytest.raises(DroneLocError):
        raise exc("boom")


def test_message_preserved():
    err = ModelLoadError("dinov2 weights missing")
    assert str(err) == "dinov2 weights missing"
