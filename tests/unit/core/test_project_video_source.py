import pytest

from src.core.project_video_source import ProjectVideoSource, ScaleLayerMetadata


def _source(**kwargs):
    values = {
        "source_id": "map_1000",
        "area_id": "test_area",
        "video_path": "reference/video.mp4",
        "database_file": "reference/database.h5",
        "calibration_file": "reference/calibration.json",
    }
    values.update(kwargs)
    return ProjectVideoSource(**values)


def test_scale_layer_round_trip_preserves_versioned_and_unknown_metadata():
    source = _source(
        scale_layer={
            "version": 1,
            "layer_id": "gsd_076",
            "nominal_gsd_m_per_px": 0.762,
            "min_gsd_m_per_px": 0.70,
            "max_gsd_m_per_px": 0.82,
            "scale_quality": "verified",
            "neighbor_layer_ids": ["gsd_038"],
            "descriptor_schema_fingerprint": "fec19ea143386a2d",
            "producer_note": "kept for forward compatibility",
        }
    )

    restored = ProjectVideoSource.from_dict(source.to_dict())

    assert isinstance(restored.scale_layer, ScaleLayerMetadata)
    assert restored.scale_layer.nominal_gsd_m_per_px == pytest.approx(0.762)
    assert restored.scale_layer.neighbor_layer_ids == ("gsd_038",)
    assert restored.to_dict()["scale_layer"]["producer_note"] == (
        "kept for forward compatibility"
    )


def test_unknown_scale_is_valid_and_does_not_invent_gsd():
    layer = ScaleLayerMetadata(layer_id="legacy_unknown")

    assert layer.nominal_gsd_m_per_px is None
    assert layer.relative_scale is None
    assert layer.scale_quality == "unknown"


@pytest.mark.parametrize(
    "metadata",
    [
        {"version": 2, "layer_id": "future"},
        {"version": 1, "layer_id": ""},
        {"version": 1, "layer_id": "bad", "nominal_gsd_m_per_px": 0},
        {
            "version": 1,
            "layer_id": "bad",
            "min_gsd_m_per_px": 0.8,
            "max_gsd_m_per_px": 0.4,
        },
        {"version": 1, "layer_id": "self", "neighbor_layer_ids": ["self"]},
    ],
)
def test_invalid_scale_layer_contract_is_rejected(metadata):
    with pytest.raises(ValueError):
        _source(scale_layer=metadata)
