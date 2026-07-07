"""Structural interfaces (typing.Protocol) for the pluggable collaborators of
the localization pipeline.

These formalize the duck-typed contracts that already exist between
``Localizer`` and its collaborators. They are Protocols, not ABCs: existing
implementations need ZERO changes — mypy verifies conformance structurally, and
``@runtime_checkable`` enables ``isinstance()`` in tests.

Concrete implementers already in the codebase:
    * Retriever                 -> localization.matcher.FastRetrieval / LanceDBRetrieval
    * GlobalDescriptorExtractor -> models.wrappers.feature_extractor.FeatureExtractor
    * LocalFeatureExtractor     -> models.wrappers.feature_extractor.FeatureExtractor
    * FrameDatabase             -> database.database_loader.DatabaseLoader

Dynamic-object masking already has an ABC (models.wrappers.masking_strategy.
MaskingStrategy), so it is intentionally not duplicated here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from numpy.typing import NDArray


@runtime_checkable
class Retriever(Protocol):
    """Top-k nearest-neighbour retrieval over global descriptors."""

    def find_similar_frames(
        self, query_desc: NDArray[Any], top_k: int = 5
    ) -> list[tuple[int, float]]: ...

    def add_descriptor(self, query_desc: NDArray[Any], frame_id: int) -> None: ...


@runtime_checkable
class GlobalDescriptorExtractor(Protocol):
    """Whole-image global descriptor(s) used for retrieval."""

    def extract_global_descriptor(self, image: NDArray[Any]) -> NDArray[Any]: ...

    def extract_global_descriptors_multi(
        self, images: list[NDArray[Any]]
    ) -> NDArray[Any]: ...


@runtime_checkable
class LocalFeatureExtractor(Protocol):
    """Local keypoints/descriptors used for geometric verification."""

    def extract_local_features(
        self, image: NDArray[Any], static_mask: NDArray[Any] | None = None
    ) -> dict[str, Any]: ...


@runtime_checkable
class FrameDatabase(Protocol):
    """Read access to a built keyframe database."""

    def get_local_features(self, frame_id: int) -> dict[str, NDArray[Any]]: ...

    def get_frame_affine(self, frame_id: int) -> NDArray[Any] | None: ...

    def get_frame_size(self, frame_id: int) -> tuple[int, int]: ...

    def get_num_frames(self) -> int: ...
