import numpy as np

from src.utils.image_preprocessor import ImagePreprocessor


def test_image_preprocessor_init():
    # Default init
    preprocessor = ImagePreprocessor()
    assert preprocessor.clahe is not None
    assert preprocessor.clahe.getClipLimit() == 3.0
    assert preprocessor.clahe.getTilesGridSize() == (8, 8)

    # Custom config init
    config = {"preprocessing": {"clahe_clip_limit": 4.0, "clahe_tile_grid": [4, 4]}}
    preprocessor2 = ImagePreprocessor(config)
    assert preprocessor2.clahe.getClipLimit() == 4.0
    assert preprocessor2.clahe.getTilesGridSize() == (4, 4)


def test_image_preprocessor_apply():
    preprocessor = ImagePreprocessor()

    # Create a simple dark RGB image with some variation
    img = np.ones((100, 100, 3), dtype=np.uint8) * 50
    img[25:75, 25:75] = 100  # Add a central lighter square

    # Apply preprocessing
    enhanced = preprocessor.preprocess(img)

    assert enhanced is not None
    assert enhanced.shape == (100, 100, 3)
    # Since CLAHE distributes contrast, the values should change.
    assert not np.array_equal(img, enhanced)


def test_image_preprocessor_empty():
    preprocessor = ImagePreprocessor()
    img = np.array([])
    enhanced = preprocessor.preprocess(img)
    assert enhanced.size == 0

    enhanced_none = preprocessor.preprocess(None)
    assert enhanced_none is None
