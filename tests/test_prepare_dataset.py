from backend.data.prepare_dataset import normalize_bbox


def test_bbox_normalization_range():
    bbox = [50, 100, 350, 500]
    width, height = 1000, 1000

    normalized = normalize_bbox(bbox, width, height)

    for value in normalized:
        assert 0 <= value <= 1000
