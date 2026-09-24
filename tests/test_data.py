from PIL import Image

from skinshift.data import center_crop_fraction, check_no_lesion_overlap, lesion_split


def test_center_crop_fraction_keeps_correct_proportion_of_area():
    img = Image.new("RGB", (100, 200))
    cropped = center_crop_fraction(img, 0.5)
    assert cropped.size == (50, 100)


def test_center_crop_fraction_is_centered_not_shifted():
    # a white square on a black background, centered; cropping to the middle 50% by area
    # (~70.7% per side) should keep it entirely white if the crop is actually centered
    img = Image.new("RGB", (100, 100), "black")
    for x in range(25, 75):
        for y in range(25, 75):
            img.putpixel((x, y), (255, 255, 255))
    cropped = center_crop_fraction(img, 0.5)
    assert cropped.getextrema() == ((255, 255), (255, 255), (255, 255))


def test_center_crop_fraction_one_is_a_no_op_on_size():
    img = Image.new("RGB", (64, 48))
    assert center_crop_fraction(img, 1.0).size == (64, 48)


def test_center_crop_fraction_rejects_bad_input():
    import pytest
    img = Image.new("RGB", (10, 10))
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            center_crop_fraction(img, bad)


def _rows(n, lesions_per_class=3):
    """Synthetic rows: several images per lesion, several lesions, so a naive image-level
    split would put images of the same lesion in different splits."""
    rows = []
    for lesion in range(n):
        for img in range(lesions_per_class):
            rows.append({"isic_id": f"L{lesion}_{img}", "lesion_id": f"L{lesion}",
                         "class_short": "nevus", "path": f"nevus/L{lesion}_{img}.jpg"})
    return rows


def test_lesion_split_has_no_overlap_across_many_seeds():
    rows = _rows(40)
    for seed in range(10):
        train, val, test = lesion_split(rows, seed=seed)
        assert len(train) + len(val) + len(test) == len(rows)
        check_no_lesion_overlap(train, val, test)  # raises on any violation


def test_lesion_split_keeps_all_images_of_one_lesion_together():
    rows = _rows(40)
    train, val, test = lesion_split(rows, seed=0)
    by_split = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        for r in split:
            by_split.setdefault(r["lesion_id"], set()).add(name)
    assert all(len(s) == 1 for s in by_split.values())


def test_missing_lesion_id_falls_back_to_isic_id():
    rows = [{"isic_id": f"IMG{i}", "lesion_id": "", "class_short": "nevus", "path": f"nevus/IMG{i}.jpg"}
            for i in range(20)]
    train, val, test = lesion_split(rows, seed=0)
    check_no_lesion_overlap(train, val, test)
    assert len(train) + len(val) + len(test) == 20
