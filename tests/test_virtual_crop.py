from pixel_pivr.virtual_crop import (
    ANSWER_SCHEMA,
    VIRTUAL_CROP_SCHEMA,
    point_centered_crop,
    transform_round2,
    transform_round2_records,
)


def source_row(target=(450, 450, 550, 550)):
    return {
        "conversations": [{"from": "human", "value": "old"}, {"from": "gpt", "value": "old"}],
        "image": "old.jpg",
        "meta": {
            "task": "detection",
            "record_id": "source-row-1",
            "pivr_reference": "small vehicle",
            "pivr_global_point": [500, 500],
            "pivr_global_target_box": None if target is None else list(target),
            "target_count": int(target is not None),
        },
    }


def test_point_crop_is_exact_and_clamped():
    assert point_centered_crop([0, 0], 1024, 768) == (0, 0, 144, 144)
    assert point_centered_crop([1000, 1000], 1024, 768) == (880, 624, 1024, 768)


def test_positive_local_record_is_two_view_box_only():
    row = transform_round2(source_row(), portable_image="images/train/a.jpg", width=1024, height=1024)
    assert row["conversations"][0]["value"].count("<image>") == 2
    assert row["conversations"][1]["value"].startswith("<box>")
    assert "<ref>" not in row["conversations"][1]["value"]
    assert row["meta"]["answer_format"] == ANSWER_SCHEMA
    assert row["meta"]["pivr_visual_context"] == "global_plus_pixel_crop_144to384_local_gate"
    assert row["image"][1]["schema"] == VIRTUAL_CROP_SCHEMA
    assert row["image"][1]["resize_hw"] == [384, 384]
    assert row["image"][1]["crop_xyxy"][2] - row["image"][1]["crop_xyxy"][0] == 144


def test_negative_local_record_uses_exact_none():
    rows = transform_round2_records(
        source_row(None), portable_image="images/train/a.jpg", width=1024, height=1024
    )
    assert len(rows) == 2
    assert all(row["conversations"][1]["value"] == "<box>None</box>" for row in rows)
    assert all(row["meta"]["target_count"] == 0 for row in rows)
    assert len(rows[0]["image"]) == 2
    assert rows[1]["image"] == "images/train/a.jpg"


def test_large_target_uses_explicit_global_fallback_without_clipping():
    target = [100, 100, 900, 900]
    rows = transform_round2_records(
        source_row(target), portable_image="images/train/a.jpg", width=1024, height=1024
    )
    assert len(rows) == 2
    local, fallback = rows
    assert local["conversations"][1]["value"] == "<box>None</box>"
    assert local["meta"]["pivr_fallback_required"] is True
    assert fallback["image"] == "images/train/a.jpg"
    assert fallback["meta"]["pivr_visual_context"] == "global_only_point_box_fallback"
    assert fallback["meta"]["pivr_global_target_box"] == target
    assert fallback["conversations"][1]["value"] == "<box><100><100><900><900></box>"
    assert fallback["conversations"][0]["value"].count("<image>") == 1


def test_subpixel_crop_overflow_uses_fallback_instead_of_clamping():
    # At 1024 pixels, x=429 maps to 439.296 while this point-centred crop starts
    # at x=440. The old one-pixel tolerance incorrectly accepted this target and
    # clamped its local x1 to zero.
    rows = transform_round2_records(
        source_row((429, 450, 550, 550)),
        portable_image="images/train/a.jpg",
        width=1024,
        height=1024,
    )
    assert len(rows) == 2
    assert rows[0]["conversations"][1]["value"] == "<box>None</box>"
    assert rows[0]["meta"]["pivr_containment_tolerance_pixels"] == 0.0
    assert rows[1]["conversations"][1]["value"] == (
        "<box><429><450><550><550></box>"
    )


def test_complete_crop_edge_target_keeps_local_box_and_pairs_global_fallback():
    # This target fits exactly but maps into the two-pixel inference safety
    # margin, so both the local prediction and observable global retry are taught.
    target = [430, 450, 550, 550]
    rows = transform_round2_records(
        source_row(target),
        portable_image="images/train/a.jpg",
        width=1024,
        height=1024,
    )
    assert len(rows) == 2
    local, fallback = rows
    assert local["conversations"][1]["value"] != "<box>None</box>"
    assert local["meta"]["pivr_target_fully_contained"] is True
    assert local["meta"]["pivr_target_touches_local_fallback_margin"] is True
    assert local["meta"]["pivr_fallback_reason"] == (
        "complete_target_touches_local_fallback_margin"
    )
    assert fallback["conversations"][1]["value"] == (
        "<box><430><450><550><550></box>"
    )
    assert fallback["meta"]["pivr_target_fully_contained"] is True
    assert fallback["meta"]["pivr_target_touches_local_fallback_margin"] is True


def test_small_source_image_is_rejected():
    try:
        transform_round2(source_row(), portable_image="images/train/a.jpg", width=128, height=1024)
    except ValueError as exc:
        assert "smaller than" in str(exc)
    else:
        raise AssertionError("Expected fixed 144-pixel crop guard")


def test_superseded_cached_roi_metadata_is_removed():
    row = source_row()
    row["meta"]["pivr_local_context_source"] = (
        "single_encode_preprojector_magnified_roi"
    )
    row["meta"]["pivr_local_roi_may_not_cover_full_target"] = True

    outputs = transform_round2_records(
        row,
        portable_image="images/train/a.jpg",
        width=1024,
        height=1024,
    )

    for output in outputs:
        assert "pivr_local_context_source" not in output["meta"]
        assert "pivr_local_roi_may_not_cover_full_target" not in output["meta"]
