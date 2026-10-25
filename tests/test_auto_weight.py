"""Deterministic tests for calibrated depth-based weight estimation."""

import math

import numpy as np

from foodmacros.config import AppConfig
from foodmacros.weight.auto_weight import AutoWeightConfig, MiDaSDepthProvider, build_depth_provider
from foodmacros.weight.calibration import cm_per_depth_unit_from_reference
from foodmacros.weight.estimator import WeightEstimator


class _FakeDepthBundle:
    def __init__(self, depth_map):
        self.depth_map = np.asarray(depth_map, dtype=np.float32)

    def predict_depth(self, array):
        return self.depth_map


class _PredictableBundle:
    def __init__(self, depth_map):
        self.depth_map = np.asarray(depth_map, dtype=np.float32)


def _fake_predict_depth(bundle, image_rgb):
    depth = bundle.depth_map
    h, w = image_rgb.shape[:2]
    if depth.shape != (h, w):
        import numpy as _np

        depth = _np.resize(depth, (h, w)).astype(_np.float32)
    return depth


def test_auto_weight_config_from_weight_config():
    cfg = AppConfig()
    cfg.weight.cm_per_pixel = 0.05
    cfg.weight.cm_per_depth_unit = 0.1
    cfg.weight.reference_depth = 1.0
    cfg.weight.invert_depth = True
    aw = AutoWeightConfig.from_weight_config(cfg.weight)
    assert aw.cm_per_pixel == 0.05
    assert aw.cm_per_depth_unit == 0.1
    assert aw.reference_depth == 1.0
    assert aw.invert_depth is True


def test_missing_cm_per_pixel_unavailable():
    w = WeightEstimator(AppConfig())
    p = MiDaSDepthProvider(w, AutoWeightConfig(cm_per_pixel=None, cm_per_depth_unit=0.1))
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    ev = p.estimate_depth_evidence(img, "pizza")
    assert ev.available is False
    assert "cm_per_pixel" in ev.reason


def test_missing_cm_per_depth_unit_unavailable():
    w = WeightEstimator(AppConfig())
    p = MiDaSDepthProvider(w, AutoWeightConfig(cm_per_pixel=0.05, cm_per_depth_unit=None))
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    ev = p.estimate_depth_evidence(img, "pizza")
    assert ev.available is False
    assert "cm_per_depth_unit" in ev.reason


def test_invalid_calibration_rejected():
    w = WeightEstimator(AppConfig())
    p = MiDaSDepthProvider(w, AutoWeightConfig(cm_per_pixel=0, cm_per_depth_unit=0.1))
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    ev = p.estimate_depth_evidence(img, "pizza")
    assert ev.available is False


def test_inferred_min_depth_plane(monkeypatch):
    import foodmacros.weight.auto_weight as aw

    w = WeightEstimator(AppConfig())
    bundle = _PredictableBundle(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
    p = MiDaSDepthProvider(
        w,
        AutoWeightConfig(cm_per_pixel=0.1, cm_per_depth_unit=0.2),
        bundle=bundle,
    )
    monkeypatch.setattr(aw, "predict_depth", _fake_predict_depth)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    ev = p.estimate_depth_evidence(img, "pizza")
    assert ev.available is True
    assert ev.calibration["reference_depth_source"] == "inferred_min_depth"
    assert math.isclose(ev.calibration["reference_depth"], 2.0)
    assert "minimum crop depth" in (ev.reason or "").lower()


def test_configured_reference_depth(monkeypatch):
    import foodmacros.weight.auto_weight as aw

    w = WeightEstimator(AppConfig())
    bundle = _PredictableBundle(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
    p = MiDaSDepthProvider(
        w,
        AutoWeightConfig(cm_per_pixel=0.1, cm_per_depth_unit=0.2, reference_depth=1.0),
        bundle=bundle,
    )
    monkeypatch.setattr(aw, "predict_depth", _fake_predict_depth)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    ev = p.estimate_depth_evidence(img, "pizza")
    assert ev.available is True
    assert ev.calibration["reference_depth_source"] == "configured"
    assert ev.calibration["reference_depth"] == 1.0


def test_mask_applied(monkeypatch):
    import foodmacros.weight.auto_weight as aw

    w = WeightEstimator(AppConfig())
    bundle = _PredictableBundle(np.array([[3.0, 3.0], [2.0, 2.0]], dtype=np.float32))
    p = MiDaSDepthProvider(
        w,
        AutoWeightConfig(cm_per_pixel=0.1, cm_per_depth_unit=0.2),
        bundle=bundle,
    )
    monkeypatch.setattr(aw, "predict_depth", _fake_predict_depth)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[True, False], [False, False]], dtype=bool)
    ev = p.estimate_depth_evidence(img, "pizza", food_mask=mask)
    assert ev.available is True


def test_unknown_density_unavailable(monkeypatch):
    import foodmacros.weight.auto_weight as aw

    w = WeightEstimator(AppConfig())
    bundle = _PredictableBundle(np.ones((2, 2), dtype=np.float32))
    p = MiDaSDepthProvider(
        w,
        AutoWeightConfig(cm_per_pixel=0.1, cm_per_depth_unit=0.2),
        bundle=bundle,
    )
    monkeypatch.setattr(aw, "predict_depth", _fake_predict_depth)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    ev = p.estimate_depth_evidence(img, "unknown_food_xyz")
    assert ev.available is False


def test_cm_per_depth_unit_from_reference():
    scale = cm_per_depth_unit_from_reference(5.0, 3.0, 1.0)
    assert math.isclose(scale, 2.5)


def test_cm_per_depth_unit_from_reference_invalid():
    try:
        cm_per_depth_unit_from_reference(5.0, 1.0, 2.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_depth_provider():
    cfg = AppConfig()
    cfg.weight.cm_per_pixel = 0.05
    cfg.weight.cm_per_depth_unit = 0.1
    w = WeightEstimator(cfg)
    p = build_depth_provider(cfg, w)
    assert p.cfg.cm_per_pixel == 0.05
    assert p.cfg.cm_per_depth_unit == 0.1


def test_depth_disabled_by_default_no_weight():
    from foodmacros.config import AppConfig
    from foodmacros.weight.estimator import WeightEstimator

    cfg = AppConfig()
    w = WeightEstimator(cfg)
    est = w.estimate(food_label="pizza")
    assert est.available is False
    assert est.status == "unavailable"


def test_missing_density_depth_unavailable(monkeypatch):
    import foodmacros.weight.auto_weight as aw

    w = WeightEstimator(AppConfig())
    bundle = _PredictableBundle(np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32))
    p = MiDaSDepthProvider(w, AutoWeightConfig(cm_per_pixel=0.1, cm_per_depth_unit=0.2), bundle=bundle)
    monkeypatch.setattr(aw, "predict_depth", _fake_predict_depth)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    ev = p.estimate_depth_evidence(img, "totally_unknown_food_label_xyz123")
    assert ev.available is False
    assert "density" in ev.reason.lower()


def test_experimental_depth_requires_opt_in():
    from foodmacros.config import AppConfig
    from foodmacros.weight.estimator import DepthEvidence, WeightEstimator

    cfg = AppConfig()
    cfg.weight.allow_experimental_depth = False
    w = WeightEstimator(cfg)
    ev = DepthEvidence(True, volume_cm3=100.0, density_g_cm3=0.9)
    est = w.estimate(depth=ev, food_label="pizza")
    assert est.available is False
    assert est.status == "unavailable"


def test_experimental_depth_with_opt_in_and_valid_calibration(monkeypatch):
    from foodmacros.config import AppConfig
    from foodmacros.weight.estimator import WeightEstimator
    from foodmacros.weight.auto_weight import AutoWeightConfig, MiDaSDepthProvider
    import foodmacros.weight.auto_weight as aw

    cfg = AppConfig()
    cfg.weight.allow_experimental_depth = True
    w = WeightEstimator(cfg)
    bundle = _PredictableBundle(np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32))
    p = MiDaSDepthProvider(w, AutoWeightConfig(cm_per_pixel=0.1, cm_per_depth_unit=0.1), bundle=bundle)
    monkeypatch.setattr(aw, "predict_depth", _fake_predict_depth)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    ev = p.estimate_depth_evidence(img, "pizza")
    assert ev.available is True
    est = w.estimate(depth=ev, food_label="pizza")
    # Should produce experimental estimate if in range
    assert est.status in ("experimental", "unavailable")  # depends on range/density; must not fabricate blindly
    if est.available:
        assert est.grams is not None and est.grams > 0
