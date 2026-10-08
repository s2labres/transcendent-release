import numpy as np
import pytest

from transcend import thresholding


def test_apply_threshold():
    thresholds = {
        "cred": {"mw": 0.4, "gw": 0.6},
        "conf": {"mw": 0.5, "gw": 0.8},
    }
    scores = {
        "cred": [0.4, 0.2, 0.7, 0.8, 0.6],
        "conf": [0.6, 0.8, 0.3, 0.2, 0.4],
    }
    y = np.array([1, 1, 1, 0, 0])
    keep_mask = thresholding.apply_threshold(thresholds, scores, y)
    assert isinstance(keep_mask, np.ndarray)
    assert keep_mask.dtype == bool
    assert len(keep_mask) == len(y)


def test_sort_by_predicted_label():
    s = np.array([0.8, 0.7, 0.6, 0.9])
    y_pred = np.array([1, 1, 0, 0])
    y_true = np.array([1, 0, 1, 0])

    s_mw, s_gw = thresholding.sort_by_predicted_label(s, y_pred, y_true, "correct")
    assert np.allclose(s_mw, [0.8])
    assert np.allclose(s_gw, [0.9])

    s_mw, s_gw = thresholding.sort_by_predicted_label(s, y_pred, y_true, "incorrect")
    assert np.allclose(s_mw, [0.7])
    assert np.allclose(s_gw, [0.6])

    s_mw, s_gw = thresholding.sort_by_predicted_label(s, y_pred, y_true, "all")
    assert np.allclose(s_mw, [0.8, 0.7])
    assert np.allclose(s_gw, [0.6, 0.9])


def test_find_quartile_thresholds():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    thresholds = thresholding.find_quartile_thresholds(
        scores, y_pred, y_true, consider="correct"
    )
    for key in ("q1", "q2", "q3", "mean"):
        assert key in thresholds
        assert "mw" in thresholds[key]
        assert "gw" in thresholds[key]


def test_get_performance_with_rejection_normal():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    keep_mask = np.array([True, True, False, True])

    perf = thresholding.get_performance_with_rejection(y_true, y_pred, keep_mask, full=True)
    assert "f1_b" in perf
    assert "f1_k" in perf
    assert "f1_r" in perf
    assert "kept_total_perc" in perf
    assert "reject_total_perc" in perf


def test_get_performance_with_rejection_empty_rejection():
    # Keep mask is all True (no rejected elements)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    keep_mask = np.array([True, True, True, True])

    perf = thresholding.get_performance_with_rejection(y_true, y_pred, keep_mask, full=True)
    assert perf["f1_k"] == 1.0
    assert perf["f1_r"] == 0.0
    assert perf["reject_total_perc"] == 0.0


def test_test_with_rejection():
    thresholds = {"cred": {"mw": 0.5, "gw": 0.5}}
    scores = {"cred": [0.6, 0.3, 0.8, 0.4]}
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    results, keep_mask = thresholding.test_with_rejection(
        thresholds, scores, y_true, y_pred, full=True
    )
    assert isinstance(results, dict)
    assert isinstance(keep_mask, np.ndarray)


def test_report_results():
    results = {
        "kept_total_perc": 0.75,
        "reject_total_perc": 0.25,
        "kept_neg_perc": 0.8,
        "reject_neg_perc": 0.2,
        "kept_pos_perc": 0.7,
        "reject_pos_perc": 0.3,
        "f1_b": 0.85,
        "f1_k": 0.90,
        "f1_r": 0.60,
        "precision_b": 0.85,
        "precision_k": 0.90,
        "precision_r": 0.60,
        "recall_b": 0.85,
        "recall_k": 0.90,
        "recall_r": 0.60,
        "tp_b": 10,
        "tp_k": 7,
        "tp_r": 3,
        "fp_b": 2,
        "fp_k": 1,
        "fp_r": 1,
        "tn_b": 10,
        "tn_k": 8,
        "tn_r": 2,
        "fn_b": 2,
        "fn_k": 1,
        "fn_r": 1,
        "tpr_b": 0.83,
        "tpr_k": 0.87,
        "tpr_r": 0.75,
        "fpr_b": 0.17,
        "fpr_k": 0.11,
        "fpr_r": 0.33,
    }
    # Pass as dict
    report = thresholding.report_results(results, quiet=True)
    assert "F1 baseline" in report

    # Pass as tuple (results, keep_mask) - backward compat guard
    report_tuple = thresholding.report_results((results, np.array([True])), quiet=True)
    assert "F1 baseline" in report_tuple
