import numpy as np
import pytest

from transcend import calibration


def test_concatenate_calibration_set_results():
    results_list = [
        {
            "cred_p_val_cal_fold": [0.5, 0.6],
            "conf_p_val_cal_fold": [0.7, 0.8],
            "pred_cal_fold": [0, 1],
            "groundtruth_cal_fold": [0, 1],
        },
        {
            "cred_p_val_cal_fold": [0.3, 0.4],
            "conf_p_val_cal_fold": [0.9, 0.85],
            "pred_cal_fold": [0, 0],
            "groundtruth_cal_fold": [0, 0],
        },
    ]

    concatenated = calibration.concatenate_calibration_set_results(results_list)
    assert len(concatenated["cred_p_val_cal"]) == 4
    assert len(concatenated["conf_p_val_cal"]) == 4
    assert len(concatenated["pred_cal"]) == 4
    assert len(concatenated["groundtruth_cal"]) == 4


def test_train_calibration_set(tmp_path):
    X = np.array([
        [-2.0, -2.0],
        [-1.5, -1.0],
        [-1.0, -1.5],
        [-2.5, -2.5],
        [1.0, 1.5],
        [1.5, 1.0],
        [2.0, 2.0],
        [2.5, 2.5],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    results = calibration.train_calibration_set(
        X, y, n_folds=2, ncpu=1, saved_data_folder=str(tmp_path)
    )
    assert len(results) == 2
    concatenated = calibration.concatenate_calibration_set_results(results)
    assert len(concatenated["cred_p_val_cal"]) == len(y)
    assert len(concatenated["conf_p_val_cal"]) == len(y)

