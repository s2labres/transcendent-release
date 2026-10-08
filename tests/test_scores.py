import numpy as np
import pytest
from scipy import sparse
from sklearn.svm import SVC

from transcend import scores


@pytest.fixture
def trained_svm():
    # Simple linearly separable 2D data
    X = np.array([
        [-2.0, -2.0],
        [-1.5, -1.0],
        [-1.0, -1.5],
        [1.0, 1.5],
        [1.5, 1.0],
        [2.0, 2.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y)
    return clf, X, y


def test_get_single_svm_ncm_dense_1d(trained_svm):
    clf, X, y = trained_svm
    # 1D slice (shape (2,)) - previously caused ValueError in scikit-learn >= 1.0
    ncm_0 = scores.get_single_svm_ncm(clf, X[0], y[0])
    ncm_1 = scores.get_single_svm_ncm(clf, X[-1], y[-1])

    assert isinstance(ncm_0, float)
    assert isinstance(ncm_1, float)


def test_get_single_svm_ncm_dense_2d(trained_svm):
    clf, X, y = trained_svm
    ncm_0 = scores.get_single_svm_ncm(clf, X[0:1], y[0])
    assert isinstance(ncm_0, float)


def test_get_single_svm_ncm_sparse(trained_svm):
    clf, X, y = trained_svm
    X_sparse = sparse.csr_matrix(X)
    ncm_sparse = scores.get_single_svm_ncm(clf, X_sparse[0], y[0])
    assert isinstance(ncm_sparse, float)


def test_get_svm_ncms(trained_svm):
    clf, X, y = trained_svm
    ncms = scores.get_svm_ncms(clf, X, y)
    assert len(ncms) == len(y)
    assert all(isinstance(val, float) for val in ncms)


def test_compute_p_values(trained_svm):
    clf, X, y = trained_svm
    ncms = scores.get_svm_ncms(clf, X, y)

    res = scores.compute_p_values_cred_and_conf(
        train_ncms=ncms,
        groundtruth_train=y,
        test_ncms=ncms,
        y_test=y,
    )

    assert "cred" in res
    assert "conf" in res
    assert len(res["cred"]) == len(y)
    assert len(res["conf"]) == len(y)

    for p in res["cred"] + res["conf"]:
        assert 0.0 <= p <= 1.0


def test_get_svm_probs(trained_svm):
    clf, X, _ = trained_svm
    probas, preds = scores.get_svm_probs(clf, X)
    assert len(probas) == len(X)
    assert len(preds) == len(X)
    assert all(0.5 <= p <= 1.0 for p in probas)
    assert all(pred in (0, 1) for pred in preds)
