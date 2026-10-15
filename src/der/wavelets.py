# SPDX-License-Identifier: MPL-2.0
"""Five-level Haar features, ported from der_wavedec / der_test_ks."""

import numpy as np
from scipy.special import ndtr


def haar_coefficients(shapes):
    """MATLAB wavedec(...,5,'haar'), symmetric extension; truncate to input length."""
    a = np.asarray(shapes, dtype=float)
    width = a.shape[1]
    details = []
    for _ in range(5):
        if a.shape[1] % 2:
            a = np.concatenate((a, a[:, -1:]), axis=1)
        even, odd = a[:, ::2], a[:, 1::2]
        details.append((even - odd) / np.sqrt(2))
        a = (even + odd) / np.sqrt(2)
    return np.concatenate([a, *details[::-1]], axis=1)[:, :width]


def ks_statistic(x):
    x = np.sort(np.asarray(x)[np.isfinite(x)])
    if len(x) < 2 or np.std(x, ddof=1) == 0:
        return 0.0
    unique, counts = np.unique(x, return_counts=True)
    right = np.cumsum(counts) / len(x)
    left = np.r_[0.0, right[:-1]]
    theoretical = ndtr((unique - x.mean()) / x.std(ddof=1))
    return float(np.max(np.abs(np.r_[left - theoretical, right - theoretical])))


def features(shapes, inputs=10):
    cc = haar_coefficients(shapes)
    scores = np.zeros(cc.shape[1])
    for j, column in enumerate(cc.T):
        sd = column.std(ddof=1) if len(column) > 1 else 0
        aux = column[(column > column.mean() - 3 * sd) & (column < column.mean() + 3 * sd)]
        if len(aux) > 10:
            scores[j] = ks_statistic(aux)
    # MATLAB ascending stable sort followed by reversal: high indices win ties.
    chosen = np.argsort(scores, kind="stable")[::-1][:inputs]
    return cc[:, chosen]
