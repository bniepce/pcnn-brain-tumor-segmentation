import numpy as np

def dice(true, pred, empty_score=1.0):

    true = np.asarray(true).astype(np.bool)
    pred = np.asarray(pred).astype(np.bool)

    if true.shape != pred.shape:
        raise ValueError("Shape of arrays should be equal")

    s = true.sum() + pred.sum()
    if s == 0:
        return empty_score

    intersection = np.logical_and(true, pred)

    return 2. * intersection.sum() / s
