import numpy as np

def binary_focal_loss(targets, predictions, alpha=0.25, gamma=2.0, eps=1e-12):

    targets = np.array(targets, dtype=float)
    predictions = np.array(predictions, dtype=float)

    # avoid log(0)
    predictions = np.clip(predictions, eps, 1 - eps)

    # probability of the true class
    pt = np.where(targets == 1, predictions, 1 - predictions)

    # focal loss
    loss = -alpha * ((1 - pt) ** gamma) * np.log(pt)

    return float(np.mean(loss))