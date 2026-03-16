import numpy as np

def label_smoothing_loss(predictions, target, epsilon=0.1):

    predictions = np.array(predictions, dtype=float)
    K = len(predictions)

    # avoid log(0)
    predictions = np.clip(predictions, 1e-12, 1.0)

    # build smoothed target distribution
    q = np.full(K, epsilon / K)
    q[target] = (1 - epsilon) + (epsilon / K)

    # cross-entropy loss
    loss = -np.sum(q * np.log(predictions))

    return float(loss)