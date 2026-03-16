import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):

    Z1 = np.array(Z1, dtype=float)
    Z2 = np.array(Z2, dtype=float)

    N = Z1.shape[0]

    # similarity matrix
    S = np.dot(Z1, Z2.T) / temperature

    # numerical stability trick
    S = S - np.max(S, axis=1, keepdims=True)

    exp_S = np.exp(S)

    # positive similarities
    pos = np.diag(exp_S)

    # denominator
    denom = np.sum(exp_S, axis=1)

    loss = -np.mean(np.log(pos / denom))

    return float(loss)