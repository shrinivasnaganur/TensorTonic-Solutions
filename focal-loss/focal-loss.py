import numpy as np

def focal_loss(y, p, gamma=2.0, eps=1e-12):

    y = np.array(y, dtype=float)
    p = np.array(p, dtype=float)

    # avoid log(0)
    p = np.clip(p, eps, 1 - eps)

    # focal loss
    loss = -( (1 - p) ** gamma * y * np.log(p) +
              (p ** gamma) * (1 - y) * np.log(1 - p) )

    return float(np.mean(loss))