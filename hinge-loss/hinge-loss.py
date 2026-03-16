import numpy as np

def hinge_loss(y, s, m=1.0, reduction="mean"):
    
    y = np.array(y)
    s = np.array(s)

    # Compute hinge loss for each example
    loss = np.maximum(0, m - y * s)

    if reduction == "sum":
        return float(np.sum(loss))
    elif reduction == "none":
        return loss
    else:  # default mean
        return float(np.mean(loss))