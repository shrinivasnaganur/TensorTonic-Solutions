import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)

    # Euclidean distance
    d = np.sqrt(np.sum((a - b) ** 2, axis=-1))

    # Contrastive loss for each pair
    loss = y * (d ** 2) + (1 - y) * (np.maximum(0, margin - d) ** 2)

    if reduction == "sum":
        return float(np.sum(loss))
    elif reduction == "none":
        return loss
    else:  # default mean
        return float(np.mean(loss))