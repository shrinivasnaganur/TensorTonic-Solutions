import numpy as np

def triplet_loss(anchor, positive, negative, margin):

    anchor = np.array(anchor)
    positive = np.array(positive)
    negative = np.array(negative)

    # Squared L2 distances
    d_ap = np.sum((anchor - positive) ** 2, axis=-1)
    d_an = np.sum((anchor - negative) ** 2, axis=-1)

    # Triplet loss
    loss = np.maximum(0, d_ap - d_an + margin)

    # Return mean if batch, otherwise single value
    return float(np.mean(loss))