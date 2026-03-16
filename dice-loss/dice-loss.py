import numpy as np

def dice_loss(y, p, eps=1e-7):

    y = np.array(y)
    p = np.array(p)

    # Flatten
    y = y.flatten()
    p = p.flatten()

    # Intersection
    intersection = np.sum(y * p)

    # Dice coefficient
    dice = (2 * intersection + eps) / (np.sum(y) + np.sum(p) + eps)

    # Dice loss
    loss = 1 - dice

    return float(loss)