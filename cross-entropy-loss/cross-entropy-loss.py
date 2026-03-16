import numpy as np

def cross_entropy_loss(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    N = len(y_true)

    # Probability of the correct class for each sample
    p = y_pred[np.arange(N), y_true]

    # Cross-entropy loss
    loss = -np.mean(np.log(p))

    return float(loss)