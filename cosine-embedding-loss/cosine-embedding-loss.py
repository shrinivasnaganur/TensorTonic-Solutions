import numpy as np

def cosine_embedding_loss(x1, x2, label, margin=0.0):

    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)

    # cosine similarity
    cos = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    if label == 1:
        loss = 1 - cos
    else:  # label == -1
        loss = max(0.0, cos - margin)

    return float(loss)