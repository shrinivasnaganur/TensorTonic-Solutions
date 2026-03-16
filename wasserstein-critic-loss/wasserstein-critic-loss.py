import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):

    real_scores = np.array(real_scores, dtype=float)
    fake_scores = np.array(fake_scores, dtype=float)

    loss = np.mean(fake_scores) - np.mean(real_scores)

    return float(loss)