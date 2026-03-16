import numpy as np

def kl_divergence(P, Q, eps=1e-12):
    
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)

    # Avoid log(0)
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)

    # KL divergence
    kl = np.sum(P * np.log(P / Q))

    return float(kl)