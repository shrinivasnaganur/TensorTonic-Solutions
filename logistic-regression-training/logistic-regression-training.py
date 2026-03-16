import numpy as np

# Numerically stable sigmoid
def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.01, steps=1000):
    """
    X : shape (N, d)  -> input features
    y : shape (N,)    -> labels (0 or 1)
    lr : learning rate
    steps : number of gradient descent iterations
    """
    
    N, d = X.shape
    
    # Initialize parameters
    w = np.zeros(d)
    b = 0.0
    
    for _ in range(steps):
        
        # Forward pass
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # Gradients
        dw = (1/N) * np.dot(X.T, (p - y))
        db = (1/N) * np.sum(p - y)
        
        # Parameter update
        w = w - lr * dw
        b = b - lr * db
    
    return w, b