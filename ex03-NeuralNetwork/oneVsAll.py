from scipy.optimize import minimize
import numpy as np
from lrCostFunction import lr_cost_function
from gradient import gradient

def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    all_theta = np.zeros((num_labels, params + 1))
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        fmin = minimize(fun=lr_cost_function, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x
    return all_theta
