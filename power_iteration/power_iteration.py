import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    y = np.array([1] * data.shape[0]).T
    eigenvalue = -1
    for i in range(num_steps):
        new_y = data.dot(y)
        eigenvalue = np.max(new_y) / np.max(y) 
        y = new_y / np.sqrt(np.sum(new_y ** 2))
    return float(eigenvalue), y