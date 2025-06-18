import numpy as np
import tensorflow as tf
from lib.network import Network
from lib.pinn import PINN
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Hermite polynomial basis (probabilists')
def hermite_basis(x, order):
    basis = [np.ones_like(x)]
    if order >= 1:
        basis.append(x)
    for n in range(2, order+1):
        basis.append(x * basis[-1] - (n-1) * basis[-2])
    return np.stack(basis, axis=-1)

def fit_pce(y_samples, x_samples, order):
    # Fit PCE coefficients for each output dimension
    basis = hermite_basis(x_samples, order)  # shape (N, order+1)
    coeffs = np.linalg.lstsq(basis, y_samples, rcond=None)[0]  # shape (order+1, ...)
    return coeffs

def pce_stats(coeffs):
    # Mean is coeff[0], variance is sum(coeff[1:]**2)
    mean = coeffs[0]
    var = np.sum(coeffs[1:]**2, axis=0)
    return mean, var

if __name__ == '__main__':
    # Settings
    order = 2
    num_samples = 1000  # Increased number of samples for smoother visualization
    # Load trained model
    network = Network().build()
    network.load_weights('trained_pinn.weights.h5')
    
    # Create a fine grid of points
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack((X.flatten(), Y.flatten()))
    
    # Sample input space with more points
    x_samples = np.random.rand(num_samples, 2)
    # Predict outputs
    y_samples = network.predict(x_samples)
    
    # Fit PCE for each output (psi, p)
    coeffs_psi = fit_pce(y_samples[:,0], x_samples[:,0], order)
    coeffs_p   = fit_pce(y_samples[:,1], x_samples[:,0], order)
    
    # Compute stats
    mean_psi, var_psi = pce_stats(coeffs_psi)
    mean_p, var_p = pce_stats(coeffs_p)
    print(f'PCE mean (psi): {mean_psi}, variance: {var_psi}')
    print(f'PCE mean (p): {mean_p}, variance: {var_p}')

    # Compute PCE predictions
    basis_psi = hermite_basis(x_samples[:,0], order)
    pce_pred_psi = np.dot(basis_psi, coeffs_psi)
    basis_p = hermite_basis(x_samples[:,0], order)
    pce_pred_p = np.dot(basis_p, coeffs_p)

    # Compute error between PINN and PCE predictions
    error_psi = np.abs(y_samples[:,0] - pce_pred_psi)
    error_p = np.abs(y_samples[:,1] - pce_pred_p)

    # Interpolate errors onto the fine grid
    error_psi_grid = griddata(x_samples, error_psi, (X, Y), method='cubic')
    error_p_grid = griddata(x_samples, error_p, (X, Y), method='cubic')

    # Plot error heatmaps
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(error_psi_grid, cmap='viridis', aspect='auto', 
                     extent=[0, 1, 0, 1], origin='lower', interpolation='bilinear')
    plt.colorbar(im1, label='Absolute Error')
    plt.title('PINN vs PCE Error Heatmap (u)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(error_p_grid, cmap='viridis', aspect='auto',
                     extent=[0, 1, 0, 1], origin='lower', interpolation='bilinear')
    plt.colorbar(im2, label='Absolute Error')
    plt.title('PINN vs PCE Error Heatmap (v)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.show()

    # Control plot: PINN vs PCE predictions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x_samples[:,0], y_samples[:,0], label='PINN (u)', alpha=0.5)
    plt.scatter(x_samples[:,0], pce_pred_psi, label='PCE (u)', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.title('PINN vs PCE Predictions (u)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(x_samples[:,0], y_samples[:,1], label='PINN (v)', alpha=0.5)
    plt.scatter(x_samples[:,0], pce_pred_p, label='PCE (v)', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.title('PINN vs PCE Predictions (v)')
    plt.legend()
    plt.tight_layout()
    plt.show() 