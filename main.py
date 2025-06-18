import lib.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B
from lib.mpc_controller import MPCController

def uv(network, xy):
    """
    Compute flow velocities (u, v) for the network with output (psi, p).

    Args:
        xy: network input variables as ndarray.

    Returns:
        (u, v) as ndarray.
    """

    xy = tf.constant(xy)
    with tf.GradientTape() as g:
        g.watch(xy)
        psi_p = network(xy)
    psi_p_j = g.batch_jacobian(psi_p, xy)
    u =  psi_p_j[..., 0, 1]
    v = -psi_p_j[..., 0, 0]
    return u.numpy(), v.numpy()

def contour(grid, x, y, z, title, levels=50):
    """
    Contour plot.

    Args:
        grid: plot position.
        x: x-array.
        y: y-array.
        z: z-array.
        title: title string.
        levels: number of contour lines.
    """

    # get the value range
    vmin = np.min(z)
    vmax = np.max(z)
    # plot a contour
    plt.subplot(grid)
    plt.contour(x, y, z, colors='k', linewidths=0.2, levels=levels)
    plt.contourf(x, y, z, cmap='rainbow', levels=levels, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.title(title)
    cbar = plt.colorbar(pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model
    for the cavity flow governed by the steady Navier-Stokes equation
    with Model Predictive Control.
    """

    # number of training samples (reduced for faster runtime)
    num_train_samples = 5000
    # number of test samples
    num_test_samples = 100

    # inlet flow velocity
    u0 = 1
    # density
    rho = 1
    # viscosity
    nu = 0.01

    # build a core network model
    network = Network().build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, rho=rho, nu=nu).build()

    # create training input
    xy_eqn = np.random.rand(num_train_samples, 2)
    xy_ub = np.random.rand(num_train_samples//2, 2)  # top-bottom boundaries
    xy_ub[..., 1] = np.round(xy_ub[..., 1])          # y-position is 0 or 1
    xy_lr = np.random.rand(num_train_samples//2, 2)  # left-right boundaries
    xy_lr[..., 0] = np.round(xy_lr[..., 0])          # x-position is 0 or 1
    xy_bnd = np.random.permutation(np.concatenate([xy_ub, xy_lr]))
    x_train = [xy_eqn, xy_bnd]

    # create training output
    zeros = np.zeros((num_train_samples, 2))
    uv_bnd = np.zeros((num_train_samples, 2))
    uv_bnd[..., 0] = u0 * np.floor(xy_bnd[..., 1])
    y_train = [zeros, zeros, uv_bnd]

    # train the model using L-BFGS-B algorithm with optimized parameters
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train,
                     factr=1e7,  # Reduced accuracy for faster convergence
                     maxiter=1000)  # Reduced max iterations
    lbfgs.fit()

    # Create meshgrid coordinates (x, y) for test plots
    x = np.linspace(0, 1, num_test_samples)
    y = np.linspace(0, 1, num_test_samples)
    x, y = np.meshgrid(x, y)
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)

    # Get initial state
    psi_p = network.predict(xy, batch_size=len(xy))
    psi, p = [psi_p[..., i].reshape(x.shape) for i in range(psi_p.shape[-1])]
    u, v = uv(network, xy)
    u = u.reshape(x.shape)
    v = v.reshape(x.shape)

    # Define target state (make u and v targets very different for demonstration)
    target_u = -1.2 * u  # Flip and scale u
    target_v = -1.5 * v  # Flip and scale v

    # Initialize MPC controller with the mesh and target fields
    mpc = MPCController(network, xy, target_u, target_v, horizon=5, dt=0.1)

    # MPC control loop
    num_steps = 20
    control_history = []
    u0_history = []
    v0_history = []
    u_field_history = []
    v_field_history = []

    u0 = 1.0  # initial inlet u velocity
    v0 = 1.0  # initial inlet v velocity
    
    print("Starting MPC control loop...")
    for step in range(num_steps):
        print(f"MPC step {step + 1}/{num_steps}")
        try:
            # Optimize inlet velocity sequence
            u0_opt, v0_opt = mpc.optimize_control()
            control_history.append((u0_opt, v0_opt))
            u0_history.append(u0_opt)
            v0_history.append(v0_opt)
            # Predict field with optimized inlet velocities
            u_pred, v_pred = mpc.predict_field(u0_opt, v0_opt)
            u_field_history.append(u_pred)
            v_field_history.append(v_pred)
            # For next step, update the controller if needed (here, we keep the target fixed)
            u0 = u0_opt
            v0 = v0_opt
            print(f"  Optimized u0={u0_opt:.4f}, v0={v0_opt:.4f}")
        except Exception as e:
            print(f"Error in MPC step {step + 1}: {e}")
            break

    print("MPC control loop completed. Creating plots...")

    # Plot results
    fig = plt.figure(figsize=(25, 10))
    gs = GridSpec(2, 4)

    # Plot initial states
    plt.subplot(gs[0, 0])
    plt.contourf(x, y, u, cmap='rainbow', levels=50)
    plt.colorbar()
    plt.title('Initial u-velocity')

    plt.subplot(gs[1, 0])
    plt.contourf(x, y, v, cmap='rainbow', levels=50)
    plt.colorbar()
    plt.title('Initial v-velocity')

    # Plot target states
    plt.subplot(gs[0, 1])
    plt.contourf(x, y, target_u, cmap='rainbow', levels=50)
    plt.colorbar()
    plt.title('Target u-velocity')

    plt.subplot(gs[1, 1])
    plt.contourf(x, y, target_v, cmap='rainbow', levels=50)
    plt.colorbar()
    plt.title('Target v-velocity')

    # Plot FINAL controlled u and v fields (last step)
    final_u = u_field_history[-1]
    final_v = v_field_history[-1]
    plt.subplot(gs[0, 2])
    plt.contourf(x, y, final_u, cmap='rainbow', levels=50)
    plt.colorbar()
    plt.title('Final Controlled u-velocity')

    plt.subplot(gs[1, 2])
    plt.contourf(x, y, final_v, cmap='rainbow', levels=50)
    plt.colorbar()
    plt.title('Final Controlled v-velocity')

    # Plot control history
    plt.subplot(gs[0, 3])
    plt.plot(u0_history, 'b-', linewidth=2, label='u0')
    plt.plot(v0_history, 'g-', linewidth=2, label='v0')
    plt.title('Inlet Velocity (u0, v0) History')
    plt.xlabel('Step')
    plt.ylabel('Inlet Value')
    plt.legend()
    plt.grid(True)

    # Plot error over time
    plt.subplot(gs[1, 3])
    errors = [np.linalg.norm(u_pred - target_u) + np.linalg.norm(v_pred - target_v) for u_pred, v_pred in zip(u_field_history, v_field_history)]
    plt.plot(errors, 'r-', linewidth=2)
    plt.title('Field Error vs Target')
    plt.xlabel('Step')
    plt.ylabel('L2 Error (u+v)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print final results
    print(f"Initial inlet velocities: u0=1.0, v0=1.0")
    print(f"Final optimized inlet velocities: u0={u0_history[-1]:.4f}, v0={v0_history[-1]:.4f}")
    print(f"Final field error: {errors[-1]:.4f}")

    # Save model weights
    network.save_weights('trained_pinn.weights.h5')
