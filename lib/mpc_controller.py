import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

class MPCController:
    def __init__(self, network, xy, target_u_field, target_v_field, horizon=5, dt=0.1):
        """
        Initialize the MPC controller.
        Args:
            network: Trained PINN network for state predictions
            xy: meshgrid coordinates for field prediction
            target_u_field: target u-velocity field (2D array)
            target_v_field: target v-velocity field (2D array)
            horizon: Prediction horizon
            dt: Time step
        """
        self.network = network
        self.xy = xy
        self.target_u_field = target_u_field
        self.target_v_field = target_v_field
        self.horizon = horizon
        self.dt = dt
        self.u_min = -2.0  # Lower bound for inlet u velocity
        self.u_max = 2.0   # Upper bound for inlet u velocity
        self.v_min = -2.0  # Lower bound for inlet v velocity
        self.v_max = 2.0   # Upper bound for inlet v velocity

    def predict_field(self, u0, v0):
        """
        Predict the velocity field for given inlet velocities u0 and v0.
        """
        try:
            u, v = self.compute_uv(self.network, self.xy)
            u_scaled = u * u0  # scale the field for u
            v_scaled = v * v0  # scale the field for v
            return u_scaled.reshape(self.target_u_field.shape), v_scaled.reshape(self.target_v_field.shape)
        except Exception as e:
            print(f"Error in predict_field: {e}")
            # Return zeros as fallback
            return np.zeros_like(self.target_u_field), np.zeros_like(self.target_v_field)

    def compute_uv(self, network, xy):
        try:
            xy = tf.constant(xy)
            with tf.GradientTape() as g:
                g.watch(xy)
                psi_p = network(xy)
            psi_p_j = g.batch_jacobian(psi_p, xy)
            u = psi_p_j[..., 0, 1]
            v = -psi_p_j[..., 0, 0]
            return u.numpy(), v.numpy()
        except Exception as e:
            print(f"Error in compute_uv: {e}")
            # Return zeros as fallback
            return np.zeros(xy.shape[0]), np.zeros(xy.shape[0])

    def cost_function(self, uv_seq):
        """
        Cost over the horizon: sum of squared errors between predicted and target u and v fields.
        uv_seq: [u0_1, v0_1, u0_2, v0_2, ..., u0_H, v0_H]
        """
        try:
            cost = 0.0
            uv_seq = np.array(uv_seq).reshape(self.horizon, 2)
            for u0, v0 in uv_seq:
                u_pred, v_pred = self.predict_field(u0, v0)
                err_u = u_pred - self.target_u_field
                err_v = v_pred - self.target_v_field
                cost += np.sum(err_u**2) + np.sum(err_v**2)
            return cost
        except Exception as e:
            print(f"Error in cost_function: {e}")
            return 1e10  # Return high cost for invalid inputs

    def optimize_control(self):
        """
        Optimize the inlet velocity sequence (u0, v0) over the horizon.
        Returns the first control input to apply.
        """
        try:
            initial_uv_seq = np.ones((self.horizon, 2)).flatten()
            bounds = [(self.u_min, self.u_max), (self.v_min, self.v_max)] * self.horizon
            result = minimize(
                self.cost_function,
                initial_uv_seq,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 50}
            )
            if result.success:
                return result.x[:2]  # Apply the first (u0, v0) in the sequence
            else:
                print(f"Optimization failed: {result.message}")
                return np.array([1.0, 1.0])  # Return default values
        except Exception as e:
            print(f"Error in optimize_control: {e}")
            return np.array([1.0, 1.0])  # Return default values

    def generate_reference_trajectory(self, current_u_field, current_v_field, steps):
        """
        Generate a reference trajectory to the target state.
        
        Args:
            current_u_field: Current u-velocity field
            current_v_field: Current v-velocity field
            steps: Number of steps in trajectory
            
        Returns:
            Reference trajectory for u and v fields
        """
        try:
            # Linear interpolation from current fields to target fields
            u_trajectory = np.zeros((steps,) + current_u_field.shape)
            v_trajectory = np.zeros((steps,) + current_v_field.shape)
            
            for i in range(steps):
                alpha = i / (steps - 1) if steps > 1 else 1.0
                u_trajectory[i] = (1 - alpha) * current_u_field + alpha * self.target_u_field
                v_trajectory[i] = (1 - alpha) * current_v_field + alpha * self.target_v_field
                
            return u_trajectory, v_trajectory
        except Exception as e:
            print(f"Error in generate_reference_trajectory: {e}")
            # Return target fields as fallback
            return np.tile(self.target_u_field, (steps, 1, 1)), np.tile(self.target_v_field, (steps, 1, 1)) 