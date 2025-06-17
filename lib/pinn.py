import tensorflow as tf
from .layer import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for the steady Navier-Stokes equation.

    Attributes:
        network: keras network model with input (x, y) and output (psi, p).
        rho: density.
        nu: viscosity.
        grads: gradient layer.
    """

    def __init__(self, network, rho=1, nu=0.01):
        """
        Args:
            network: keras network model with input (x, y) and output (psi, p).
            rho: density.
            nu: viscosity.
        """

        self.network = network
        self.rho = rho
        self.nu = nu
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for the steady Navier-Stokes equation.

        Returns:
            PINN model for the steady Navier-Stokes equation with
                input: [ (x, y) relative to equation,
                         (x, y) relative to boundary condition ],
                output: [ (u, v) relative to equation (must be zero),
                          (psi, psi) relative to boundary condition (psi is duplicated because outputs require the same dimensions),
                          (u, v) relative to boundary condition ]
        """

        # equation input: (x, y)
        xy_eqn = tf.keras.layers.Input(shape=(2,))
        # boundary condition
        xy_bnd = tf.keras.layers.Input(shape=(2,))

        # compute gradients relative to equation
        _, p_grads, u_grads, v_grads = self.grads(xy_eqn)
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads

        # compute equation loss using Keras layers
        u_eqn = tf.keras.layers.Multiply()([u, u_x])
        v_u_y = tf.keras.layers.Multiply()([v, u_y])
        u_eqn = tf.keras.layers.Add()([u_eqn, v_u_y])
        p_x_rho = tf.keras.layers.Lambda(lambda x: x * (1.0/self.rho))(p_x)
        u_eqn = tf.keras.layers.Add()([u_eqn, p_x_rho])
        nu_u_xx = tf.keras.layers.Lambda(lambda x: x * self.nu)(u_xx)
        nu_u_yy = tf.keras.layers.Lambda(lambda x: x * self.nu)(u_yy)
        u_eqn = tf.keras.layers.Subtract()([u_eqn, tf.keras.layers.Add()([nu_u_xx, nu_u_yy])])

        v_eqn = tf.keras.layers.Multiply()([u, v_x])
        v_v_y = tf.keras.layers.Multiply()([v, v_y])
        v_eqn = tf.keras.layers.Add()([v_eqn, v_v_y])
        p_y_rho = tf.keras.layers.Lambda(lambda x: x * (1.0/self.rho))(p_y)
        v_eqn = tf.keras.layers.Add()([v_eqn, p_y_rho])
        nu_v_xx = tf.keras.layers.Lambda(lambda x: x * self.nu)(v_xx)
        nu_v_yy = tf.keras.layers.Lambda(lambda x: x * self.nu)(v_yy)
        v_eqn = tf.keras.layers.Subtract()([v_eqn, tf.keras.layers.Add()([nu_v_xx, nu_v_yy])])

        uv_eqn = tf.keras.layers.Concatenate()([u_eqn, v_eqn])

        # compute gradients relative to boundary condition
        psi_bnd, _, u_grads_bnd, v_grads_bnd = self.grads(xy_bnd)
        # compute boundary condition loss
        psi_bnd = tf.keras.layers.Concatenate()([psi_bnd, psi_bnd])
        uv_bnd = tf.keras.layers.Concatenate()([u_grads_bnd[0], v_grads_bnd[0]])

        # build the PINN model for the steady Navier-Stokes equation
        return tf.keras.models.Model(
            inputs=[xy_eqn, xy_bnd], outputs=[uv_eqn, psi_bnd, uv_bnd])
