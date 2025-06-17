import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class PCELayer(Layer):
    """
    Polynomial Chaos Expansion (PCE) layer for uncertainty quantification.
    """
    
    def __init__(self, order=2, num_terms=None, **kwargs):
        """
        Initialize PCE layer.
        
        Args:
            order: Order of polynomial expansion
            num_terms: Number of terms in expansion (if None, will be calculated based on order)
        """
        super().__init__(**kwargs)
        self.order = order
        self.num_terms = num_terms if num_terms is not None else (order + 1) * (order + 2) // 2
        
    def hermite_poly(self, x, n):
        """
        Compute Hermite polynomials up to order n.
        
        Args:
            x: Input tensor
            n: Order of polynomial
            
        Returns:
            Hermite polynomial values
        """
        if n == 0:
            return tf.ones_like(x)
        elif n == 1:
            return x
        else:
            return x * self.hermite_poly(x, n-1) - (n-1) * self.hermite_poly(x, n-2)
            
    def compute_basis(self, x):
        """
        Compute PCE basis functions.
        
        Args:
            x: Input tensor
            
        Returns:
            PCE basis functions
        """
        basis = []
        for i in range(self.order + 1):
            basis.append(self.hermite_poly(x, i))
        return tf.stack(basis, axis=-1)
        
    def call(self, inputs):
        """
        Expand input using PCE.
        
        Args:
            inputs: List of [x, coeffs] where x is input tensor and coeffs are PCE coefficients
            
        Returns:
            PCE expansion
        """
        x, coeffs = inputs
        # Squeeze x to shape (batch,) if it has shape (batch, 1)
        if len(x.shape) > 1 and x.shape[-1] == 1:
            x = tf.squeeze(x, axis=-1)
        basis = self.compute_basis(x)  # shape (batch, order+1)
        # coeffs shape (batch, order+1)
        return tf.reduce_sum(basis * coeffs, axis=-1, keepdims=True)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "order": self.order,
            "num_terms": self.num_terms
        })
        return config 