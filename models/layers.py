import tensorflow as tf
import numpy as np


def custom_gather(params, indices_edges):
    """
    Batch-aware gather operation for graph edges.
    
    Args:
        params: tf.Tensor of shape [n_samples, n_nodes, d_out], dtype=tf.float32
        indices_edges: tf.Tensor of shape [n_samples, n_edges], dtype=tf.int32
    
    Returns:
        tf.Tensor of shape [n_samples, n_edges, d_out], dtype=tf.float32
    """
    # Get all relevant dimensions
    n_samples = tf.shape(params)[0]
    n_nodes = tf.shape(params)[1]
    n_edges = tf.shape(indices_edges)[1]
    d_out = tf.shape(params)[2]

    # Build indices for the batch dimension using tf.range (cleaner in TF 2.x)
    indices_batch = tf.range(n_samples, dtype=tf.int32)
    indices_batch = tf.expand_dims(indices_batch, 1)
    indices_batch = tf.tile(indices_batch, [1, n_edges])

    # Flatten the indices
    indices = n_nodes * indices_batch + indices_edges
    indices_flat = tf.reshape(indices, [-1, 1])

    # Flatten the node parameters
    params_flat = tf.reshape(params, [-1, d_out])

    # Perform the gather operation
    gathered_flat = tf.gather_nd(params_flat, indices_flat)

    # Un-flatten the result
    gathered = tf.reshape(gathered_flat, [n_samples, n_edges, d_out])

    return gathered


def custom_scatter(indices_edges, params, shape):
    """
    Batch-aware scatter operation for aggregating edge values to nodes.
    
    Args:
        indices_edges: tf.Tensor of shape [n_samples, n_edges], dtype=tf.int32
        params: tf.Tensor of shape [n_samples, n_edges, d_F], dtype=tf.float32
        shape: List/Tensor of shape [n_samples, n_nodes, d_F]
    
    Returns:
        tf.Tensor of shape [n_samples, n_nodes, d_F], dtype=tf.float32
    """
    # Get all relevant dimensions
    n_samples = tf.shape(params)[0]
    n_nodes = shape[1]
    n_edges = tf.shape(params)[1]
    d_F = tf.shape(params)[2]

    # Build indices for the batch dimension
    indices_batch = tf.range(n_samples, dtype=tf.int32)
    indices_batch = tf.expand_dims(indices_batch, 1)
    indices_batch = tf.tile(indices_batch, [1, n_edges])

    # Stack batch and edge dimensions
    indices = n_nodes * indices_batch + indices_edges
    indices_flat = tf.reshape(indices, [-1, 1])

    # Flatten the edge parameters
    params_flat = tf.reshape(params, [n_samples * n_edges, d_F])

    # Perform the scatter operation
    scattered_flat = tf.scatter_nd(
        indices_flat,
        params_flat,
        shape=[n_samples * n_nodes, d_F]
    )

    # Un-flatten the result
    scattered = tf.reshape(scattered_flat, [n_samples, n_nodes, d_F])

    return scattered


class FullyConnected(tf.keras.layers.Layer):
    """
    Fully connected MLP block using tf.keras.layers API.
    
    Args:
        latent_dimension: Hidden dimension size
        hidden_layers: Number of layers
        non_lin: Activation function ('leaky_relu', 'relu', 'tanh', 'elu')
        input_dim: Input dimension (if None, uses latent_dimension)
        output_dim: Output dimension (if None, uses latent_dimension)
        name: Layer name
    """
    
    def __init__(
        self,
        latent_dimension=10,
        hidden_layers=3,
        non_lin='leaky_relu',
        input_dim=None,
        output_dim=None,
        name='fc_block',
        **kwargs
    ):
        super(FullyConnected, self).__init__(name=name, **kwargs)
        
        # Store parameters
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.non_lin_name = non_lin
        self.input_dim = input_dim if input_dim is not None else latent_dimension
        self.output_dim = output_dim if output_dim is not None else latent_dimension
        
        # Select activation function
        if non_lin == 'tanh':
            self.activation = tf.keras.activations.tanh
        elif non_lin == 'leaky_relu':
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif non_lin == 'relu':
            self.activation = tf.keras.activations.relu
        elif non_lin == 'elu':
            self.activation = tf.keras.activations.elu
        else:
            raise ValueError(f"Unknown activation: {non_lin}")
        
        # Build layers
        self.dense_layers = []
        for layer_idx in range(self.hidden_layers):
            # Determine layer dimensions
            if layer_idx == 0:
                in_dim = self.input_dim
            else:
                in_dim = self.latent_dimension
            
            if layer_idx == self.hidden_layers - 1:
                out_dim = self.output_dim
            else:
                out_dim = self.latent_dimension
            
            # Create dense layer with Xavier initialization
            dense = tf.keras.layers.Dense(
                out_dim,
                activation=None,  # We'll apply activation manually
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                bias_initializer=tf.keras.initializers.GlorotNormal(),
                name=f'{name}_dense_{layer_idx}'
            )
            self.dense_layers.append(dense)
    
    def call(self, inputs, training=None):
        """
        Forward pass through the MLP.
        
        Args:
            inputs: Tensor of shape [n_samples, n_elements, input_dim]
            training: Boolean for training mode
        
        Returns:
            Tensor of shape [n_samples, n_elements, output_dim]
        """
        # Get input shape
        n_samples = tf.shape(inputs)[0]
        n_elem = tf.shape(inputs)[1]
        d = tf.shape(inputs)[2]
        
        # Flatten to [n_samples * n_elem, d]
        h = tf.reshape(inputs, [-1, d])
        
        # Pass through all layers
        for layer_idx, dense in enumerate(self.dense_layers):
            h = dense(h)
            
            # Apply activation to all layers except the last
            if layer_idx < self.hidden_layers - 1:
                if isinstance(self.activation, tf.keras.layers.Layer):
                    h = self.activation(h)
                else:
                    h = self.activation(h)
        
        # Reshape back to [n_samples, n_elem, output_dim]
        output = tf.reshape(h, [n_samples, n_elem, -1])
        
        return output
    
    def get_config(self):
        """Get layer configuration for serialization"""
        config = super().get_config()
        config.update({
            'latent_dimension': self.latent_dimension,
            'hidden_layers': self.hidden_layers,
            'non_lin': self.non_lin_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
        })
        return config