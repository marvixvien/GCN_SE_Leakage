"""
Physics-based loss functions for Water Distribution System (WDS)
Hydraulic simulation using Graph Convolutional Networks

"""

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
    n_samples = tf.shape(params)[0]
    n_nodes = tf.shape(params)[1]
    n_edges = tf.shape(indices_edges)[1]
    d_out = tf.shape(params)[2]

    indices_batch = tf.range(n_samples, dtype=tf.int32)
    indices_batch = tf.expand_dims(indices_batch, 1)
    indices_batch = tf.tile(indices_batch, [1, n_edges])

    indices = n_nodes * indices_batch + indices_edges
    indices_flat = tf.reshape(indices, [-1, 1])

    params_flat = tf.reshape(params, [-1, d_out])
    gathered_flat = tf.gather_nd(params_flat, indices_flat)
    gathered = tf.reshape(gathered_flat, [n_samples, n_edges, d_out])

    return gathered


def custom_scatter(indices_edges, params, shape):
    """
    Batch-aware scatter operation for aggregating edge values to nodes.
    
    Args:
        indices_edges: tf.Tensor of shape [n_samples, n_edges], dtype=tf.int32
        params: tf.Tensor of shape [n_samples, n_edges, d_F], dtype=tf.float32
        shape: Target shape [n_samples, n_nodes, d_F]
    
    Returns:
        tf.Tensor of shape [n_samples, n_nodes, d_F], dtype=tf.float32
    """
    n_samples = tf.shape(params)[0]
    n_nodes = shape[1]
    n_edges = tf.shape(params)[1]
    d_F = tf.shape(params)[2]

    indices_batch = tf.range(n_samples, dtype=tf.int32)
    indices_batch = tf.expand_dims(indices_batch, 1)
    indices_batch = tf.tile(indices_batch, [1, n_edges])

    indices = n_nodes * indices_batch + indices_edges
    indices_flat = tf.reshape(indices, [-1, 1])

    params_flat = tf.reshape(params, [n_samples * n_edges, d_F])

    scattered_flat = tf.scatter_nd(
        indices_flat, 
        params_flat, 
        shape=[n_samples * n_nodes, d_F]
    )

    scattered = tf.reshape(scattered_flat, [n_samples, n_nodes, d_F])

    return scattered


class Problem:
    """
    Water Distribution System (WDS) hydraulic simulation problem.
    Implements physics-based constraints for GCN training.
    """

    def __init__(self, name='WDS_Hydraulic_Simulation'):
        self.name = name
        
        # Input dimensions
        self.d_in_A = 1  # Edge features (pipe resistance coefficient)
        self.d_in_B = 4  # Node features (demand indicator, demand, head indicator, known head)

        # Output dimensions
        self.d_out = 1  # Hydraulic head

        # Equation dimension per node
        self.d_F = 1

        # Initial guess for head (in meters)
        self.initial_U = np.array([399.], dtype=np.float32)

        # Standardization constants (computed from training data)
        # B features: [demand_indicator, demand, head_indicator, known_head]
        self.B_mean = np.array(
            [0.0, 0.005844823156379753, 0.0, 15.621948897809508], 
            dtype=np.float32
        )
        self.B_std = np.array(
            [1.0, 0.004587796037337935, 1.0, 77.32483667915464], 
            dtype=np.float32
        )
        
        # A features: [start_node, end_node, pipe_coefficient]
        self.A_mean = np.array(
            [0.0, 0.0, 0.08255444886402044], 
            dtype=np.float32
        )
        self.A_std = np.array(
            [1.0, 1.0, 0.17580375469346807], 
            dtype=np.float32
        )

    def cost_function(self, U, A, B):
        """
        Compute physics-based loss for WDS hydraulic state estimation.
        
        Implements:
        1. Mass balance (continuity equation) at junctions
        2. Head consistency at source nodes (reservoirs/tanks)
        
        Args:
            U: Predicted heads, shape [n_samples, n_nodes, 1]
            A: Edge features, shape [n_samples, n_edges, 3]
               A[:,:,0] = start node index
               A[:,:,1] = end node index
               A[:,:,2] = pipe resistance coefficient (1/c_ij)
            B: Node features, shape [n_samples, n_nodes, 4]
               B[:,:,0] = demand indicator (1: junction, 0: source)
               B[:,:,1] = demand [L/s]
               B[:,:,2] = head indicator (1: unknown, 0: known)
               B[:,:,3] = known head [m]
        
        Returns:
            cost_per_sample: tf.Tensor of shape [n_samples], physics violation cost
        """
        # Check for NaN/Inf values
        tf.debugging.check_numerics(U, 'U contains NaN or Inf')
        tf.debugging.check_numerics(A, 'A contains NaN or Inf')
        tf.debugging.check_numerics(B, 'B contains NaN or Inf')

        # Get dimensions
        n_samples = tf.shape(U)[0]
        n_nodes = tf.shape(U)[1]
        n_edges = tf.shape(A)[1]

        # Extract indices from A matrix
        indices_from = tf.cast(A[:, :, 0], tf.int32)  # Start nodes
        indices_to = tf.cast(A[:, :, 1], tf.int32)    # End nodes
        
        # Extract edge characteristics (pipe resistance coefficient)
        A_ij = A[:, :, 2:3]  # [n_samples, n_edges, 1]

        # Extract node features
        Nd = B[:, :, 0:1]           # Demand indicator (1: junction, 0: source)
        demand = B[:, :, 1:2]       # Demand [L/s]
        Nh = B[:, :, 2:3]           # Head indicator (1: unknown, 0: known)
        source_head = B[:, :, 3:4]  # Known head [m]

        # Compute actual head values (predicted for junctions, known for sources)
        H_i = custom_gather(
            Nd * U[:, :, 0:1] + (1 - Nd) * source_head, 
            indices_from
        )
        H_j = custom_gather(
            Nd * U[:, :, 0:1] + (1 - Nd) * source_head, 
            indices_to
        )

        # Compute head difference across pipes
        H_ij = H_i - H_j  # [n_samples, n_edges, 1]

        # Hazen-Williams equation: Q = sign(ΔH) * |ΔH|^0.54 * (1/c_ij)
        n = tf.constant(1.0 / 1.852, dtype=tf.float32)  # ≈ 0.54
        
        # Compute flow through pipes [L/s]
        Q_ij = tf.sign(H_ij) * tf.pow(
            tf.maximum(tf.abs(H_ij), 1e-9) * A_ij, 
            n
        )

        # Mass balance violation at junctions
        inflow = custom_scatter(indices_to, Q_ij, [n_samples, n_nodes, 1])
        outflow = custom_scatter(indices_from, Q_ij, [n_samples, n_nodes, 1])
        
        delta_Q = Nd * (-demand - outflow + inflow) ** 2

        # Head consistency at sources (reservoirs/tanks)
        delta_H = (1 - Nh) * (U[:, :, 0:1] - source_head) ** 2

        # Aggregate losses
        cost_per_sample = (
            tf.reduce_mean(delta_Q, axis=[1, 2]) + 
            tf.reduce_mean(delta_H, axis=[1, 2])
        )

        return cost_per_sample

    def supervised_loss(self, U_pred, U_true, mask=None):
        """
        Compute supervised MSE loss.
        
        Args:
            U_pred: Predicted heads [n_samples, n_nodes, 1]
            U_true: Ground truth heads [n_samples, n_nodes, 1]
            mask: Optional mask for semi-supervised learning [n_samples, n_nodes, 1]
        
        Returns:
            loss: Mean squared error
        """
        squared_diff = (U_pred - U_true) ** 2
        
        if mask is not None:
            squared_diff = squared_diff * mask
            return tf.reduce_sum(squared_diff) / (tf.reduce_sum(mask) + 1e-10)
        else:
            return tf.reduce_mean(squared_diff)

    def combined_loss(self, U_pred, U_true, A, B, beta=0.5, mask=None):
        """
        Combined supervised + physics-based loss.
        
        Args:
            U_pred: Predicted heads
            U_true: Ground truth heads
            A: Edge features
            B: Node features
            beta: Weight for supervised loss (1-beta for physics loss)
            mask: Optional mask for semi-supervised learning
        
        Returns:
            total_loss: Weighted combination of losses
            loss_supervised: Supervised component
            loss_physics: Physics component
        """
        # Supervised loss
        loss_supervised = self.supervised_loss(U_pred, U_true, mask)
        
        # Physics-based loss
        loss_physics = tf.reduce_mean(self.cost_function(U_pred, A, B))
        
        # Combined loss
        total_loss = beta * loss_supervised + (1 - beta) * loss_physics
        
        return total_loss, loss_supervised, loss_physics
