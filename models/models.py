import os
import sys
import json
import time
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from models.layers import FullyConnected, custom_gather, custom_scatter


class DeepStatisticalSolver(tf.keras.Model):
    def __init__(
        self,
        latent_dimension=10,
        hidden_layers=3,
        correction_updates=5,
        alpha=1e-3,
        non_linearity='leaky_relu',
        batch_size=10,
        name='physics_gcn',
        directory='./',
        default_data_directory='datasets/spring/default',
        model_to_restore=None,
        proxy=True,
        **kwargs
    ):
        super(DeepStatisticalSolver, self).__init__(name=name, **kwargs)
        
        # Store hyperparameters
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.correction_updates = correction_updates
        self.alpha_value = alpha
        self.non_linearity = non_linearity
        self.batch_size = batch_size
        self.directory = directory
        self.default_data_directory = default_data_directory
        self.current_train_iter = 0
        self.proxy = proxy  # True: supervised learning, False: unsupervised learning
        
        # Initialize problem configuration
        self._load_problem_config(default_data_directory)
        
        # Restore or initialize configuration
        if model_to_restore is not None and os.path.exists(model_to_restore):
            self._restore_config(model_to_restore)
        
        # Build model weights
        self._build_model_weights()
        
        # Initialize training utilities
        self.optimizer = None
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        
        # Log configuration
        self.log_config()
    
    def _load_problem_config(self, default_data_directory):
        """Load problem-specific configuration"""
        try:
            sys.path.append(self.default_data_directory)

            from problem import Problem
            self.problem = Problem()
            self.d_in_A = self.problem.d_in_A
            self.d_in_B = self.problem.d_in_B
            self.d_out = self.problem.d_out
            self.d_F = self.problem.d_F
            self.initial_U = self.problem.initial_U
            
            # Standardization constants
            self.B_mean = self.problem.B_mean
            self.B_std = self.problem.B_std
            self.A_mean = self.problem.A_mean
            self.A_std = self.problem.A_std
            
        except ImportError as e:
            logging.error(f"Cannot load problem.py from {default_data_directory}")
            raise e
    
    def _restore_config(self, model_dir):
        """Restore configuration from saved model"""
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            logging.info(f'Restoring configuration from {config_path}')
            with open(config_path, 'r') as f:
                config = json.load(f)
            self._set_config(config)
        else:
            logging.warning(f'No config.json found in {model_dir}')
    
    def _build_model_weights(self):
        """Build all trainable model components"""
        
        # Message passing networks (phi_from, phi_to, phi_loop)
        self.phi_from_layers = []
        self.phi_to_layers = []
        self.phi_loop_layers = []
        
        # Node update networks (psi)
        self.psi_layers = []
        
        # Decoder networks (xi)
        self.decoder_layers = []
        
        # Build layers for each correction update
        for update in range(self.correction_updates):
            # Edge message networks
            self.phi_from_layers.append(
                FullyConnected(
                    latent_dimension=self.latent_dimension,
                    hidden_layers=self.hidden_layers,
                    non_lin=self.non_linearity,
                    input_dim=2 * self.latent_dimension + self.d_in_A,
                    output_dim=self.latent_dimension,
                    name=f'{self.name}_phi_from_{update}'
                )
            )
            
            self.phi_to_layers.append(
                FullyConnected(
                    latent_dimension=self.latent_dimension,
                    hidden_layers=self.hidden_layers,
                    non_lin=self.non_linearity,
                    input_dim=2 * self.latent_dimension + self.d_in_A,
                    output_dim=self.latent_dimension,
                    name=f'{self.name}_phi_to_{update}'
                )
            )
            
            self.phi_loop_layers.append(
                FullyConnected(
                    latent_dimension=self.latent_dimension,
                    hidden_layers=self.hidden_layers,
                    non_lin=self.non_linearity,
                    input_dim=2 * self.latent_dimension + self.d_in_A,
                    output_dim=self.latent_dimension,
                    name=f'{self.name}_phi_loop_{update}'
                )
            )
            
            # Node update network
            self.psi_layers.append(
                FullyConnected(
                    latent_dimension=self.latent_dimension,
                    hidden_layers=self.hidden_layers,
                    non_lin=self.non_linearity,
                    input_dim=4 * self.latent_dimension + self.d_in_B,
                    output_dim=self.latent_dimension,
                    name=f'{self.name}_psi_{update}'
                )
            )
        
        # Build decoders (one more than correction updates)
        for update in range(self.correction_updates + 1):
            self.decoder_layers.append(
                FullyConnected(
                    latent_dimension=self.latent_dimension,
                    hidden_layers=self.hidden_layers,
                    non_lin=self.non_linearity,
                    input_dim=self.latent_dimension,
                    output_dim=self.d_out,
                    name=f'{self.name}_decoder_{update}'
                )
            )
    
    def call(self, inputs, training=False):
        """
        Forward pass through the GCN.
        
        Args:
            inputs: Tuple of (A, B) where:
                A: Edge features [batch, n_edges, d_in_A + 2]
                B: Node features [batch, n_nodes, d_in_B]
            training: Boolean for training mode
        
        Returns:
            Dictionary containing:
                - 'predictions': List of predictions at each update
                - 'final_prediction': Final hydraulic head prediction
        """
        A, B = inputs
        
        A = tf.cast(A, tf.float32)
        B = tf.cast(B, tf.float32)

        # Get dimensions
        batch_size = tf.shape(A)[0]
        n_nodes = tf.shape(B)[1]
        n_edges = tf.shape(A)[1]
        
        # Standardize inputs
        A_mean_tf = tf.constant(self.A_mean, dtype=tf.float32)
        A_mean_tf = tf.reshape(A_mean_tf, [1, 1, -1])
    
        A_std_tf = tf.constant(self.A_std, dtype=tf.float32)
        A_std_tf = tf.reshape(A_std_tf, [1, 1, -1])
    
        B_mean_tf = tf.constant(self.B_mean, dtype=tf.float32)
        B_mean_tf = tf.reshape(B_mean_tf, [1, 1, -1])
    
        B_std_tf = tf.constant(self.B_std, dtype=tf.float32)
        B_std_tf = tf.reshape(B_std_tf, [1, 1, -1])
        
        a = (A - A_mean_tf) / A_std_tf
        b = (B - B_mean_tf) / B_std_tf
        
        # Extract edge indices (not standardized)
        indices_from = tf.cast(A[:, :, 0], tf.int32)
        indices_to = tf.cast(A[:, :, 1], tf.int32)
        
        # Extract normalized edge features
        a_ij = a[:, :, 2:]
        
        # Create loop mask
        mask_loop = tf.cast(tf.equal(indices_from, indices_to), tf.float32)
        mask_loop = tf.expand_dims(mask_loop, -1)
        
        # Initial offset for predictions
        initial_U_tf = tf.reshape(self.initial_U, [1, 1, -1])
        initial_U_tf = tf.tile(initial_U_tf, [batch_size, n_nodes, 1])
        
        # Initialize latent state H[0] = 0
        H = tf.zeros([batch_size, n_nodes, self.latent_dimension])
        
        # Store predictions at each update
        predictions = []
        
        # Initial prediction U[0]
        # After
        U = self.decoder_layers[0](H, training=training) + tf.cast(initial_U_tf, tf.float32)
        predictions.append(U)
        
        # Correction updates (message passing iterations)
        for update in range(self.correction_updates):
            # Gather node states at edge endpoints
            H_from = custom_gather(H, indices_from)
            H_to = custom_gather(H, indices_to)
            
            # Concatenate edge inputs
            phi_input = tf.concat([H_from, H_to, a_ij], axis=2)
            
            # Compute edge messages
            phi_from = self.phi_from_layers[update](phi_input, training=training)
            phi_to = self.phi_to_layers[update](phi_input, training=training)
            phi_loop = self.phi_loop_layers[update](phi_input, training=training)
            
            # Apply loop mask
            phi_from = phi_from * (1.0 - mask_loop)
            phi_to = phi_to * (1.0 - mask_loop)
            phi_loop = phi_loop * mask_loop
            
            # Aggregate messages at nodes
            phi_from_sum = custom_scatter(
                indices_from,
                phi_from,
                [batch_size, n_nodes, self.latent_dimension]
            )
            phi_to_sum = custom_scatter(
                indices_to,
                phi_to,
                [batch_size, n_nodes, self.latent_dimension]
            )
            phi_loop_sum = custom_scatter(
                indices_to,
                phi_loop,
                [batch_size, n_nodes, self.latent_dimension]
            )
            
            # Concatenate node inputs
            psi_input = tf.concat([
                H,
                phi_from_sum,
                phi_to_sum,
                phi_loop_sum,
                b
            ], axis=2)
            
            # Compute node update
            correction = self.psi_layers[update](psi_input, training=training)
            
            # Update latent state
            H = H + self.alpha_value * correction
            
            # Decode to prediction
            U = self.decoder_layers[update + 1](H, training=training) + tf.cast(initial_U_tf, tf.float32)
            predictions.append(U)
        
        return {
            'predictions': predictions,
            'final_prediction': predictions[-1]
        }
    
    @tf.function
    def train_step(self, A, B, U_gt, discount):
        """
        Single training step with gradient computation.
        
        Args:
            A: Edge features [batch, n_edges, d_in_A + 2]
            B: Node features [batch, n_nodes, d_in_B]
            U_gt: Ground truth heads [batch, n_nodes, d_out]
            discount: Discount factor for layer-wise loss
        
        Returns:
            Dictionary of losses
        """
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self((A, B), training=True)
            predictions = outputs['predictions']
            
            # Compute losses at each update
            total_loss = 0.0
            supervised_losses = []
            physics_losses = []
            
            for update_idx, U_pred in enumerate(predictions):
                # Supervised loss (MSE)
                loss_supervised = self.problem.supervised_loss(U_pred, U_gt)
                
                # Physics-based loss
                loss_physics = tf.reduce_mean(
                    self.problem.cost_function(U_pred, A, B)
                )
                
                # Combined loss based on proxy setting
                if self.proxy:
                    # Supervised learning: prioritize supervised loss
                    loss_combined = loss_supervised
                else:
                    # Unsupervised learning: use physics-based loss only
                    loss_combined = loss_physics
                
                # Apply discount factor
                discount_factor = discount ** (self.correction_updates - update_idx)
                total_loss += loss_combined * discount_factor
                
                supervised_losses.append(loss_supervised)
                physics_losses.append(loss_physics)
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {
            'total_loss': total_loss,
            'final_supervised_loss': supervised_losses[-1],
            'final_physics_loss': physics_losses[-1],
            'final_loss': (
                supervised_losses[-1] if self.proxy else physics_losses[-1]
            )
        }
    
    def train(
        self,
        max_iter=10000,
        learning_rate=1e-3,
        discount=0.9,
        data_directory='datasets/asnet2_enforce5_b4/',
        save_step=1000,
        save_frequency=5000
    ):
        """
        Train the model using the provided dataset.
        
        Args:
            max_iter: Maximum training iterations
            learning_rate: Initial learning rate
            discount: Discount factor for layer-wise loss
            data_directory: Path to training data
            save_step: Validation frequency
            save_frequency: Model checkpoint frequency
        """
        logging.info('='*80)
        logging.info('Starting Training')
        logging.info('='*80)
        logging.info(f'Max iterations: {max_iter}')
        logging.info(f'Learning rate: {learning_rate}')
        logging.info(f'Discount: {discount}')
        logging.info(f'Data directory: {data_directory}')
        logging.info(f'Training mode: {"Supervised" if self.proxy else "Unsupervised"}')
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Load datasets
        train_dataset = self._load_dataset(data_directory, 'train')
        val_dataset = self._load_dataset(data_directory, 'val')
        
        # Setup TensorBoard
        train_summary_writer = tf.summary.create_file_writer(
            os.path.join(self.directory, 'logs', 'train')
        )
        val_summary_writer = tf.summary.create_file_writer(
            os.path.join(self.directory, 'logs', 'val')
        )
        
        # Training loop
        train_iter = iter(train_dataset)
        
        for iteration in tqdm(range(max_iter), desc='Training'):
            self.current_train_iter = iteration
            
            # Get next batch
            try:
                A, B, U_gt = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataset)
                A, B, U_gt = next(train_iter)
            
            # Training step
            losses = self.train_step(A, B, U_gt, discount)
            
            # Update metrics
            self.train_loss_metric.update_state(losses['final_loss'])
            
            # Log to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('total_loss', losses['total_loss'], step=iteration)
                tf.summary.scalar('final_loss', losses['final_loss'], step=iteration)
                tf.summary.scalar('supervised_loss', losses['final_supervised_loss'], step=iteration)
                tf.summary.scalar('physics_loss', losses['final_physics_loss'], step=iteration)
            
            # Periodic validation and saving
            if (iteration % save_step == 0) or (iteration == max_iter - 1):
                # Validate
                val_loss = self._validate(val_dataset, discount)
                
                # Log results
                train_loss = self.train_loss_metric.result().numpy()
                logging.info(f'\nIteration {iteration}:')
                logging.info(f'  Train loss: {train_loss:.6f}')
                logging.info(f'  Val loss: {val_loss:.6f}')
                
                # Log to TensorBoard
                with val_summary_writer.as_default():
                    tf.summary.scalar('loss', val_loss, step=iteration)
                
                # Reset metrics
                self.train_loss_metric.reset_state()
                self.val_loss_metric.reset_state()
                
                # Save model
                self.save_model()
            
            # Save checkpoint
            if iteration % save_frequency == 0 and iteration > 0:
                checkpoint_dir = os.path.join(
                    self.directory,
                    f'checkpoint_{iteration}'
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.save_weights(os.path.join(checkpoint_dir, 'model_weights.weights.h5'))
                logging.info(f'Checkpoint saved at iteration {iteration}')
        
        # Final save
        self.save_model()
        logging.info('\nTraining completed!')
    
    def _validate(self, val_dataset, discount):
        """Run validation on validation dataset"""
        val_losses = []
        
        for A, B, U_gt in val_dataset.take(10):  # Validate on 10 batches
            outputs = self((A, B), training=False)
            U_pred = outputs['final_prediction']
            
            # Compute loss based on proxy setting
            if self.proxy:
                # Supervised validation
                loss = self.problem.supervised_loss(U_pred, U_gt)
            else:
                # Unsupervised validation (physics-based)
                loss = tf.reduce_mean(
                    self.problem.cost_function(U_pred, A, B)
                )
            
            val_losses.append(loss.numpy())
        
        return np.mean(val_losses)
    
    def _load_dataset(self, data_directory, mode):
        """
        Load TFRecord dataset.
        
        Args:
            data_directory: Path to data
            mode: 'train', 'val', or 'test'
        
        Returns:
            tf.data.Dataset
        """
        tfrecord_path = os.path.join(data_directory, f'{mode}.tfrecords')
        
        def parse_tfrecord(serialized):
            """Parse a single TFRecord example"""
            features = {
                'A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'B': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'U': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
            }
            
            parsed = tf.io.parse_single_example(serialized, features)
            
            # Reshape to proper dimensions
            A = tf.reshape(parsed['A'], [-1, self.d_in_A + 2])
            B = tf.reshape(parsed['B'], [-1, self.d_in_B])
            U = tf.reshape(parsed['U'], [-1, self.d_out])
            
            return A, B, U
        
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.repeat()
        elif mode == 'val':
            dataset = dataset.repeat()
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def evaluate(self, mode='test', data_directory='datasets/asnet2_enforce5_b4/'):
        """
        Evaluate model on test/validation set.
        
        Args:
            mode: 'test' or 'val'
            data_directory: Path to data
        
        Returns:
            Dictionary of evaluation metrics
        """
        logging.info(f'\nEvaluating on {mode} set...')
        
        # Load numpy data for evaluation
        A = np.load(os.path.join(data_directory, f'A_{mode}.npy'))
        B = np.load(os.path.join(data_directory, f'B_{mode}.npy'))
        U_gt = np.load(os.path.join(data_directory, f'U_{mode}.npy'))
        
        # Get predictions
        outputs = self((A, B), training=False)
        U_pred = outputs['final_prediction'].numpy()
        
        # Compute metrics
        mse = np.mean((U_pred - U_gt) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(U_pred - U_gt))
        
        # Correlation
        corr = np.corrcoef(U_pred.flatten(), U_gt.flatten())[0, 1]
        
        # Physics loss
        physics_loss = np.mean(
            self.problem.cost_function(
                tf.constant(U_pred, dtype=tf.float32),
                tf.constant(A, dtype=tf.float32),
                tf.constant(B, dtype=tf.float32)
            ).numpy()
        )
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': corr,
            'physics_loss': physics_loss
        }
        
        # Save predictions
        pred_file = os.path.join(self.directory, f'predictions_{mode}.npz')
        np.savez(
            pred_file,
            U_pred=U_pred,
            U_gt=U_gt,
            A=A,
            B=B
        )
        logging.info(f'Predictions saved to: {pred_file}')
        
        return metrics
    
    def save_model(self):
        """Save model configuration and weights"""
        # Save configuration
        config = self._get_config()
        config_path = os.path.join(self.directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Save weights
        weights_path = os.path.join(self.directory, 'model_weights.weights.h5')
        self.save_weights(weights_path)
        
        logging.info(f'Model saved to: {self.directory}')
    
    def load_model(self, model_dir):
        """Load model from directory"""
        # Load configuration
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self._set_config(config)
        
        # Load weights
        weights_path = os.path.join(model_dir, 'model_weights.weights.h5')
    
        if not os.path.exists(weights_path):
            # Fallback: try TensorFlow format
            weights_path = os.path.join(model_dir, 'model_weights')
            if not os.path.exists(weights_path + '.index'):
                raise FileNotFoundError(f"Model weights not found in {model_dir}")
        
        print(f"Loading model weights from: {weights_path}")
        
        # CRITICAL: Build the model by calling it on dummy data
        print("Building model with dummy data...")
        dummy_A = tf.zeros([1, 10, self.d_in_A + 2], dtype=tf.float32)
        dummy_B = tf.zeros([1, 10, self.d_in_B], dtype=tf.float32)
        
        # Call the model to build it
        _ = self([dummy_A, dummy_B], training=False)
        print(" Model built successfully")
        
        # Load the weights
        self.load_weights(weights_path)
        print(" Model weights loaded successfully")
        
        # Verify loading was successful
        trainable_params = sum([tf.size(w).numpy() for w in self.trainable_weights])
        print(f" Model has {trainable_params:,} trainable parameters")
        
        if trainable_params == 0:
            raise ValueError("Model has 0 parameters! Check model architecture.")
        
    def _get_config(self):
        """Get model configuration as dictionary"""
        return {
            'latent_dimension': self.latent_dimension,
            'hidden_layers': self.hidden_layers,
            'correction_updates': self.correction_updates,
            'alpha': self.alpha_value,
            'non_linearity': self.non_linearity,
            'batch_size': self.batch_size,
            'name': self.name,
            'directory': self.directory,
            'current_train_iter': self.current_train_iter,
            'd_in_A': self.d_in_A,
            'd_in_B': self.d_in_B,
            'd_out': self.d_out,
            'initial_U': self.initial_U.tolist(),
            'A_mean': self.A_mean.tolist(),
            'A_std': self.A_std.tolist(),
            'B_mean': self.B_mean.tolist(),
            'B_std': self.B_std.tolist(),
            'proxy': self.proxy
        }
    
    def _set_config(self, config):
        """Set model configuration from dictionary"""
        self.latent_dimension = config['latent_dimension']
        self.hidden_layers = config['hidden_layers']
        self.correction_updates = config['correction_updates']
        self.alpha_value = config['alpha']
        self.non_linearity = config.get('non_linearity', 'leaky_relu')
        self.batch_size = config['batch_size']
        self.current_train_iter = config['current_train_iter']
        self.d_in_A = config['d_in_A']
        self.d_in_B = config['d_in_B']
        self.d_out = config['d_out']
        self.initial_U = np.array(config['initial_U'], dtype=np.float32)
        self.A_mean = np.array(config['A_mean'], dtype=np.float32)
        self.A_std = np.array(config['A_std'], dtype=np.float32)
        self.B_mean = np.array(config['B_mean'], dtype=np.float32)
        self.B_std = np.array(config['B_std'], dtype=np.float32)
        self.proxy = bool(config.get('proxy', True))
    
    def log_config(self):
        """Log model configuration"""
        logging.info('\nModel Configuration:')
        logging.info('-' * 40)
        logging.info(f'  Name: {self.name}')
        logging.info(f'  Directory: {self.directory}')
        logging.info(f'  Latent dimension: {self.latent_dimension}')
        logging.info(f'  Hidden layers: {self.hidden_layers}')
        logging.info(f'  Correction updates: {self.correction_updates}')
        logging.info(f'  Alpha: {self.alpha_value}')
        logging.info(f'  Non-linearity: {self.non_linearity}')
        logging.info(f'  Batch size: {self.batch_size}')
        logging.info(f'  Learning mode: {"Supervised" if self.proxy else "Unsupervised"}')
        logging.info(f'  d_in_A: {self.d_in_A}')
        logging.info(f'  d_in_B: {self.d_in_B}')
        logging.info(f'  d_out: {self.d_out}')