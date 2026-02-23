import os
import sys
import json
import time
import logging
import argparse
import tensorflow as tf
import numpy as np

from models.models import DeepStatisticalSolver


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


def convert_to_native_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert (dict, list, numpy type, etc.)
    
    Returns:
        Object with Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def setup_gpu(gpu_id=None):
    """Configure GPU settings for TensorFlow 2.x"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            if gpu_id is not None:
                # Use specific GPU
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                gpu = gpus[gpu_id]
            else:
                # Use all GPUs
                gpu = gpus[0]
            
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logging.info(f'Using GPU: {gpu}')
            
        except RuntimeError as e:
            logging.error(f'GPU configuration error: {e}')
    else:
        logging.info('No GPU found, using CPU')


def setup_logger(result_dir):
    """Setup logging to both console and file"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(result_dir, 'model.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s: %(message)s", 
        datefmt='%Y-%m-%d %H:%M'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Physics-Integrated GCN for WDS State Estimation'
    )

    # Inference mode
    parser.add_argument('--infer_data', type=str, default=None,
        help='Data path for inference mode. Requires --result_dir to be specified.')

    # Training parameters
    parser.add_argument('--random_seed', type=int, default=None,
        help='Random seed for reproducibility. Random by default.')
    parser.add_argument('--gpu', type=int, default=None,
        help='GPU device ID to use. Uses all available GPUs if not specified.')
    parser.add_argument('--max_iter', type=int, default=100000,
        help='Maximum number of training iterations')
    parser.add_argument('--batch_size', type=int, default=500,
        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
        help='Initial learning rate')
    parser.add_argument('--discount', type=float, default=0.9,
        help='Discount factor for layer-wise loss weighting')
    parser.add_argument('--track_validation', type=int, default=1000,
        help='Validate and save model every N iterations')
    parser.add_argument('--data_directory', type=str, 
        default='datasets/asnet2_1/',
        help='Directory containing training data')
    parser.add_argument('--alpha', type=float, default=0.5,
        help='Weight for supervised loss (1-alpha for physics loss)')

    # Model architecture parameters
    parser.add_argument('--latent_dimension', type=int, default=20,
        help='Dimension of latent node embeddings')
    parser.add_argument('--hidden_layers', type=int, default=2,
        help='Number of hidden layers in MLPs')
    parser.add_argument('--correction_updates', type=int, default=20,
        help='Number of GCN message passing iterations')
    parser.add_argument('--non_linearity', type=str, default='leaky_relu',
        choices=['relu', 'leaky_relu', 'elu', 'tanh'],
        help='Activation function for neural networks')

    # Model saving/loading
    parser.add_argument('--result_dir', type=str, default=None,
        help='Directory to save/load model. Creates new dir if not specified.')
    parser.add_argument('--save_frequency', type=int, default=5000,
        help='Save model checkpoint every N iterations')

    # Learning mode
    parser.add_argument('--unsupervised', action='store_true',
        help='Enable unsupervised (physics-based) learning mode. Default is supervised.')

    return parser.parse_args()


def main():
    """Main training/inference pipeline"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine proxy value (True for supervised, False for unsupervised)
    proxy = not args.unsupervised
    
    # Set random seeds for reproducibility
    if args.random_seed is not None:
        tf.random.set_seed(args.random_seed)
        np.random.seed(args.random_seed)
        logging.info(f'Random seed set to: {args.random_seed}')
    
    # Configure GPU
    setup_gpu(args.gpu)
    
    # Setup results directory
    if args.result_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join('results', timestamp)
    else:
        result_dir = args.result_dir
    
    # Create directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(result_dir)
    
    # Log configuration
    logger.info('='*80)
    logger.info('Physics-Integrated GCN for WDS State Estimation')
    logger.info('='*80)
    logger.info(f'TensorFlow version: {tf.__version__}')
    logger.info(f'Result directory: {result_dir}')
    logger.info(f'Data directory: {args.data_directory}')
    logger.info(f'Learning mode: {"Supervised" if proxy else "Unsupervised (Physics-based)"}')
    logger.info('')
    
    # Save configuration
    config_file = os.path.join(result_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f'Configuration saved to: {config_file}')
    
    # ========================================================================
    # INFERENCE MODE
    # ========================================================================
    if (args.infer_data is not None) and (args.result_dir is not None):
        logger.info('\n' + '='*80)
        logger.info('INFERENCE MODE')
        logger.info('='*80)
        
        # Load model from checkpoint
        logger.info(f'Loading model from: {args.result_dir}')
        model = DeepStatisticalSolver(
            latent_dimension=args.latent_dimension,
            hidden_layers=args.hidden_layers,
            correction_updates=args.correction_updates,
            alpha=args.alpha,
            non_linearity=args.non_linearity,
            batch_size=args.batch_size,
            name='physics_gcn',
            directory=result_dir,
            default_data_directory=args.data_directory,
            model_to_restore=args.result_dir,
            proxy=proxy
        )
        
        # Evaluate on test data
        logger.info(f'Evaluating on: {args.infer_data}')
        test_metrics = model.evaluate(
            mode='test',
            data_directory=args.infer_data
        )
        
        logger.info('\nTest Results:')
        logger.info('-'*40)
        for metric_name, metric_value in test_metrics.items():
            logger.info(f'  {metric_name}: {metric_value:.6f}')
        
        # Save predictions
        predictions_file = os.path.join(result_dir, 'predictions_test.npz')
        logger.info(f'\nPredictions saved to: {predictions_file}')
    
    # ========================================================================
    # TRAINING MODE
    # ========================================================================
    else:
        logger.info('\n' + '='*80)
        logger.info('TRAINING MODE')
        logger.info('='*80)
        
        # Log hyperparameters
        logger.info('\nHyperparameters:')
        logger.info('-'*40)
        logger.info(f'  Latent dimension: {args.latent_dimension}')
        logger.info(f'  Hidden layers: {args.hidden_layers}')
        logger.info(f'  Correction updates: {args.correction_updates}')
        logger.info(f'  Alpha (step size): {args.alpha}')
        logger.info(f'  Learning rate: {args.learning_rate}')
        logger.info(f'  Batch size: {args.batch_size}')
        logger.info(f'  Discount factor: {args.discount}')
        logger.info(f'  Non-linearity: {args.non_linearity}')
        logger.info(f'  Max iterations: {args.max_iter}')
        logger.info(f'  Learning mode: {"Supervised" if proxy else "Unsupervised (Physics-based)"}')
        logger.info('')
        
        # Build model
        model = DeepStatisticalSolver(
            latent_dimension=args.latent_dimension,
            hidden_layers=args.hidden_layers,
            correction_updates=args.correction_updates,
            alpha=args.alpha,
            non_linearity=args.non_linearity,
            batch_size=args.batch_size,
            name='physics_gcn',
            directory=result_dir,
            default_data_directory=args.data_directory,
            model_to_restore=args.result_dir,
            proxy=proxy
        )
        
        # Train model
        logger.info('Starting training...\n')
        train_start_time = time.time()
        
        try:
            model.train(
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                discount=args.discount,
                data_directory=args.data_directory,
                save_step=args.track_validation,
                save_frequency=args.save_frequency
            )
        except KeyboardInterrupt:
            logger.info('\n\nTraining interrupted by user')
        
        train_duration = time.time() - train_start_time
        logger.info(f'\nTraining completed in {train_duration/3600:.2f} hours')
        
        # Evaluate on validation set
        logger.info('\n' + '='*80)
        logger.info('VALIDATION EVALUATION')
        logger.info('='*80)
        
        val_metrics = model.evaluate(
            mode='val',
            data_directory=args.data_directory
        )
        
        logger.info('Validation Results:')
        logger.info('-'*40)
        for metric_name, metric_value in val_metrics.items():
            logger.info(f'  {metric_name}: {metric_value:.6f}')
        
        # Evaluate on test set
        logger.info('\n' + '='*80)
        logger.info('TEST EVALUATION')
        logger.info('='*80)
        
        test_metrics = model.evaluate(
            mode='test',
            data_directory=args.data_directory
        )
        
        logger.info('Test Results:')
        logger.info('-'*40)
        for metric_name, metric_value in test_metrics.items():
            logger.info(f'  {metric_name}: {metric_value:.6f}')
        
        # Save final model
        final_model_path = os.path.join(result_dir, 'final_model.h5')
        model.save(final_model_path)
        logger.info(f'\nFinal model saved to: {final_model_path}')
        
        # Save metrics summary
        metrics_summary = {
            'validation': val_metrics,
            'test': test_metrics,
            'training_duration_hours': train_duration / 3600,
            'learning_mode': 'supervised' if proxy else 'unsupervised'
        }
        
        # Convert to native types for JSON serialization
        metrics_summary = convert_to_native_types(metrics_summary)

        metrics_file = os.path.join(result_dir, 'metrics_summary.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        logger.info(f'Metrics summary saved to: {metrics_file}')
    
    logger.info('\n' + '='*80)
    logger.info('Done!')
    logger.info('='*80)


if __name__ == '__main__':
    main()