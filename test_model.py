import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models.models import DeepStatisticalSolver
from utils.data_loader import load_numpy_dataset


def setup_logger():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def plot_predictions(U_pred, U_gt, save_path):
    """
    Plot predicted vs ground truth heads.
    
    Args:
        U_pred: Predicted heads
        U_gt: Ground truth heads
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(U_gt.flatten(), U_pred.flatten(), alpha=0.5, s=1)
    
    # Perfect prediction line
    min_val = min(U_gt.min(), U_pred.min())
    max_val = max(U_gt.max(), U_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Labels and title
    plt.xlabel('Ground Truth Head (m)', fontsize=12)
    plt.ylabel('Predicted Head (m)', fontsize=12)
    plt.title('GCN Predictions vs Ground Truth', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f'Prediction plot saved to: {save_path}')


def plot_error_distribution(U_pred, U_gt, save_path):
    """
    Plot distribution of prediction errors.
    
    Args:
        U_pred: Predicted heads
        U_gt: Ground truth heads
        save_path: Path to save plot
    """
    errors = (U_pred - U_gt).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Error (m)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(errors, vert=True)
    axes[1].set_ylabel('Error (m)', fontsize=12)
    axes[1].set_title('Error Box Plot', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f'Error distribution plot saved to: {save_path}')


def compute_detailed_metrics(U_pred, U_gt):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        U_pred: Predicted heads
        U_gt: Ground truth heads
    
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    pred_flat = U_pred.flatten()
    gt_flat = U_gt.flatten()
    
    # Basic metrics
    mse = np.mean((pred_flat - gt_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - gt_flat))
    
    # Correlation
    correlation = np.corrcoef(pred_flat, gt_flat)[0, 1]
    
    # R-squared
    ss_res = np.sum((gt_flat - pred_flat) ** 2)
    ss_tot = np.sum((gt_flat - np.mean(gt_flat)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    # Percentile errors
    errors = np.abs(pred_flat - gt_flat)
    p10 = np.percentile(errors, 10)
    p50 = np.percentile(errors, 50)
    p90 = np.percentile(errors, 90)
    p95 = np.percentile(errors, 95)
    p99 = np.percentile(errors, 99)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'correlation': float(correlation),
        'r2_score': float(r2_score),
        'error_p10': float(p10),
        'error_p50': float(p50),
        'error_p90': float(p90),
        'error_p95': float(p95),
        'error_p99': float(p99),
        'max_error': float(np.max(errors)),
        'min_error': float(np.min(errors))
    }
    
    return metrics


def test_model(model_dir, data_directory, mode='test'):
    """
    Test a trained model.
    
    Args:
        model_dir: Directory containing trained model
        data_directory: Directory containing test data
        mode: 'test' or 'val'
    """
    logging.info('='*80)
    logging.info('Model Testing')
    logging.info('='*80)
    logging.info(f'Model directory: {model_dir}')
    logging.info(f'Data directory: {data_directory}')
    logging.info(f'Mode: {mode}')
    
    # Load model
    logging.info('\nLoading model...')
    model = DeepStatisticalSolver(
        latent_dimension=20,
        hidden_layers=2,
        correction_updates=20,
        alpha=1e-2,
        non_linearity='leaky_relu',
        batch_size=500,
        name='physics_gcn',
        directory=model_dir,
        default_data_directory = data_directory
    )
    model.load_model(model_dir)

    # After: model.load_model(model_dir)

    # Verify model loaded correctly
    print("\n" + "="*80)
    print("VERIFYING MODEL LOADED CORRECTLY")
    print("="*80)

    # Check parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")

    if trainable_params == 0:
        print("❌ ERROR: Model has 0 parameters!")
        sys.exit(1)

    # Check weight statistics
    import numpy as np
    weight_values = [w.numpy().flatten() for w in model.trainable_weights]
    all_weights = np.concatenate(weight_values)

    print(f"\nWeight statistics:")
    print(f"  Mean: {all_weights.mean():.6f}")
    print(f"  Std: {all_weights.std():.6f}")
    print(f"  Min: {all_weights.min():.6f}")
    print(f"  Max: {all_weights.max():.6f}")
    print(f"  Non-zero weights: {np.count_nonzero(all_weights):,}")

    # Test prediction
    print("\nTesting prediction on dummy data...")
    dummy_A = tf.zeros([1, 10, 3], dtype=tf.float32)
    dummy_B = tf.zeros([1, 10, 4], dtype=tf.float32)
    output = model([dummy_A, dummy_B], training=False)

    if isinstance(output, dict) and 'final_prediction' in output:
        pred = output['final_prediction']
        print(f"  Model output shape: {pred.shape}")
        print(f"  Output mean: {pred.numpy().mean():.6f}")
        print(f"  Output std: {pred.numpy().std():.6f}")

    print("="*80 + "\n")

    logging.info('Model loaded successfully!')
    
    # Load data
    logging.info(f'\nLoading {mode} data...')
    A, B, U_gt = load_numpy_dataset(data_directory, mode)
    logging.info(f'Data shapes - A: {A.shape}, B: {B.shape}, U: {U_gt.shape}')
    
    # Get predictions
    logging.info('\nGenerating predictions...')
    outputs = model((A, B), training=False)
    U_pred = outputs['final_prediction'].numpy()
    
    # Compute metrics
    logging.info('\nComputing metrics...')
    metrics = compute_detailed_metrics(U_pred, U_gt)
    
    # Print metrics
    logging.info('\n' + '='*80)
    logging.info('EVALUATION RESULTS')
    logging.info('='*80)
    logging.info(f'  MSE:         {metrics["mse"]:.6e}')
    logging.info(f'  RMSE:        {metrics["rmse"]:.6f} m')
    logging.info(f'  MAE:         {metrics["mae"]:.6f} m')
    logging.info(f'  Correlation: {metrics["correlation"]:.6f}')
    logging.info(f'  R² Score:    {metrics["r2_score"]:.6f}')
    logging.info('\nError Percentiles:')
    logging.info(f'  10th:        {metrics["error_p10"]:.6f} m')
    logging.info(f'  50th:        {metrics["error_p50"]:.6f} m')
    logging.info(f'  90th:        {metrics["error_p90"]:.6f} m')
    logging.info(f'  95th:        {metrics["error_p95"]:.6f} m')
    logging.info(f'  99th:        {metrics["error_p99"]:.6f} m')
    logging.info(f'  Max:         {metrics["max_error"]:.6f} m')
    
    # Create output directory
    output_dir = os.path.join(model_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'metrics_{mode}.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f'\nMetrics saved to: {metrics_path}')
    
    # Save predictions
    pred_path = os.path.join(output_dir, f'predictions_{mode}.npz')
    np.savez(pred_path, U_pred=U_pred, U_gt=U_gt, A=A, B=B)
    logging.info(f'Predictions saved to: {pred_path}')
    
    # Generate plots
    logging.info('\nGenerating visualizations...')
    plot_path_pred = os.path.join(output_dir, f'predictions_{mode}.png')
    plot_predictions(U_pred, U_gt, plot_path_pred)
    
    plot_path_error = os.path.join(output_dir, f'error_distribution_{mode}.png')
    plot_error_distribution(U_pred, U_gt, plot_path_error)
    
    logging.info('\n' + '='*80)
    logging.info('Testing completed!')
    logging.info('='*80)


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test Physics-Integrated GCN')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--data_directory', type=str, required=True,
                        help='Directory containing test data')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'val'],
                        help='Evaluation mode')
    
    args = parser.parse_args()
    
    setup_logger()
    test_model(args.model_dir, args.data_directory, args.mode)


if __name__ == '__main__':
    main()