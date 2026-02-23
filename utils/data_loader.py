import os
import numpy as np
import logging


def load_numpy_dataset(data_directory, mode='train'):
    """
    Load numpy dataset from directory.
    
    Args:
        data_directory: Path to directory containing numpy files
        mode: One of 'train', 'val', 'test'
    
    Returns:
        A: Edge features [n_samples, n_edges, d_in_A + 2]
        B: Node features [n_samples, n_nodes, d_in_B]
        U: Ground truth heads [n_samples, n_nodes, d_out]
    
    Expected file naming conventions:
        - {mode}_A.npy or A_{mode}.npy: Edge features
        - {mode}_B.npy or B_{mode}.npy: Node features
        - {mode}_U.npy or U_{mode}.npy or {mode}_labels.npy: Ground truth
    """
    
    # Normalize path
    data_directory = data_directory.rstrip('/')
    
    # Try different naming conventions
    file_patterns = [
        # Pattern 1: mode_X.npy
        (f'{mode}_A.npy', f'{mode}_B.npy', f'{mode}_U.npy'),
        # Pattern 2: X_mode.npy
        (f'A_{mode}.npy', f'B_{mode}.npy', f'U_{mode}.npy'),
        # Pattern 3: mode_labels.npy for U
        (f'{mode}_A.npy', f'{mode}_B.npy', f'{mode}_labels.npy'),
        # Pattern 4: X_mode.npy with labels
        (f'A_{mode}.npy', f'B_{mode}.npy', f'labels_{mode}.npy'),
    ]
    
    A_file = None
    B_file = None
    U_file = None
    
    # Find files that exist
    for a_pattern, b_pattern, u_pattern in file_patterns:
        a_path = os.path.join(data_directory, a_pattern)
        b_path = os.path.join(data_directory, b_pattern)
        u_path = os.path.join(data_directory, u_pattern)
        
        if os.path.exists(a_path) and os.path.exists(b_path) and os.path.exists(u_path):
            A_file = a_path
            B_file = b_path
            U_file = u_path
            break
    
    if A_file is None:
        # List available files for debugging
        available_files = os.listdir(data_directory)
        raise FileNotFoundError(
            f"Could not find {mode} data files in {data_directory}\n"
            f"Available files: {available_files}\n"
            f"Expected one of these patterns:\n"
            f"  - {mode}_A.npy, {mode}_B.npy, {mode}_U.npy\n"
            f"  - A_{mode}.npy, B_{mode}.npy, U_{mode}.npy"
        )
    
    # Load data
    logging.info(f'Loading data from {data_directory}')
    logging.info(f'  A: {os.path.basename(A_file)}')
    logging.info(f'  B: {os.path.basename(B_file)}')
    logging.info(f'  U: {os.path.basename(U_file)}')
    
    A = np.load(A_file)
    B = np.load(B_file)
    U = np.load(U_file)
    
    # Ensure correct shapes
    # A should be [n_samples, n_edges, features]
    # B should be [n_samples, n_nodes, features]
    # U should be [n_samples, n_nodes, 1] or [n_samples, n_nodes]
    
    # If U is 2D, expand to 3D
    if len(U.shape) == 2:
        U = np.expand_dims(U, axis=-1)
    
    logging.info(f'Loaded shapes - A: {A.shape}, B: {B.shape}, U: {U.shape}')
    
    return A, B, U


def load_normalization_stats(data_directory):
    """
    Load normalization statistics for inputs.
    
    Args:
        data_directory: Path to directory containing stats files
    
    Returns:
        Dictionary with keys 'A_mean', 'A_std', 'B_mean', 'B_std'
    """
    stats = {}
    
    stat_files = {
        'A_mean': 'A_mean.npy',
        'A_std': 'A_std.npy',
        'B_mean': 'B_mean.npy',
        'B_std': 'B_std.npy'
    }
    
    for key, filename in stat_files.items():
        filepath = os.path.join(data_directory, filename)
        if os.path.exists(filepath):
            stats[key] = np.load(filepath)
        else:
            logging.warning(f'Normalization file not found: {filepath}')
            stats[key] = None
    
    return stats


def compute_normalization_stats(A, B, save_dir=None):
    """
    Compute mean and std for normalization.
    
    Args:
        A: Edge features [n_samples, n_edges, features]
        B: Node features [n_samples, n_nodes, features]
        save_dir: Optional directory to save stats
    
    Returns:
        Dictionary with normalization statistics
    """
    # Compute along samples and spatial dimensions, keep feature dimension
    A_mean = np.mean(A, axis=(0, 1), keepdims=False)
    A_std = np.std(A, axis=(0, 1), keepdims=False)
    
    B_mean = np.mean(B, axis=(0, 1), keepdims=False)
    B_std = np.std(B, axis=(0, 1), keepdims=False)
    
    # Avoid division by zero
    A_std = np.where(A_std == 0, 1.0, A_std)
    B_std = np.where(B_std == 0, 1.0, B_std)
    
    stats = {
        'A_mean': A_mean,
        'A_std': A_std,
        'B_mean': B_mean,
        'B_std': B_std
    }
    
    # Save if directory provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for key, value in stats.items():
            filepath = os.path.join(save_dir, f'{key}.npy')
            np.save(filepath, value)
            logging.info(f'Saved {key} to {filepath}')
    
    return stats


def normalize_data(A, B, stats):
    """
    Normalize data using provided statistics.
    
    Args:
        A: Edge features
        B: Node features
        stats: Dictionary with 'A_mean', 'A_std', 'B_mean', 'B_std'
    
    Returns:
        Normalized A and B
    """
    A_normalized = (A - stats['A_mean']) / stats['A_std']
    B_normalized = (B - stats['B_mean']) / stats['B_std']
    
    return A_normalized, B_normalized


def load_and_prepare_data(data_directory, mode='train', normalize=False):
    """
    Load and optionally normalize data.
    
    Args:
        data_directory: Path to data directory
        mode: 'train', 'val', or 'test'
        normalize: Whether to normalize the data
    
    Returns:
        A, B, U (normalized if requested)
    """
    # Load data
    A, B, U = load_numpy_dataset(data_directory, mode)
    
    # Optionally normalize
    if normalize:
        stats = load_normalization_stats(data_directory)
        
        if all(stats.values()):
            A, B = normalize_data(A, B, stats)
            logging.info('Data normalized using saved statistics')
        else:
            logging.warning('Normalization requested but stats not found. Using raw data.')
    
    return A, B, U
