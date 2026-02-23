# GCN-Based State Estimation for Water Distribution Systems

> A Graph Convolutional Network (GCN) framework for semi-supervised state estimation, leakage detection, and inference in Water Distribution Systems (WDS).<br>
> **Author**: Soumyajit Banerjee

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Workflow Overview](#workflow-overview)
- [Module Reference](#module-reference)
  - [Root Scripts](#root-scripts)
  - [datasets/](#datasets)
  - [generate_datasets/](#generate_datasets)
  - [models/](#models)
  - [leakage/](#leakage)
  - [utils/](#utils)
  - [results/](#results)
- [Usage](#usage)
- [Key Concepts](#key-concepts)
- [Notes & Conventions](#notes--conventions)

---

## Overview

This project implements a **Graph Convolutional Network (GCN)** for state estimation in Water Distribution Systems (WDS). The core idea is **semi-supervised learning**: the model is trained using pressure/flow measurements from only **~10% of sensor nodes** and learns to generalize and infer states at **all nodes** in the network graph.

The pipeline consists of three major stages:

1. **Dataset Generation** — Hydraulic simulation data is generated from an `.inp` network file and exported to `.npy` format.
2. **GCN Training & Testing** — The GCN is trained in a semi-supervised fashion using the sparse sensor data, then evaluated.
3. **Inference & Leakage Detection** — The trained model is used to infer full-network states and detect leakages.

---

## Project Structure

```
GNN_StateEstimation_WDS-main/
│
├── main.py                  # Main driver: trains the GCN model
├── test_model.py            # Evaluates the trained GCN model
├── gcn_inference_xlsx.py    # Infers all-node states from 10% sensor data; exports to Excel
│
├── datasets/
│   └── asnet2_5/
│       ├── problem.py       # Problem definition: graph topology, node/edge attributes
│       └── *.npy            # Pre-generated numpy training/validation data arrays
│
├── generate_datasets/
│   ├── *.inp                # EPANET network input file (WDS topology & parameters)
│   ├── ConvertData.py       # Converts EPANET simulation output to .npy training format
│   └── generate_data_eps.py # Runs hydraulic simulations with epsilon perturbations
│
├── models/
│   ├── layers.py            # Custom GCN layer implementations (e.g., GraphConv)
│   └── models.py            # GCN model architecture definition
│
├── leakage/
│   ├── train.py             # Trains the leakage detection model
│   ├── predict.py           # Runs leakage prediction on new/test data
│   └── src/
│       ├── config.py        # Configuration: hyperparameters, paths, thresholds
│       ├── data_loader.py   # Loads and preprocesses data for leakage module
│       ├── detector.py      # Core leakage detection logic and anomaly scoring
│       ├── models.py        # Model architecture for leakage detection
│       └── persistence.py   # Saves/loads trained leakage models to/from disk
│
├── utils/
│   └── data_loader.py       # Shared utility: loads .npy datasets, builds graph objects
│
└── results/
    └── *.pkl / *.pt / *.csv # Saved model weights, training logs, and result exports
```

---

## Installation & Requirements

### Prerequisites

- Python 3.8+
- [EPANET](https://www.epa.gov/water-research/epanet) (for generating new hydraulic simulation data)

### Python Dependencies

Install via pip:

```bash
pip install torch torch-geometric numpy scipy pandas openpyxl matplotlib
```

| Package | Purpose |
|---|---|
| `torch` | Deep learning backend |
| `torch-geometric` | GCN layers and graph data utilities |
| `numpy` | Numerical array operations |
| `scipy` | Sparse matrix support for graph adjacency |
| `pandas` / `openpyxl` | Excel export in `gcn_inference_xlsx.py` |
| `matplotlib` | Plotting training curves and results |

> **Note:** PyTorch Geometric has specific install requirements depending on your CUDA version. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for the correct command.

---

## Workflow Overview

```
[EPANET .inp file]
        │
        ▼
generate_datasets/generate_data_eps.py   ← Simulate hydraulic scenarios
        │
        ▼
generate_datasets/ConvertData.py         ← Convert to .npy arrays
        │
        ▼
datasets/asnet2_5/*.npy                  ← Training data ready
        │
        ▼
main.py                                  ← Train GCN (semi-supervised, 10% nodes)
        │
        ├──► results/                    ← Saved model weights & training logs
        │
        ▼
test_model.py                            ← Evaluate GCN on test set
        │
        ▼
gcn_inference_xlsx.py                    ← Infer all-node pressures → Excel output
        │
        ▼
leakage/train.py                         ← Train leakage detector on GCN outputs
        │
        ▼
leakage/predict.py                       ← Detect leakages in new scenarios
```

---

## Module Reference

### Root Scripts

#### `main.py` — GCN Training Driver

The primary entry point. Handles:

- Loading the graph and dataset via `utils/data_loader.py`
- Constructing the GCN model from `models/models.py`
- Running the semi-supervised training loop (10% labeled sensor nodes)
- Saving trained model weights and training metrics to `results/`

## Usage

You can train the model in either supervised or semi-supervised modes depending on your dataset and experimental setup. Use the examples below to run `main.py` with the appropriate flags.

### Supervised Learning:

To train the model using standard supervised learning, point the `--data_directory` to your dataset and omit the unsupervised flag:

```bash
python main.py \
  --data_directory datasets/asnet2_5/ \
  --max_iter 80000 \
  --batch_size 500 \
  --learning_rate 1e-3 \
  --latent_dimension 20 \
  --hidden_layers 2 \
  --correction_updates 20 \
  --alpha 0.5 \
  --non_linearity leaky_relu \
  --track_validation 1000 \
  --gpu 0
```

### Semi-supervised learning:

To enable semi-supervised learning, append the --unsupervised flag to your command. Be sure to update the --data_directory to match the number of measurement locations you are using.

#### 1 measurement location

```bash
python main.py \
  --data_directory datasets/asnet2_1/ \
  --max_iter 80000 \
  --batch_size 500 \
  --learning_rate 1e-3 \
  --latent_dimension 20 \
  --hidden_layers 2 \
  --correction_updates 20 \
  --alpha 0.5 \
  --non_linearity leaky_relu \
  --track_validation 1000 \
  --gpu 0 \
  --unsupervised
```

#### 5 measurement location:

```bash
python main.py \
  --data_directory datasets/asnet2_5/ \
  --max_iter 80000 \
  --batch_size 500 \
  --learning_rate 1e-3 \
  --latent_dimension 20 \
  --hidden_layers 2 \
  --correction_updates 20 \
  --alpha 0.5 \
  --non_linearity leaky_relu \
  --track_validation 1000 \
  --gpu 0 \
  --unsupervised
```


Key hyperparameters (typically configured at the top of the file or via argparse):

| Parameter | Description |
|---|---|
| `epochs` | Number of training iterations |
| `lr` | Learning rate |
| `hidden` | Hidden layer dimension |
| `dropout` | Dropout rate |
| `sensor_ratio` | Fraction of nodes used as labeled sensors (default: 0.10) |

---

#### `test_model.py` — GCN Evaluation

Loads a pre-trained GCN model from `results/` and evaluates it on the test split. Reports metrics such as MAE, RMSE, and R² for pressure/flow state estimation across all nodes.

**Usage:**
```bash
python test_model.py
```

---

#### `gcn_inference_xlsx.py` — Full-Network Inference → Excel Export

Takes the trained GCN model and runs inference to predict states at **all nodes** (not just the 10% sensors). Exports the inferred pressure and flow values as a structured `.xlsx` spreadsheet for further analysis or reporting.

**Usage:**
```bash
python gcn_inference_xlsx.py
```

**Output:** An Excel file (saved to `results/` or current directory) with columns for node ID, predicted pressure, predicted flow, and optionally ground truth values.

---

### `datasets/`

#### `asnet2_5/problem.py`

Defines the specific WDS problem instance for the `asnet2_5` network. Contains:

- Graph topology (adjacency structure, node/pipe connectivity)
- Node feature definitions (e.g., elevation, demand pattern)
- Edge feature definitions (e.g., pipe diameter, length, roughness)
- Train/validation/test split indices
- Sensor node mask (the 10% labeled nodes)

#### `*.npy` Files

Pre-generated NumPy binary arrays storing:

| File pattern | Contents |
|---|---|
| `*_features.npy` | Node feature matrix `[num_scenarios, num_nodes, num_features]` |
| `*_labels.npy` | Ground truth pressures/flows `[num_scenarios, num_nodes]` |
| `*_adj.npy` | Adjacency matrix or edge index for the network graph |

---

### `generate_datasets/`

#### `*.inp` — EPANET Network File

Defines the physical Water Distribution System: pipes, junctions, reservoirs, pumps, valves, demand patterns, and hydraulic parameters. This is the source of truth for the network topology.

#### `generate_data_eps.py` — Hydraulic Simulation

Uses EPANET (via `wntr` or `epynet`) to run multiple hydraulic simulations with **epsilon-perturbed** demand conditions. Each simulation produces a unique pressure/flow scenario, creating a diverse training dataset.

**Usage:**
```bash
python generate_datasets/generate_data_eps.py
```

Configurable parameters include: number of scenarios, perturbation magnitude (epsilon), random seed, and output path.

#### `ConvertData.py` — Data Conversion

Converts raw EPANET simulation outputs into the `.npy` format expected by the GCN training pipeline. Handles unit normalization, node indexing alignment, and train/val/test splitting.

**Usage:**
```bash
python generate_datasets/ConvertData.py
```

---

### `models/`

#### `layers.py` — GCN Layer Definitions

Implements custom graph convolutional layers. Typical contents:

- **`GraphConvolution`**: Standard spectral GCN layer performing `H' = σ(D⁻¹/²AD⁻¹/²HW)` where A is the adjacency matrix, H is node features, and W is a learnable weight matrix.
- Optional: attention-based variants or edge-feature-aware layers.

#### `models.py` — GCN Model Architecture

Assembles the full GCN model using layers from `layers.py`. Architecture typically consists of:

- Input projection layer
- 2–4 stacked `GraphConvolution` layers with ReLU activations and dropout
- Output regression layer predicting pressure/flow at each node

The model is designed for **semi-supervised regression**: during training, the loss is computed only on the labeled sensor nodes; during inference, predictions are produced for all nodes.

---

### `leakage/`

A self-contained sub-module for leakage detection, built on top of GCN-inferred states.

#### `train.py`

Trains the leakage detection model using normal and leakage-scenario data. Calls into `src/` for data loading, model definition, and model persistence.

**Usage:**
```bash
python leakage/train.py
```

#### `predict.py`

Loads a trained leakage model and runs prediction on new scenario data, outputting leakage probability scores or binary leakage/no-leakage decisions per node or pipe segment.

**Usage:**
```bash
python leakage/predict.py
```

#### `src/config.py`

Centralized configuration for the leakage module. Contains:

- File paths for data and saved models
- Model hyperparameters (hidden size, learning rate, epochs)
- Detection thresholds (e.g., anomaly score cutoff)
- Feature selection flags

#### `src/data_loader.py`

Loads and preprocesses leakage scenario data. Handles merging of GCN-inferred states with ground truth leakage labels, feature normalization, and batching for the leakage model.

#### `src/detector.py`

Core leakage detection logic. Implements:

- Residual computation: difference between GCN-predicted and measured pressures
- Anomaly scoring algorithm
- Spatial leakage localization (identifying which node/pipe has a leak)

#### `src/models.py`

Defines the leakage detection model architecture (e.g., MLP or LSTM operating on residual pressure sequences).

#### `src/persistence.py`

Handles serialization and deserialization of trained leakage models using `torch.save` / `torch.load` or `pickle`. Ensures reproducible model loading for prediction.

---

### `utils/`

#### `data_loader.py`

Shared utility functions used across the main pipeline:

- **`load_data(dataset)`**: Loads `.npy` arrays and constructs a `torch_geometric.data.Data` graph object.
- **`normalize_features()`**: Applies min-max or z-score normalization to node features.
- **`get_sensor_mask()`**: Returns the boolean mask of the 10% sensor nodes used for semi-supervised training.
- **`train_val_test_split()`**: Splits scenario indices into training, validation, and test sets.

---

### `results/`

Stores all outputs from `main.py` and `test_model.py`:

| File type | Contents |
|---|---|
| `*.pt` | Saved PyTorch model state dictionaries (`torch.save`) |
| `*.pkl` | Pickled training history (loss curves, metric logs) |
| `*.csv` | Per-node prediction vs. ground truth tabular results |
| `*.xlsx` | Excel exports from `gcn_inference_xlsx.py` |

---

## Usage

### End-to-End: From Raw Network to Inference

**Step 1 — Generate training data** (skip if `.npy` files already exist):
```bash
python generate_datasets/generate_data_eps.py
python generate_datasets/ConvertData.py
```

**Step 2 — Train the GCN:**
```bash
python main.py
```

**Step 3 — Evaluate the GCN:**
```bash
python test_model.py
```

**Step 4 — Export full-network inference to Excel:**
```bash
python gcn_inference_xlsx.py
```

**Step 5 — Train leakage detector:**
```bash
python leakage/train.py
```

**Step 6 — Predict leakages on new data:**
```bash
python leakage/predict.py
```

---

## Key Concepts

**Semi-Supervised GCN:** Only 10% of nodes have sensor readings (labeled). The GCN propagates information across the graph structure so that unlabeled nodes benefit from their neighbors' measurements, enabling accurate full-network state estimation with minimal instrumentation.

**State Estimation in WDS:** The goal is to determine the hydraulic state (pressure at junctions, flow in pipes) across the entire network given sparse sensor observations — analogous to power system state estimation.

**Graph Construction:** The WDS is modeled as a graph where nodes are junctions/reservoirs and edges are pipes/pumps/valves. Node features may include elevation and demand; edge features may include pipe length, diameter, and roughness coefficient.

**Leakage Detection:** A leakage causes a pressure drop in the vicinity of the leak point. By comparing GCN-predicted pressures (trained on normal conditions) against actual observations, anomalies can be identified and localized.

---

## Notes & Conventions

- All dataset paths are relative to the project root. Run scripts from `GNN_StateEstimation_WDS-main/`.
- The `datasets/asnet2_5/` folder name reflects the network name (`asnet2_5`). Adding a new network requires creating a corresponding folder and `problem.py`.
- Model checkpoints in `results/` are overwritten on each training run unless filenames are versioned in `main.py`.
- The leakage module (`leakage/`) is designed to be modular — `src/config.py` is the primary file to edit when adapting it to a new dataset or threshold.
- EPANET `.inp` files can be edited with [EPANET 2.2](https://www.epa.gov/water-research/epanet) or [WNTR](https://wntr.readthedocs.io/) in Python.
