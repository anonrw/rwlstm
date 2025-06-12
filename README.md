# RW-LSTM: Read-Write Long Short-Term Memory Networks

A PyTorch implementation of Read-Write LSTM networks for time series prediction with adaptive data correction capabilities.

## Overview

This implementation extends traditional LSTM networks with the ability to adaptively modify input data during training based on gradient information. The RW-LSTM can "read" from and "write" to the input data during the training process.

### Key Features

- **Adaptive Data Correction**: Modifies input sequences during training using gradient-based updates
- **Multiple Variants**: Supports different gradient scaling functions (ReLU, Sigmoid, None)
- **Moving Average Tracking**: Uses moving averages for stable gradient updates
- **Configurable Architecture**: Flexible model parameters and training configurations
- **Visualization Tools**: Built-in plotting and comparison functionality

## Model Variants

- **LSTM**: Baseline LSTM without data correction
- **RW**: Basic RW-LSTM with raw gradient updates
- **RW-ReLU**: RW-LSTM with ReLU-scaled gradients
- **RW-Sigmoid**: RW-LSTM with Sigmoid-scaled gradients

## Installation

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

## Usage

### Basic Usage

```python
from rw_lstm import train_rw_lstm, Config

# Load your time series data (1D numpy array)
data = your_time_series_data

# Create configuration
config = Config()

# Train model
model, scaler, train_dataset, val_dataset, train_losses, val_losses, split_idx = train_rw_lstm(
    data, model_type='rw', config=config
)
```

### Configuration Options

```python
config = Config()

# Data parameters
config.window_size = 3              # Input sequence length
config.train_split_size = 0.9       # Train/validation split ratio

# Model parameters
config.input_size = 1               # Input feature size
config.output_size = 1              # Output size
config.num_lstm_layers = 1          # Number of LSTM layers
config.lstm_size = 20               # LSTM hidden size
config.dropout = 0.2                # Dropout rate

# Training parameters
config.batch_size = 32              # Batch size
config.num_epochs = 50              # Training epochs
config.learning_rate = 1e-2         # Learning rate
config.scheduler_step_size = 40     # Learning rate scheduler step

# RW-LSTM specific parameters
config.correction_rate = 0.001      # Data correction learning rate
config.epoch_threshold = 45         # Epoch to start data correction
config.moving_avg_decay = 0.9       # Moving average decay rate
```

### Complete Example

```python
import numpy as np
from rw_lstm import train_rw_lstm, Config

# Generate sample data
np.random.seed(42)
t = np.linspace(0, 50, 500)
data = np.sin(0.1 * t) + 0.1 * np.random.randn(500)

# Configure model
config = Config()
config.num_epochs = 100
config.window_size = 5

# Train different variants
models = ['lstm', 'rw', 'rw_relu', 'rw_sigmoid']
results = {}

for model_type in models:
    print(f"Training {model_type}...")
    model, scaler, _, _, _, _, _ = train_rw_lstm(data, model_type, config)
    results[model_type] = model
```

## Data Format

The model expects 1D time series data as a NumPy array:

```python
# Example data formats
data = np.array([1.2, 1.5, 1.8, 2.1, ...])  # Simple 1D array
data = df['value'].values                     # From pandas DataFrame
```

## Key Classes and Functions

### Core Classes

- **`RWLSTMModel`**: Main LSTM model with data correction capability
- **`Config`**: Configuration management
- **`MovingAverageTracker`**: Tracks moving averages for gradient updates
- **`Normalizer`**: Z-score normalization for input data
- **`TimeSeriesDataset`**: PyTorch dataset for windowed time series

### Main Functions

- **`train_rw_lstm(data, model_type, config)`**: Main training function
- **`evaluate_model(model, dataset, scaler, config)`**: Model evaluation
- **`plot_results(...)`**: Visualization of results
- **`prepare_data(data, config)`**: Data preprocessing and windowing

## Data Correction Mechanism

The RW-LSTM implements data correction through:

1. Computes gradients of loss with respect to input data
2. Applies optional scaling functions (ReLU/Sigmoid) to gradients
3. Updates moving average of squared gradients
4. Modifies input sequences using the computed updates

Update formula:
```
update = -correction_rate * scaled_gradients / sqrt(moving_avg + epsilon)
```

## Output

The training function returns:
- `model`: Trained PyTorch model
- `scaler`: Data normalizer for inverse transformation
- `train_dataset`: Training dataset with potentially corrected data
- `val_dataset`: Validation dataset
- `train_losses`: Training loss history
- `val_losses`: Validation loss history
- `split_idx`: Index where data was split into train/validation

## Visualization

The implementation includes plotting functions that show:
- Original vs corrected data (for RW variants)
- Training predictions
- Validation predictions
- Model comparison plots

## File Structure

```
your_project/
├── rw_lstm.py          # Main implementation
├── README.md           # This file
└── data/              # Your time series data
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Running the Code

To run the complete example with NAB data:

```python
if __name__ == "__main__":
    results = main()
```

This will train all model variants and display comparison results.

## Notes

- Data correction starts after `epoch_threshold` to allow initial convergence
- The `correction_rate` should be small (0.001-0.01) to maintain training stability
- Larger `window_size` provides more context but increases computational cost
- The moving average decay rate affects the stability of gradient updates
- The present code is set to work for input_size=1, for other values the call of update_data_with_gradients should be modified
