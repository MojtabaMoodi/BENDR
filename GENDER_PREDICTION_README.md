---
created time: 2025-06-01T20:04
updated time: 2025-06-01T21:08
---
# Gender Prediction with BENDR

This modification extends the BENDR (Brain Embedding Neural Data Representation) framework to predict gender from EEG data instead of sleep stage classification.

## Overview

The original BENDR system was designed for sleep stage classification using the Sleep-EDF dataset. This modification:

1. **Replaces** sleep stage classification with binary gender classification (Male/Female)
2. **Automatically extracts** gender information from EDF file headers
3. **Reuses** the same powerful BENDR model architectures for the new task
4. **Creates** a new dataset configuration specifically for gender prediction

## Key Changes Made

### 1. Custom Gender Data Loader (`dn3_ext.py`)
- **LoaderGenderEDF**: New class that extracts gender from EDF headers using `pyedflib`
- Automatically creates artificial events based on gender labels
- Maps: Female → Class 0, Male → Class 1

### 2. New Configuration Files
- **`configs/gender_prediction.yml`**: Main experiment configuration
- **`configs/gender_datasets.yml`**: Dataset-specific configuration for gender classification
- Updated **`configs/metrics.yml`**: Added metrics for binary classification

### 3. Updated Dependencies
- Added **`pyedflib>=0.1.26`** to `requirements.txt` for reading EDF headers

### 4. Modified Training Script
- **`downstream.py`**: Updated default configuration to use gender prediction
- **`utils.py`**: Added support for the new gender dataset loader

## Usage

### Basic Usage
```bash
# Run with default gender prediction configuration
python downstream.py
```

### Advanced Usage
```bash
# LinearHead BENDR with frozen encoder
python downstream.py linear --ds-config configs/gender_prediction.yml --freeze-encoder --results-filename gender_results.xlsx

# Full BENDR model
python downstream.py BENDR --ds-config configs/gender_prediction.yml --results-filename gender_results.xlsx

# Random initialization comparison
python downstream.py linear --ds-config configs/gender_prediction.yml --random-init --results-filename gender_results.xlsx
```

### Configuration Options

The following configurations are available in `downstream.py`:

```python
# Gender Config No. 1: Full BENDR with pre-trained weights
sys.argv += ['BENDR', '--ds-config', 'configs/gender_prediction.yml', '--results-filename', 'gender_results.xlsx', '--precision', 'fp32']

# Gender Config No. 2: LinearHead BENDR (lighter model)
sys.argv += ['linear', '--ds-config', 'configs/gender_prediction.yml', '--results-filename', 'gender_results.xlsx', '--precision', 'fp32']

# Gender Config No. 3: Full BENDR with random initialization
sys.argv += ['BENDR', '--ds-config', 'configs/gender_prediction.yml', '--random-init', '--results-filename', 'gender_results.xlsx', '--precision', 'fp16']

# Gender Config No. 4: Full BENDR with frozen encoder
sys.argv += ['BENDR', '--ds-config', 'configs/gender_prediction.yml', '--freeze-encoder', '--results-filename', 'gender_results.xlsx', '--precision', 'fp32']

# Gender Config No. 5: LinearHead BENDR with random initialization
sys.argv += ['linear', '--ds-config', 'configs/gender_prediction.yml', '--random-init', '--results-filename', 'gender_results.xlsx', '--precision', 'fp16']

# Gender Config No. 6: LinearHead BENDR with frozen encoder (default)
sys.argv += ['linear', '--ds-config', 'configs/gender_prediction.yml', '--freeze-encoder', '--results-filename', 'gender_results.xlsx', '--precision', 'fp32']
```

## Testing

Run the test script to verify the implementation:

```bash
python test_gender_loader.py
```

This will:
1. Test the gender extraction from EDF files
2. Verify dataset creation with the new configuration
3. Check data shapes and target distributions

## Model Architectures

The system supports two model architectures:

### 1. **LinearHeadBENDR** (`linear`)
- Encoder + Adaptive pooling + Linear classifier
- Lighter model, faster training
- Good baseline performance

### 2. **BENDRClassification** (`BENDR`) 
- Full encoder + contextualizer (transformer) + classifier
- More powerful model, slower training
- Better performance on complex tasks

## Data Flow

1. **EDF Files** → **LoaderGenderEDF** → **Gender Labels**
2. **Raw EEG** → **BENDR Encoder** → **Feature Representations**
3. **Features** → **Classification Head** → **Gender Predictions**

## Results

Results are saved in Excel format with the following metrics:
- **Accuracy**: Overall classification accuracy
- **Balanced Accuracy (BAC)**: Accounts for class imbalance
- **AUROC**: Area under ROC curve for binary classification

## Files Changed

- `dn3_ext.py`: Added `LoaderGenderEDF` class
- `utils.py`: Added gender loader support
- `downstream.py`: Updated default configurations
- `requirements.txt`: Added `pyedflib` dependency
- `configs/gender_prediction.yml`: New experiment config
- `configs/gender_datasets.yml`: New dataset config
- `configs/metrics.yml`: Added gender classification metrics

## Dataset Requirements

- **Sleep-EDF Dataset**: Must be available at `/Volumes/Data/SSC/sleep-cassette/`
- **EDF Files**: Must contain gender information in headers
- **Pre-trained Weights**: BENDR encoder and contextualizer weights at:
  - `/Volumes/Data/encoder.pt`
  - `/Volumes/Data/contextualizer.pt`

## Expected Performance

Gender classification from EEG is a challenging task. Expected performance:
- **Random Baseline**: 50% accuracy (binary classification)
- **Expected Range**: 55-75% accuracy depending on:
  - Data quality and quantity
  - Model architecture choice
  - Training configuration
  - Subject-specific variations

## Troubleshooting

### Common Issues

1. **"Unknown gender" errors**: Some EDF files may not have gender info in headers
2. **Path errors**: Update file paths in configurations to match your setup
3. **Memory errors**: Reduce batch size in `configs/gender_datasets.yml`
4. **CUDA errors**: Use `--precision fp16` or CPU-only mode

### Debug Mode

To debug data loading issues:
```python
import mne
mne.set_log_level(True)  # Enable verbose MNE logging
```

## Future Improvements

1. **Multi-modal Features**: Combine EEG with demographic data
2. **Cross-dataset Validation**: Test on other EEG datasets
3. **Interpretability**: Add attention visualization for gender-specific patterns
4. **Data Augmentation**: Improve generalization with synthetic data
5. **Uncertainty Quantification**: Add confidence estimates to predictions 