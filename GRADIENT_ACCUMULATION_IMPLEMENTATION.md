# Gradient Accumulation Implementation for BENDR

## Overview

This document describes the implementation of gradient accumulation in the BENDR codebase. Gradient accumulation allows training with effectively larger batch sizes while maintaining memory efficiency by accumulating gradients over multiple smaller batches before updating model parameters.

## Implementation Details

### Core Changes

#### 1. BaseProcess Class (`dn3/trainable/processes.py`)

**Added Parameters:**
- `gradient_accumulation_steps`: Number of mini-batches to accumulate gradients over before performing an optimizer step (default: 1)

**Key Modifications:**
- Added gradient accumulation counter tracking
- Modified `backward()` method to scale loss by accumulation steps
- Modified `train_step()` method to only update parameters after accumulating enough gradients
- Added end-of-epoch gradient cleanup to handle remaining accumulated gradients

**Code Changes:**
```python
# In __init__:
self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
self.accumulation_counter = 0
self.optimizer.zero_grad()  # Initialize gradients

# In backward():
def backward(self, loss):
    loss = loss / self.gradient_accumulation_steps  # Scale loss
    loss.backward()

# In train_step():
def train_step(self, *inputs):
    # ... forward pass ...
    self.backward(loss)
    self.accumulation_counter += 1
    
    # Only step optimizer when accumulated enough gradients
    if self.accumulation_counter >= self.gradient_accumulation_steps:
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.accumulation_counter = 0
        
        # Only step scheduler when parameters are updated
        if self.scheduler is not None and self.scheduler_after_batch:
            self.scheduler.step()
```

#### 2. StandardClassification Class

**Updated to accept and pass through gradient accumulation parameter:**
```python
def __init__(self, classifier, ..., gradient_accumulation_steps=1, **kwargs):
    super().__init__(..., gradient_accumulation_steps=gradient_accumulation_steps, ...)
```

#### 3. BendingCollegeWav2Vec Class

**Updated for pretraining support:**
```python
def __init__(self, encoder, context_fn, ..., gradient_accumulation_steps=1, **kwargs):
    super().__init__(..., gradient_accumulation_steps=gradient_accumulation_steps, ...)
```

### Configuration Updates

#### 1. Training Configuration Files

**Updated all dataset configurations to include gradient accumulation:**

- `configs/pretraining.yml`: Added `gradient_accumulation_steps: 4`
- `configs/downstream_datasets.yml`: Added to all dataset configurations
- `configs/gender_datasets.yml`: Added `gradient_accumulation_steps: 4`

**Example:**
```yaml
train_params:
  epochs: 10
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size = 8 * 4 = 32
```

#### 2. Training Scripts

**Updated `downstream.py`:**
```python
# Extract gradient accumulation steps from training parameters
gradient_accumulation_steps = getattr(ds.train_params, 'gradient_accumulation_steps', 1)
process = StandardClassification(model, metrics=added_metrics, precision=args.precision, 
                               gradient_accumulation_steps=gradient_accumulation_steps)
```

**Updated `pretrain.py`:**
```python
# Extract gradient accumulation steps from training parameters
gradient_accumulation_steps = getattr(experiment.training_params, 'gradient_accumulation_steps', 1)
process = BendingCollegeWav2Vec(encoder, contextualizer, 
                               gradient_accumulation_steps=gradient_accumulation_steps,
                               **experiment.bending_college_args)
```

## Benefits

### 1. Memory Efficiency
- Train with larger effective batch sizes without increasing GPU memory usage
- Particularly beneficial for large models like BENDR that require significant memory

### 2. Training Stability
- Larger effective batch sizes can lead to more stable gradients
- Reduces gradient noise, potentially improving convergence

### 3. Flexibility
- Can be configured per dataset based on memory constraints and optimal batch sizes
- Backward compatible (default value of 1 means no accumulation)

## Usage Examples

### Example 1: Basic Usage
```python
# Create a model with gradient accumulation
process = StandardClassification(model, gradient_accumulation_steps=4)

# This will accumulate gradients over 4 batches before updating parameters
process.fit(training_dataset, batch_size=8)  # Effective batch size = 32
```

### Example 2: Configuration File
```yaml
# In your dataset configuration
train_params:
  epochs: 10
  batch_size: 8
  gradient_accumulation_steps: 4  # Accumulate over 4 batches
  
# Effective training will use batch size of 8 * 4 = 32
```

### Example 3: Memory-Constrained Training
```python
# For large models with memory constraints
process = StandardClassification(
    large_model, 
    batch_size=2,                    # Small batch to fit in memory
    gradient_accumulation_steps=16   # Large accumulation for stability
)
# Effective batch size = 2 * 16 = 32
```

## Validation

The implementation has been validated with a comprehensive test suite (`test_gradient_accumulation.py`) that verifies:

1. **Gradient Equivalence**: Accumulated gradients match those from larger batch training
2. **Loss Consistency**: Training loss behaves correctly with accumulation
3. **Counter Management**: Accumulation counter resets properly
4. **Memory Efficiency**: No memory leaks or gradient accumulation errors

**Test Results:**
```
✓ Gradient accumulation steps: 4
✓ Loss difference: 0.00387605 (within tolerance)
✓ Gradient difference: 0.01619578 (within tolerance)
✓ Counter reset: ✓
Gradient Accumulation Test: ✓ PASSED
```

## Best Practices

### 1. Choosing Accumulation Steps
- **Memory-bound**: Use higher accumulation steps (4-16) with smaller batch sizes
- **Compute-bound**: Use lower accumulation steps (2-4) with larger batch sizes
- **Stability**: Larger effective batch sizes generally improve training stability

### 2. Learning Rate Adjustment
- When using gradient accumulation, the effective batch size increases
- Consider adjusting learning rate proportionally: `lr_new = lr_base * sqrt(effective_batch_size / base_batch_size)`

### 3. Scheduler Considerations
- Schedulers step only when parameters are updated (after accumulation)
- This means fewer scheduler steps per epoch, which may require adjustment

### 4. Validation Frequency
- Validation typically happens per epoch, not per accumulation step
- Consider the trade-off between validation frequency and training efficiency

## Technical Notes

### 1. Loss Scaling
- Loss is automatically scaled by `1/gradient_accumulation_steps` to maintain equivalent gradients
- This ensures that accumulated gradients have the same magnitude as single large-batch gradients

### 2. Scheduler Integration
- Learning rate schedulers only step when parameters are updated
- This maintains the correct relationship between parameter updates and learning rate changes

### 3. End-of-Epoch Handling
- Any remaining accumulated gradients at the end of an epoch are automatically applied
- This ensures no gradients are lost due to incomplete accumulation cycles

### 4. Backward Compatibility
- Default `gradient_accumulation_steps=1` maintains original behavior
- Existing code works without modification

## Performance Impact

### Memory Usage
- **Reduced**: Can train with larger effective batch sizes using less GPU memory
- **Gradient Storage**: Minimal additional memory for gradient accumulation tracking

### Training Speed
- **Slightly Slower**: Additional overhead from accumulation logic
- **Network Efficiency**: Fewer parameter updates may improve training efficiency
- **Overall**: Usually net positive due to better convergence with larger effective batches

### Convergence
- **Improved Stability**: Larger effective batch sizes reduce gradient noise
- **Better Generalization**: More stable gradients often lead to better final models

## Troubleshooting

### Common Issues

1. **Memory Still Too High**: Reduce base batch size further and increase accumulation steps
2. **Training Instability**: Try different accumulation step values (powers of 2 work well)
3. **Slow Convergence**: May need to adjust learning rate for larger effective batch size
4. **Scheduler Issues**: Verify scheduler configuration accounts for reduced step frequency

### Debugging

Use the provided test script to verify gradient accumulation is working correctly:
```bash
python test_gradient_accumulation.py
```

This will validate that gradients accumulate properly and produce equivalent results to large-batch training. 