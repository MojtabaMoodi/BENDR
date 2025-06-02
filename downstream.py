import torch
from torch.ao.quantization import (
    get_default_qat_module_mappings,
    get_default_qat_qconfig,
    prepare_qat
)
from torch.ao.nn.qat.modules.conv import (
    Conv1d as QATConv1d,
    Conv2d as QATConv2d,
    Conv3d as QATConv3d
)
import torch.nn as nn
import tqdm
import argparse
import gc
import os

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import objgraph
import numpy as np

import time
import utils
import random
from torch.amp import GradScaler, autocast
from result_tracking import ThinkerwiseResultTracker

from dn3.configuratron import ExperimentConfig
from dn3.data.dataset import Thinker
from dn3.trainable.processes import StandardClassification

from dn3_ext import BENDRClassification, LinearHeadBENDR

# Since we are doing a lot of loading, this is nice to suppress some tedious information
import mne
import sys
mne.set_log_level(False)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if using CUDA
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def get_model_size(model, include_buffers=True):
    """
    Returns the size of the PyTorch model in megabytes.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        include_buffers (bool): Whether to include model buffers (like in BatchNorm) in the size.

    Returns:
        float: Model size in MB.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024 ** 2)  # Convert bytes to MB

def apply_qconfig_recursive(model, qconfig, qat_mapping):
    for name, module in model.named_children():
        module_type = type(module)
        if module_type in qat_mapping and hasattr(qat_mapping[module_type], '_FLOAT_MODULE'):
            float_type = qat_mapping[module_type]._FLOAT_MODULE
            module.qconfig = qconfig if isinstance(module, float_type) else None
        else:
            module.qconfig = None  # Not in mapping
        apply_qconfig_recursive(module, qconfig, qat_mapping)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('model', choices=utils.MODEL_CHOICES)
    parser.add_argument('--ds-config', default="configs/downstream.yml", help="The DN3 config file to use.")
    parser.add_argument('--metrics-config', default="configs/metrics.yml", help="Where the listings for config "
                                                                                "metrics are stored.")
    parser.add_argument('--subject-specific', action='store_true', help="Fine-tune on target subject alone.")
    parser.add_argument('--mdl', action='store_true', help="Fine-tune on target subject using all extra data.")
    parser.add_argument('--freeze-encoder', action='store_true', help="Whether to keep the encoder stage frozen. "
                                                                      "Will only be done if not randomly initialized.")
    parser.add_argument('--random-init', action='store_true', help='Randomly initialized BENDR for comparison.')
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute BENDR over multiple GPUs')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers.')
    parser.add_argument('--results-filename', default=None, help='What to name the spreadsheet produced with all '
                                                                 'final results.')
    # Add argument to use different precisions
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp32', help='Precision to use: full (fp32), half (fp16), or int8 quantization.')
    
    # If no args are provided (e.g. during VS code debugging), set default args
    if len(sys.argv) == 1:
        # Gender prediction configurations
        # These configurations test different combinations of model architectures,
        # initialization strategies, and precision settings for gender classification
        
        # sys.argv += ['BENDR', '--ds-config', 'configs/gender_prediction.yml', '--results-filename', 'gender_results.xlsx', '--precision', 'fp32'] # Gender Config No. 1 
        # Full BENDR model with pre-trained weights (most powerful configuration)
        # Uses both encoder and contextualizer with transformer architecture
        # Expected to provide best performance but slowest training
        
        # sys.argv += ['linear', '--ds-config', 'configs/gender_prediction.yml', '--results-filename', 'gender_results.xlsx', '--precision', 'fp32'] # Gender Config No. 2
        # LinearHead BENDR with pre-trained encoder (balanced configuration)
        # Uses pre-trained encoder but simpler linear classifier head
        # Good trade-off between performance and training speed
        
        # sys.argv += ['BENDR', '--ds-config', 'configs/gender_prediction.yml', '--random-init', '--results-filename', 'gender_results.xlsx', '--precision', 'fp16'] # Gender Config No. 3
        # Full BENDR with random initialization (baseline comparison)
        # Tests whether pre-trained weights provide advantage over random init
        # Uses fp16 precision for memory efficiency during longer training
        
        # sys.argv += ['BENDR', '--ds-config', 'configs/gender_prediction.yml', '--freeze-encoder', '--results-filename', 'gender_results.xlsx', '--precision', 'fp32'] # Gender Config No. 4
        # Full BENDR with frozen encoder (transfer learning approach)
        # Only trains contextualizer and classifier, preserving encoder features
        # Tests whether pre-trained encoder features are sufficient
        
        # sys.argv += ['linear', '--ds-config', 'configs/gender_prediction.yml', '--random-init', '--results-filename', 'gender_results.xlsx', '--precision', 'fp16'] # Gender Config No. 5
        # LinearHead BENDR with random initialization (lightweight baseline)
        # Simplest model architecture without pre-trained weights
        # Provides baseline performance comparison for transfer learning benefits
        
        sys.argv += ['linear', '--ds-config', 'configs/gender_prediction.yml', '--freeze-encoder', '--results-filename', 'gender_results.xlsx', '--precision', 'fp32'] # Gender Config No. 6
        # LinearHead BENDR with frozen encoder (DEFAULT - optimal for development)
        # Fast training with frozen pre-trained encoder features
        # Best configuration for iterative development and testing
        # Provides good performance with minimal computational requirements
        
        # Original sleep stage configurations (commented out)
        # These were the original configurations for sleep stage classification
        # Kept for reference and potential comparison studies
        # sys.argv += ['BENDR', '--ds-config', 'configs/downstream.yml', '--results-filename', 'results.xlsx', '--precision', 'fp32'] # Config No. 1
        # sys.argv += ['linear', '--ds-config', 'configs/downstream.yml', '--results-filename', 'results.xlsx', '--precision', 'fp32'] # Config No. 2
        # sys.argv += ['BENDR', '--ds-config', 'configs/downstream.yml', '--random-init', '--results-filename', 'results.xlsx', '--precision', 'fp16'] # Config No. 3
        # sys.argv += ['BENDR', '--ds-config', 'configs/downstream.yml', '--freeze-encoder', '--results-filename', 'results.xlsx', '--precision', 'fp32'] # Config No. 4
        # sys.argv += ['linear', '--ds-config', 'configs/downstream.yml', '--random-init', '--results-filename', 'results.xlsx', '--precision', 'fp16'] # Config No. 5
        # sys.argv += ['linear', '--ds-config', 'configs/downstream.yml', '--freeze-encoder', '--results-filename', 'results.xlsx', '--precision', 'fp32'] # Config No. 6


    args = parser.parse_args()
    experiment = ExperimentConfig(args.ds_config)
    if args.results_filename:
        results = ThinkerwiseResultTracker()

    for ds_name, ds in tqdm.tqdm(experiment.datasets.items(), total=len(experiment.datasets.items()), desc='Datasets'):
        added_metrics, retain_best, _ = utils.get_ds_added_metrics(ds_name, args.metrics_config)
        for fold, (training, validation, test) in enumerate(tqdm.tqdm(utils.get_lmoso_iterator(ds_name, ds))):

            # Force garbage collection before each fold
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                tqdm.tqdm.write(f"GPU memory before fold: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                tqdm.tqdm.write(torch.cuda.memory_summary())

            if args.model == utils.MODEL_CHOICES[0]:
                model = BENDRClassification.from_dataset(training, multi_gpu=args.multi_gpu, args=args)
            else:
                model = LinearHeadBENDR.from_dataset(training, args=args)

            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

            # Move model to device first
            model = model.to(device)

            if not args.random_init:
                model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights,
                                              freeze_encoder=args.freeze_encoder)
                
            # If mixed-precision then half the model
            if args.precision == 'fp16':
                # Convert model to half precision
                model = model.half()
                scaler = GradScaler()
                # Convert training, validation datasets to half precision
                training = [(x.half(), y) for x, y in training]
                validation = [(x.half(), y) for x, y in validation]
            elif args.precision == 'int8':
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        print(f"[OK] {name}: {type(module)}")
                    elif isinstance(module, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
                        print(f"[SKIP-NDQ] {name}: {type(module)}")
                    else:
                        print(f"[SKIP] {name}: {type(module)}")
                # Quantize the model to int8; fake quantization occurs in float32
                qconfig = get_default_qat_qconfig('fbgemm')
                qat_mapping = get_default_qat_module_mappings()
                qat_mapping[nn.Conv1d] = QATConv1d
                # qat_mapping[nn.Conv2d] = QATConv2d
                # qat_mapping[nn.Conv3d] = QATConv3d

                # 1. Remove any existing qconfig to avoid fallback behavior
                for module in model.modules():
                    module.qconfig = None

                # 2. Apply the qconfig to the model
                apply_qconfig_recursive(model, qconfig, qat_mapping)

                # 3. Prepare the model for QAT
                prepare_qat(model, mapping=qat_mapping, inplace=True)

                print("=============================> qat_mapping", qat_mapping)
                print("=============================> model.encoder", model.encoder)

            process = StandardClassification(model, metrics=added_metrics, precision=args.precision)
            optimizer = torch.optim.Adam(process.parameters(), ds.lr, eps=1e-4, weight_decay=0.01)
            process.set_optimizer(optimizer)
            # import pdb; pdb.set_trace()
            print(f"Model size: {get_model_size(process):.2f} MB")

            if args.precision == 'fp16':
                with autocast(device_type=device):
                    process.fit(training_dataset=training, validation_dataset=validation, 
                                warmup_frac=0.1, retain_best=retain_best, 
                                pin_memory=False, **ds.train_params)
            elif args.precision == 'int8':
                # Dummy forward pass to initialize the model
                # This is necessary for the quantization process to work correctly
                # and to ensure that the model is in the correct state before training
                # model.train() is needed to set the model in training mode and learn with fake quantization
                # and to ensure that the quantization process works correctly
                model.train()
                with torch.no_grad():
                    for x, _ in training[:1]:  # one batch is enough
                        model(x.to(device))
                process.fit(training_dataset=training, validation_dataset=validation, 
                            warmup_frac=0.1, retain_best=retain_best, 
                            pin_memory=False, **ds.train_params)
                # Convert the model to int8
                # This is necessary to finalize the quantization process
                # and to ensure that the model is in the correct state after training
                model = torch.ao.quantization.convert(model.cpu(), inplace=False).to(device)
            else:
                process.fit(training_dataset=training, validation_dataset=validation, 
                            warmup_frac=0.1, retain_best=retain_best, 
                            pin_memory=False, **ds.train_params)

            if args.results_filename:
                if isinstance(test, Thinker):
                    results.add_results_thinker(process, ds_name, test)
                else:
                    results.add_results_all_thinkers(process, ds_name, test, Fold=fold+1)
                results.to_spreadsheet(args.results_filename)

            # explicitly garbage collect here, don't want to fit two models in GPU at once
            del process
            # Skip objgraph to avoid memory issues and file creation
            # objgraph.show_backrefs(model, filename='sample-backref-graph.png')
            del model
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Reduced sleep time
            time.sleep(2)

        if args.results_filename:
            results.performance_summary(ds_name)
            results.to_spreadsheet(args.results_filename)
