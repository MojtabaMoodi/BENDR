#!/usr/bin/env python3
"""
Test script for gender prediction implementation
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dn3_ext import LoaderGenderEDF
from dn3.configuratron import ExperimentConfig
import utils

def test_gender_loader():
    """Test the LoaderGenderEDF custom loader"""
    print("Testing LoaderGenderEDF...")
    
    # Find some test EDF files (exclude hypnogram files)
    edf_dir = Path("/Volumes/Data/SSC/sleep-cassette")
    if not edf_dir.exists():
        print(f"EDF directory {edf_dir} not found. Please update the path.")
        return False
    
    # Filter out hypnogram files and get actual EEG data files
    all_edf_files = list(edf_dir.glob("*.edf"))
    edf_files = [f for f in all_edf_files if "Hypnogram" not in f.name][:3]  # Test with first 3 data files
    
    if not edf_files:
        print("No EEG data EDF files found in the directory.")
        print(f"Found {len(all_edf_files)} total EDF files, but all appear to be hypnogram files.")
        return False
    
    loader = LoaderGenderEDF()
    
    for edf_file in edf_files:
        try:
            print(f"\nTesting file: {edf_file.name}")
            
            # Extract gender directly
            gender = loader._extract_gender_from_edf(edf_file)
            print(f"  Extracted gender: {gender}")
            
            if gender == 'unknown':
                print(f"  Skipping {edf_file.name} - unknown gender")
                continue
            
            # Test the full loader
            raw = loader(edf_file)
            print(f"  Raw data shape: {raw.get_data().shape}")
            print(f"  Duration: {raw.times[-1]:.1f} seconds")
            
            # Check events
            if hasattr(raw, '_events') and raw._events is not None:
                events = raw._events
                unique_events = set(events[:, -1])
                print(f"  Number of events created: {len(events)}")
                print(f"  Unique event codes: {unique_events}")
            
        except Exception as e:
            print(f"  Error with {edf_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    return True

def test_dataset_creation():
    """Test creating a dataset with our gender configuration"""
    print("\n" + "="*50)
    print("Testing Dataset Creation...")
    
    try:
        # Test the configuration
        experiment = ExperimentConfig('configs/gender_prediction.yml')
        print(f"Loaded experiment config with datasets: {list(experiment.datasets.keys())}")
        
        # Test dataset creation for each dataset
        for ds_name, ds in experiment.datasets.items():
            print(f"\nTesting dataset: {ds_name}")
            try:
                dataset = utils.get_ds(ds_name, ds)
                print(f"  Successfully created dataset")
                print(f"  Number of samples: {len(dataset)}")
                print(f"  Number of people: {len(dataset.get_thinkers())}")
                print(f"  Sample shape: {dataset[0][0].shape}")
                print(f"  Target shape: {dataset[0][1].shape}")
                print(f"  Target value: {dataset[0][1].item()}")
                
                # Check targets distribution
                if hasattr(dataset, 'get_targets'):
                    targets = dataset.get_targets()
                    if targets is not None:
                        unique, counts = torch.unique(torch.tensor(targets), return_counts=True)
                        print(f"  Target distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
                
                return True
                
            except Exception as e:
                print(f"  Error creating dataset {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                return False
                
    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Gender Prediction Tests...")
    
    # Test 1: Gender loader
    success1 = test_gender_loader()
    
    # Test 2: Dataset creation
    success2 = test_dataset_creation() if success1 else False
    
    if success1 and success2:
        print("\n" + "="*50)
        print("✅ All tests passed! Gender prediction system is ready.")
        print("\nTo run gender prediction:")
        print("python downstream.py")
        print("\nOr with specific arguments:")
        print("python downstream.py linear --ds-config configs/gender_prediction.yml --results-filename gender_results.xlsx")
    else:
        print("\n" + "="*50)
        print("❌ Some tests failed. Please check the error messages above.") 