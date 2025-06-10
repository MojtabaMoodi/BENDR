from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification
from dn3.trainable.models import TIDNet

# Since we are doing a lot of loading, this is nice to suppress some tedious information
import mne
mne.set_log_level(False)

config_filename = '/Users/mojtaba/Library/CloudStorage/OneDrive-UniversityofWaterloo/Think/Time Management/UWaterloo/Research/my research/BENDR/examples/my_config.yml'
experiment = ExperimentConfig(config_filename)
ds_config = experiment.datasets['mmidb']

dataset = ds_config.auto_construct_dataset()


import sys
import os
import objgraph
import torch
import time

# Add the parent directory to the path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import utils
from dn3_ext import BENDRClassification
from dn3.metrics.base import balanced_accuracy
from result_tracking import ThinkerwiseResultTracker

results = ThinkerwiseResultTracker()
filename = 'results.csv'

for ds_name, ds in (experiment.datasets.items()):
        added_metrics, retain_best, _ = utils.get_ds_added_metrics(ds_name, '../configs/metrics.yml')
        for training, validation, test in utils.get_lmoso_iterator(ds_name, ds):
                # training = Float16DatasetWrapper(training)
                # validation = Float16DatasetWrapper(validation)
                # test = Float16DatasetWrapper(test)
                model = BENDRClassification.from_dataset(training)
                model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights, freeze_encoder=True)
                model = model.half()

                process = StandardClassification(model, metrics=balanced_accuracy)
                
                process.evaluate(test)