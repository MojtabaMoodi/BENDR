import mne
import torch
import numpy as np

from collections.abc import Iterable
from torch.utils.data.dataset import random_split
from random import seed as py_seed
from numpy.random import seed as np_seed
from torch import manual_seed
from torch.cuda import manual_seed_all as gpu_seed


def init_seed(seed, hard=False):
    """
    Set a constant random seed to have reproducible runs
    Parameters
    ----------
    seed
    hard: bool
         If you are having trouble reproducing runs with multiple GPUs, seeting hard to True should fix it, but it will
         slow performance a bit. This may result in errors if you try to use inherrently non-deterministic algorithms.
    """
    py_seed(seed)
    gpu_seed(seed)
    manual_seed(seed)
    np_seed(seed)
    if hard:
        torch.use_deterministic_algorithms()


class DN3ConfigException(BaseException):
    """
    Exception to be triggered when DN3-configuration parsing fails.
    """
    pass


class DN3atasetException(BaseException):
    """
    Exception to be triggered when DN3-dataset-specific issues arise.
    """
    pass


class DN3atasetNanFound(BaseException):
    """
    Exception to be triggered when DN3-dataset variants load NaN data, or data becomes NaN when pushed through
    transforms.
    """
    pass


def rand_split(dataset, frac=0.75):
    if frac >= 1:
        return dataset
    samples = len(dataset)
    return random_split(dataset, lengths=[round(x) for x in [samples*frac, samples*(1-frac)]])


def unfurl(_set: set):
    _list = list(_set)
    for i in range(len(_list)):
        if not isinstance(_list[i], Iterable):
            _list[i] = [_list[i]]
    return tuple(x for z in _list for x in z)


def min_max_normalize(x: torch.Tensor, low=-1, high=1):
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    return (high - low) * x


def handle_overlapping_events(events):
    """
    Adjust sample indices for overlapping events to treat them as distinct instances.

    Parameters
    ----------
    events : np.ndarray
        The original events array of shape (n_events, 3).

    Returns
    -------
    np.ndarray
        Adjusted events array with unique sample indices for overlapping events.
    """
    unique_events = []
    sample_offset = 1  # Offset to apply to overlapping events

    for i in range(len(events)):
        sample_idx, sample_value, event_id = events[i]
        # Check for overlapping events at the same sample index
        while any((e[0] == sample_idx for e in unique_events)):
            sample_idx += sample_offset
        unique_events.append([sample_idx, sample_value, event_id])

    return np.array(unique_events)


def make_epochs_from_raw(raw: mne.io.Raw, tmin, tlen, event_ids=None, baseline=None, decim=1, filter_bp=None,
                         drop_bad=False, use_annotations=False, chunk_duration=None):
    sfreq = raw.info['sfreq']
    if filter_bp is not None:
        if isinstance(filter_bp, (list, tuple)) and len(filter_bp) == 2:
            raw.load_data()
            raw.filter(filter_bp[0], filter_bp[1])
        else:
            print('Filter must be provided as a two-element list [low, high]')

    try:
        if use_annotations:
            events = mne.events_from_annotations(raw, event_id=event_ids, chunk_duration=chunk_duration)[0]
        else:
            # TODO: Check if this is the right way to do this
            events = mne.find_events(raw)
            events = events[[i for i in range(len(events)) if events[i, -1] in event_ids.keys()], :]
    except ValueError as e:
        if "No stim channels found" not in str(e):
            raise DN3ConfigException(*e.args) from e

        # The expected shape of the events is (n_events, 3), where each row is [sample_index, 0, event_id]
        # events[0] contains the events and events[1] contains the event_id information
        events = mne.events_from_annotations(raw)
        events = events[0]

    try:
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / sfreq, preload=True, decim=decim,
                      baseline=baseline, reject_by_annotation=drop_bad)
    except RuntimeError as e:
        if "Event time samples were not unique" in str(e):
            # Handle overlapping events by adjusting sample indices
            events = handle_overlapping_events(events)
            print("Adjusted overlapping events.")
            epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / sfreq, preload=True, decim=decim,
                              baseline=baseline, reject_by_annotation=drop_bad)
        else:
            raise DN3ConfigException(*e.args) from e
    
    return epochs


def skip_inds_from_bad_spans(epochs: mne.Epochs, bad_spans: list):
    if bad_spans is None:
        return None

    start_times = epochs.events[:, 0] / epochs.info['sfreq']
    end_times = start_times + epochs.tmax - epochs.tmin

    skip_inds = []
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        for bad_start, bad_end in bad_spans:
            if bad_start <= start < bad_end or bad_start < end <= bad_end:
                skip_inds.append(i)
                break

    return skip_inds


# From: https://github.com/pytorch/pytorch/issues/7455
class LabelSmoothedCrossEntropyLoss(torch.nn.Module):
    """this loss performs label smoothing to compute cross-entropy with soft labels, when smoothing=0.0, this
    is the same as torch.nn.CrossEntropyLoss"""

    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
