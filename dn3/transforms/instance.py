import warnings

import torch
import numpy as np

# from random import choices
from typing import List

from .channels import map_dataset_channels_deep_1010, DEEP_1010_CH_TYPES, SCALE_IND, \
    EEG_INDS, EOG_INDS, REF_INDS, EXTRA_INDS, DEEP_1010_CHS_LISTING
from dn3.utils import min_max_normalize

from torch.nn.functional import interpolate


def same_channel_sets(channel_sets: list):
    """
    Checks if all channel sets in the list have the same shape and structure.
    Returns True if all channel sets are consistent, otherwise returns False.

    Args:
        channel_sets (list): List of channel set arrays to compare.

    Returns:
        bool: True if all channel sets are consistent, False otherwise.
    """
    return not any(
        chs.shape[0] != channel_sets[0].shape[0]
        or chs.shape[1] != channel_sets[0].shape[1]
        for chs in channel_sets[1:]
    )


class InstanceTransform(object):
    """
    Base class for trial-level transforms applied to loaded tensors in EEG/MEG data processing.
    Provides an interface for modifying trial data, channel representations, sampling frequency, and sequence length.
    """

    def __init__(self, only_trial_data=True):
        """
        Trial transforms are, for the most part, simply operations that are performed on the loaded tensors when they are
        fetched via the :meth:`__call__` method. Ideally this is implemented with pytorch operations for ease of execution
        graph integration.
        """
        self.only_trial_data = only_trial_data

    def __str__(self):
        """
        Returns the name of the transform class as a string.
        Useful for logging and debugging purposes.

        Returns:
            str: The class name of the transform.
        """
        return self.__class__.__name__

    def __call__(self, *x):
        """
        Modifies a batch of tensors.
        Parameters
        ----------
        x : torch.Tensor, tuple
            The trial tensor, not including a batch-dimension. If initialized with `only_trial_data=False`, then this
            is a tuple of all ids, labels, etc. being propagated.
        Returns
        -------
        x : torch.Tensor, tuple
            The modified trial tensor, or tensors if not `only_trial_data`
        """
        raise NotImplementedError()

    def new_channels(self, old_channels):
        """
        This is an optional method that indicates the transformation modifies the representation and/or presence of
        channels.

        Parameters
        ----------
        old_channels : ndarray
                       An array whose last two dimensions are channel names and channel types.

        Returns
        -------
        new_channels : ndarray
                      An array with the channel names and types after this transformation. Supports the addition of
                      dimensions e.g. a list of channels into a rectangular grid, but the *final two dimensions* must
                      remain the channel names, and types respectively.
        """
        return old_channels

    def new_sfreq(self, old_sfreq):
        """
        This is an optional method that indicates the transformation modifies the sampling frequency of the underlying
        time-series.

        Parameters
        ----------
        old_sfreq : float

        Returns
        -------
        new_sfreq : float
        """
        return old_sfreq

    def new_sequence_length(self, old_sequence_length):
        """
        This is an optional method that indicates the transformation modifies the length of the acquired extracts,
        specified in number of samples.

        Parameters
        ----------
        old_sequence_length : int

        Returns
        -------
        new_sequence_length : int
        """
        return old_sequence_length


class _PassThroughTransform(InstanceTransform):
    """
    A transform that returns its input unchanged.
    Useful as a placeholder or for cases where no transformation is desired.
    """

    def __init__(self):
        """
        Initializes the transform with only_trial_data set to False.
        This ensures the transform can operate on all data, not just trial data.
        """
        super().__init__(only_trial_data=False)

    def __call__(self, *x):
        """
        Returns the input arguments unchanged.
        This allows the transform to act as a no-op in transformation pipelines.

        Args:
            *x: Arbitrary input arguments, typically tensors or data structures.

        Returns:
            tuple: The input arguments, unchanged.
        """
        return x


class ZScore(InstanceTransform):
    """
    Applies z-score normalization to the input tensor.
    Normalizes the data using the provided mean and standard deviation, or computes them from the input if not provided.
    """
    def __init__(self, mean=None, std=None):
        """
        Initializes the ZScore transform with optional mean and standard deviation.
        If mean or std are not provided, they will be computed from the input tensor during normalization.

        Args:
            mean (float, optional): The mean value for normalization. Defaults to None.
            std (float, optional): The standard deviation for normalization. Defaults to None.
        """
        super(ZScore, self).__init__()
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        """
        Applies z-score normalization to the input tensor.
        Uses the provided mean and standard deviation, or computes them from the input if not specified.

        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        mean = x.mean() if self.mean is None else self.mean
        std = x.std() if self.std is None else self.std
        return (x - mean) / std


class FixedScale(InstanceTransform):
    """
    Scales the input tensor to a fixed range using min-max normalization.
    The output values are mapped to the interval [low_bound, high_bound].
    """
    def __init__(self, low_bound=-1, high_bound=1):
        """
        Initializes the FixedScale transform with specified lower and upper bounds.

        Args:
            low_bound (float, optional): The minimum value of the scaled output. Defaults to -1.
            high_bound (float, optional): The maximum value of the scaled output. Defaults to 1.
        """
        super().__init__()
        self.low_bound = low_bound
        self.high_bound = high_bound

    def __call__(self, x):
        """
        Applies min-max normalization to the input tensor, scaling it to the specified bounds.

        Args:
            x (torch.Tensor): The input tensor to scale.

        Returns:
            torch.Tensor: The scaled tensor with values in [low_bound, high_bound].
        """
        return min_max_normalize(x, self.low_bound, self.high_bound)


class TemporalPadding(InstanceTransform):
    """
    Pads the input tensor along the temporal dimension with specified values.
    The number of samples added to the start and end are determined by start_padding and end_padding.
    """
    def __init__(self, start_padding, end_padding, mode='constant', constant_value=0):
        """
        Pad the number of samples.

        Parameters
        ----------
        start_padding : int
                        The number of padded samples to add to the beginning of a trial
        end_padding : int
                      The number of padded samples to add to the end of a trial
        mode : str
               See `pytorch documentation <https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad>`_
        constant_value : float
               If mode is 'constant' (the default), the value to compose the samples of.
        """
        super().__init__()
        self.start_padding = start_padding
        self.end_padding = end_padding
        self.mode = mode
        self.constant_value = constant_value

    def __call__(self, x):
        """
        Pads the input tensor along the temporal dimension with specified values.
        The number of samples added to the start and end are determined by start_padding and end_padding.

        Args:
            x (torch.Tensor): The input tensor to pad.

        Returns:
            torch.Tensor: The padded tensor.
        """
        pad = [self.start_padding, self.end_padding] + [0 for _ in range(2, x.shape[-1])]
        return torch.nn.functional.pad(x, pad, mode=self.mode, value=self.constant_value)

    def new_sequence_length(self, old_sequence_length):
        """
        Calculates the new sequence length after temporal padding is applied.
        The result is the original sequence length plus the number of samples added at the start and end.

        Args:
            old_sequence_length (int): The original sequence length.

        Returns:
            int: The new sequence length after padding.
        """
        return old_sequence_length + self.start_padding + self.end_padding


class TemporalInterpolation(InstanceTransform):
    """
    Interpolates the input tensor to a specified sequence length using a chosen interpolation mode.
    This transform is a DN3 wrapper for PyTorch's interpolate function and supports both single trial and batch input.
    """

    def __init__(self, desired_sequence_length, mode='nearest', new_sfreq=None):
        """
        Initializes the TemporalInterpolation transform with the desired sequence length, interpolation mode, and optional new sampling frequency.
        This is in essence a DN3 wrapper for the pytorch function
        `interpolate() <https://pytorch.org/docs/stable/nn.functional.html>`_

        Currently only supports single dimensional samples (i.e. channels have not been projected into more dimensions)

        Warnings
        --------
        Using this function to downsample data below a suitable nyquist frequency given the low-pass filtering of the
        original data will cause dangerous aliasing artifacts that will heavily affect data quality to the point of it
        being mostly garbage.

        Parameters
        ----------
        desired_sequence_length: int
                                 The desired new sequence length of incoming data.
        mode: str
              The technique that will be used for upsampling data, by default 'nearest' interpolation. Other options
              are listed under pytorch's interpolate function.
        new_sfreq: float, None
                   If specified, registers the change in sampling frequency
        """
        super().__init__()
        self._new_sequence_length = desired_sequence_length
        self.mode = mode
        self._new_sfreq = new_sfreq

    def __call__(self, x):
        """
        Interpolates the input tensor to a new sequence length using the specified mode.
        Supports both single trial and batch input, and only operates on single dimensional channel data.

        Args:
            x (torch.Tensor): The input tensor to interpolate.

        Returns:
            torch.Tensor: The interpolated tensor with the new sequence length.

        Raises:
            ValueError: If the input tensor does not have 2 or 3 dimensions.
        """
        # squeeze and unsqueeze because these are done before batching
        if len(x.shape) == 2:
            return interpolate(x.unsqueeze(0), self._new_sequence_length, mode=self.mode).squeeze(0)
        # Supports batch dimension
        elif len(x.shape) == 3:
            return interpolate(x, self._new_sequence_length, mode=self.mode)
        else:
            raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

    def new_sequence_length(self, old_sequence_length):
        """
        Returns the new sequence length after interpolation.
        The new sequence length is set during initialization and does not depend on the original length.

        Args:
            old_sequence_length (int): The original sequence length.

        Returns:
            int: The new sequence length after interpolation.
        """
        return self._new_sequence_length

    def new_sfreq(self, old_sfreq):
        """
        Returns the new sampling frequency if specified, otherwise returns the original.
        The new sampling frequency is set during initialization and does not depend on the original value.

        Args:
            old_sfreq (float): The original sampling frequency.

        Returns:
            float: The new sampling frequency after interpolation.
        """
        return self._new_sfreq if self._new_sfreq is not None else old_sfreq


class CropAndUpSample(TemporalInterpolation):
    """
    Randomly crops the input tensor to a variable length and then upsamples it to a fixed sequence length.
    The crop length is randomly selected between a minimum value and the current sequence length.
    """

    def __init__(self, original_sequence_length, crop_sequence_min):
        """
        Initializes the CropAndUpSample transform with the original sequence length and minimum crop length.

        Args:
            original_sequence_length (int): The target sequence length after upsampling.
            crop_sequence_min (int): The minimum length to which the input can be cropped.
        """
        super().__init__(original_sequence_length)
        self.crop_sequence_min = crop_sequence_min

    def __call__(self, x):
        """
        Randomly crops the input tensor along the temporal dimension and upsamples it to the original sequence length.

        Args:
            x (torch.Tensor): The input tensor to crop and upsample.

        Returns:
            torch.Tensor: The cropped and upsampled tensor.
        """
        crop_len = np.random.randint(low=self.crop_sequence_min, high=x.shape[-1])
        return super(CropAndUpSample, self).__call__(x[:, :crop_len])

    def new_sequence_length(self, old_sequence_length):
        """
        Returns the new sequence length after cropping and upsampling.

        Args:
            old_sequence_length (int): The original sequence length.

        Returns:
            int: The new sequence length after transformation.
        """
        return old_sequence_length - self.crop_sequence_min


class TemporalCrop(InstanceTransform):

    def __init__(self, cropped_length, start_offset=None):
        """
        Crop to a new length of `cropped_length` from the specified `start_offset`, or randomly select an offset.

        Parameters
        ----------
        cropped_length : int
                         The cropped sequence length (in samples).
        start_offset : int, None, List[int]
                       If a single int, the sample to start the crop from. If `None`, will uniformly select from
                       within the possible start offsets (such that `cropped_length` is not violated). If
                       a list of ints, these specify the *un-normalized* probabilities for different start offsets.
        """
        super().__init__()
        if isinstance(start_offset, int):
            start_offset = [0 for _ in range(start_offset-1)] + [1]
        elif isinstance(start_offset, list):
            summed = sum(start_offset)
            start_offset = [w / summed for w in start_offset]
        self.start_offset = start_offset
        self._new_length = cropped_length

    def _get_start_offset(self, full_length):
        """
        Determines the starting offset for cropping the input tensor.
        The offset can be fixed, randomly selected, or chosen based on a probability distribution.

        Args:
            full_length (int): The total length of the input tensor.

        Returns:
            int: The starting offset for cropping.
        """
        possible_starts = full_length - self._new_length
        assert possible_starts >= 0
        if self.start_offset is None:
            start_offset = np.random.choice(possible_starts) if possible_starts > 0 else 0
        elif isinstance(self.start_offset, list):
            start_offset = self.start_offset + [0 for _ in range(len(self.start_offset), possible_starts)]
            start_offset = np.random.choice(possible_starts, p=start_offset)
        else:
            start_offset = self.start_offset
        return int(start_offset)

    def __call__(self, x):
        """
        Crops the input tensor to a specified length starting from a given or randomly selected offset.
        The crop can be fixed or randomly chosen based on the start_offset parameter.

        Args:
            x (torch.Tensor): The input tensor to crop.

        Returns:
            torch.Tensor: The cropped tensor of length `cropped_length`.
        """
        start_offset = self._get_start_offset(x.shape[-1])
        return x[..., start_offset:start_offset + self._new_length]

    def new_sequence_length(self, old_sequence_length):
        """
        Returns the new sequence length after cropping.
        The new sequence length is set during initialization and does not depend on the original length.

        Args:
            old_sequence_length (int): The original sequence length.

        Returns:
            int: The new sequence length after cropping.
        """
        return self._new_length


class CropAndResample(TemporalInterpolation):

    def __init__(self, desired_sequence_length, stdev, truncate=None, mode='nearest', new_sfreq=None,
                 crop_side='both', allow_uncroppable=False):
        super().__init__(desired_sequence_length, mode=mode, new_sfreq=new_sfreq)
        self.stdev = stdev
        self.allow_uncrop = allow_uncroppable
        self.truncate = float("inf") if truncate is None else truncate
        if crop_side not in ['right', 'left', 'both']:
            raise ValueError("The crop-side should either be 'left', 'right', or 'both'")
        self.crop_side = crop_side

    @staticmethod
    def trunc_norm(mean, std, max_diff):
        """
        Samples an integer from a truncated normal distribution centered at `mean` with standard deviation `std`.
        The sampled value is guaranteed to be within `max_diff` of the mean.

        Args:
            mean (float): The mean of the normal distribution.
            std (float): The standard deviation of the normal distribution.
            max_diff (float): The maximum allowed absolute difference from the mean.

        Returns:
            int: An integer sampled from the truncated normal distribution.
        """
        val = None
        while val is None or abs(val - mean) > max_diff:
            val = int(np.random.normal(mean, std))
        return val

    def new_sequence_length(self, old_sequence_length):
        return self._new_sequence_length

    def __call__(self, x):
        """
        Crops and resamples the input tensor to a new sequence length using a truncated normal distribution for the crop offset.
        The crop can be applied to the left, right, or both sides, and the result is then resampled to the desired sequence length.

        Args:
            x (torch.Tensor): The input tensor to crop and resample.

        Returns:
            torch.Tensor: The cropped and resampled tensor.

        Raises:
            AssertionError: If the maximum allowed crop difference is not positive.
        """
        max_diff = min(x.shape[-1] - self._new_sequence_length, self.truncate)
        if self.allow_uncrop and max_diff == 0:
            return x
        assert max_diff > 0
        crop = np.random.choice(['right', 'left']) if self.crop_side == 'both' else self.crop_side
        if self.crop_side == 'right':
            offset = self.trunc_norm(self._new_sequence_length, self.stdev, max_diff)
            return super(CropAndResample, self).__call__(x[:, :offset])
        else:
            offset = self.trunc_norm(x.shape[-1] - self._new_sequence_length, self.stdev, max_diff)
            return super(CropAndResample, self).__call__(x[:, offset:])


class MappingDeep1010(InstanceTransform):
    """
    Maps various channel sets into the Deep10-10 scheme, and normalizes data between [-1, 1] with an additional scaling
    parameter to describe the relative scale of a trial with respect to the entire dataset.

    See https://doi.org/10.1101/2020.12.17.423197  for description.
    """
    def __init__(self, dataset, max_scale=None, return_mask=False):
        """
        Creates a Deep10-10 mapping for the provided dataset.

        Parameters
        ----------
        dataset : Dataset

        max_scale : float
                    If specified, the scale ind is filled with the relative scale of the trial with respect
                    to this, otherwise uses dataset.info.data_max - dataset.info.data_min.
        return_mask : bool
                      If `True` (`False` by default), an additional tensor is returned after this transform that
                      says which channels of the mapping are in fact in use.
        """
        super().__init__()
        self.mapping = map_dataset_channels_deep_1010(dataset.channels)
        if max_scale is not None:
            self.max_scale = max_scale
        elif dataset.info is None or dataset.info.data_max is None or dataset.info.data_min is None:
            print(f"Warning: Did not find data scale information for {dataset}")
            self.max_scale = None
        else:
            self.max_scale = dataset.info.data_max - dataset.info.data_min
        self.return_mask = return_mask

    @staticmethod
    def channel_listing():
        return DEEP_1010_CHS_LISTING

    def __call__(self, x):
        """
        Maps the input tensor to the Deep10-10 channel scheme and normalizes data between [-1, 1].
        Also computes and inserts a scaling parameter and optionally returns a mask of used channels.

        Args:
            x (torch.Tensor): The input tensor to map and normalize.

        Returns:
            torch.Tensor or tuple: The mapped and normalized tensor, and optionally a mask of used channels.
        """
        if self.max_scale is not None:
            scale = 2 * (torch.clamp_max((x.max() - x.min()) / self.max_scale, 1.0) - 0.5)
        else:
            scale = 0

        x = (x.transpose(1, 0) @ self.mapping).transpose(1, 0)

        for ch_type_inds in (EEG_INDS, EOG_INDS, REF_INDS, EXTRA_INDS):
            x[ch_type_inds, :] = min_max_normalize(x[ch_type_inds, :])

        used_channel_mask = self.mapping.sum(dim=0).bool()
        x[~used_channel_mask, :] = 0

        x[SCALE_IND, :] = scale

        return (x, used_channel_mask) if self.return_mask else x

    def new_channels(self, old_channels: np.ndarray):
        """
        Generates a new channel list after mapping to the Deep10-10 scheme.
        Returns an array of tuples containing the new channel names and their types.

        Args:
            old_channels (np.ndarray): The original channel names and types.

        Returns:
            np.ndarray: An array of (channel_name, channel_type) tuples after mapping.
        """
        channels = []
        for row in range(self.mapping.shape[1]):
            active = self.mapping[:, row].nonzero().numpy()
            if len(active) > 0:
                channels.append("-".join([old_channels[i.item(), 0] for i in active]))
            else:
                channels.append(None)
        return np.array(list(zip(channels, DEEP_1010_CH_TYPES)))


class Deep1010ToEEG(InstanceTransform):

    def __init__(self):
        super().__init__(only_trial_data=False)

    def new_channels(self, old_channels):
        """
        Returns the subset of channels corresponding to EEG channels from the Deep10-10 channel set.
        This method extracts only the EEG channels, excluding EOG, reference, and extra channels.

        Args:
            old_channels (np.ndarray): The original Deep10-10 channel names and types.

        Returns:
            np.ndarray: The EEG-only channel names and types.
        """
        return old_channels[EEG_INDS]

    def __call__(self, *x):
        """
        Extracts only the EEG channels from tensors with Deep10-10 channel length.
        Applies the selection to all input tensors that match the Deep10-10 channel count.

        Args:
            *x: Arbitrary input tensors, typically with channel as the first dimension.

        Returns:
            list: The input tensors with only EEG channels retained.
        """
        x = list(x)
        for i in range(len(x)):
            # Assume every tensor that has deep1010 length should be modified
            if len(x[i].shape) > 0 and x[i].shape[0] == len(DEEP_1010_CHS_LISTING):
                x[i] = x[i][EEG_INDS, ...]

        return x


class To1020(InstanceTransform):

    # This is a transform that takes the Deep10-10 channel set and maps it to the 10-20 system.
    EEG_20_div = [
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'T5', 'P3', 'PZ', 'P4', 'T6',
                'O1', 'O2'
    ]

    def __init__(self, only_trial_data=True, include_scale_ch=True, include_ref_chs=False):
        """
        Transforms incoming Deep1010 data into exclusively the more limited 1020 channel set.
        """
        super(To1020, self).__init__(only_trial_data=only_trial_data)
        self._inds_20_div = [DEEP_1010_CHS_LISTING.index(ch) for ch in self.EEG_20_div]
        if include_ref_chs:
            self._inds_20_div.append([DEEP_1010_CHS_LISTING.index(ch) for ch in ['A1', 'A2']])
        if include_scale_ch:
            self._inds_20_div.append(SCALE_IND)

    def new_channels(self, old_channels):
        return old_channels[self._inds_20_div]

    def __call__(self, *x):
        """
        Extracts the 10-20 system channels from tensors with Deep10-10 channel length.
        Applies the selection to all input tensors that match the Deep10-10 channel count.

        Args:
            *x: Arbitrary input tensors, typically with channel as the first dimension.

        Returns:
            list: The input tensors with only 10-20 system channels retained.
        """
        x = list(x)
        for i in range(len(x)):
            # Assume every tensor that has deep1010 length should be modified
            if len(x[i].shape) > 0 and x[i].shape[0] == len(DEEP_1010_CHS_LISTING):
                x[i] = x[i][self._inds_20_div, ...]
        return x


class MaskAuxiliariesDeep1010(InstanceTransform):

    MASK_THESE = REF_INDS + EOG_INDS + EXTRA_INDS

    def __init__(self, randomize=False):
        super(MaskAuxiliariesDeep1010, self).__init__()
        self.randomize = randomize

    def __call__(self, x):
        """
        Masks the auxiliary channels (reference, EOG, and extra channels) in the input tensor.
        If randomize is True, fills these channels with random values in [-1, 1]; otherwise, sets them to zero.

        Args:
            x (torch.Tensor): The input tensor to mask.

        Returns:
            torch.Tensor: The tensor with auxiliary channels masked.
        """
        if self.randomize:
            x[self.MASK_THESE, :] = 2 * torch.rand_like(x[self.MASK_THESE, :]) - 1
        else:
            x[self.MASK_THESE] = 0
        return x


class NoisyBlankDeep1010(InstanceTransform):

    def __init__(self, mask_index=1, purge_mask=False):
        super().__init__(only_trial_data=False)
        self.mask_index = mask_index
        self.purge_mask = purge_mask

    def __call__(self, *x):
        """
        Replaces the unmasked channels in the first tensor with random noise and optionally removes the mask tensor.
        The unmasked channels are filled with random values in [-1, 1]; if purge_mask is True, the mask tensor is removed from the output.

        Args:
            *x: A tuple or list where the first element is a tensor and the mask is at position `mask_index`.

        Returns:
            list: The modified tensors, with unmasked channels randomized and optionally the mask removed.
        """
        assert isinstance(x, (list, tuple)) and len(x) > 1
        x = list(x)
        blanks = x[0][~x[self.mask_index], :]
        x[0][~x[self.mask_index], :] = 2 * torch.rand_like(blanks) - 1
        if self.purge_mask:
            x = x.pop(self.mask_index)
        return x


class AdditiveEogDeep1010(InstanceTransform):

    EEG_IND_END = EOG_INDS[0] - 1

    def __init__(self, p=0.1, max_intensity=0.3, blank_eog_p=1.0):
        super().__init__()
        self.max_intensity = max_intensity
        self.p = p
        self.blanking_p = blank_eog_p

    def __call__(self, x):
        """
        Adds scaled EOG signals to a random subset of EEG channels and optionally blanks EOG channels.
        For each EOG channel, a random subset of EEG channels receives an additive scaled version of the EOG signal, and EOG channels may be zeroed out with a given probability.

        Args:
            x (torch.Tensor): The input tensor containing EEG and EOG channels.

        Returns:
            torch.Tensor: The tensor with EOG signals added to EEG channels and EOG channels optionally blanked.
        """
        affected_inds = (torch.rand(EOG_INDS[0] - 1).lt(self.p)).nonzero()
        for which_eog in EOG_INDS:
            this_affected_ids = affected_inds[torch.rand_like(affected_inds.float()).lt(0.25)]
            x[this_affected_ids, :] = self.max_intensity * torch.rand_like(this_affected_ids.float()).unsqueeze(-1) *\
                                      x[which_eog]
            # Short circuit the random call as these are more frequent
            if self.blanking_p != 0 and (self.blanking_p == 1 or torch.rand(1) < self.blanking_p):
                x[which_eog, :] = 0
        return x


class UniformTransformSelection(InstanceTransform):

    def __init__(self, transform_list, weights=None, suppress_warnings=False):
        """
        Uniformly selects a transform from the `transform_list` with probabilities according to `weights`.

        Parameters
        ----------
        transform_list: List[InstanceTransform]
                        List of transforms to select from.
        weights: None, List[float]
           This is either `None`, in which case the transforms are selected with equal probability. Or relative
           probabilities to select from `transform_list`. This means, it does not have to be normalized probabilities
           *(this doesn't have to sum to one)*. If `len(transform_list) == len(weights) - 1`, it will be assumed that
           the final weight expresses the likelihood of *no transform*.
        """
        super().__init__(only_trial_data=False)
        self.suppress_warnings = suppress_warnings
        for x in transform_list:
            assert isinstance(x, InstanceTransform)
        self.transforms = transform_list
        if weights is None:
            self._choice_weights = [1 / len(transform_list) for _ in range(len(transform_list))]
        else:
            if len(weights) == len(transform_list) + 1:
                self.transforms.append(_PassThroughTransform())
            total_weight = sum(weights)
            self._choice_weights = [p / total_weight for p in weights]
            assert len(weights) == len(transform_list)

    def __call__(self, *x):
        """
        Randomly selects and applies one of the available transforms to the input data, according to the specified weights.
        If the selected transform operates only on trial data, it is applied to the first input tensor; otherwise, it is applied to all inputs.

        Args:
            *x: Input tensors or data structures to be transformed.

        Returns:
            The result of applying the selected transform to the input data.
        """
        transform = np.random.choice(self.transforms, p=self._choice_weights)
        # if don't need to support <3.6, this is faster
        # which = choices(self.transforms, self._choice_weights)
        if hasattr(transform, 'only_trial_data') and transform.only_trial_data:
            return [transform(x[0]), *x[1:]]
        else:
            return transform(*x)

    def new_channels(self, old_channels):
        all_new = [transform.new_channels(old_channels) for transform in self.transforms]
        if not self.suppress_warnings and not same_channel_sets(all_new):
            warnings.warn('Multiple channel representations!')
        return all_new[0]

    def new_sfreq(self, old_sfreq):
        all_new = {transform.new_sfreq(old_sfreq) for transform in self.transforms}
        if not self.suppress_warnings and len(all_new) > 1:
            warnings.warn('Multiple new sampling frequencies!')
        return all_new.pop()

    def new_sequence_length(self, old_sequence_length):
        all_new = {
            transform.new_sequence_length(old_sequence_length)
            for transform in self.transforms
        }
        if not self.suppress_warnings and len(all_new) > 1:
            warnings.warn('Multiple new sequence lengths!')
        return all_new.pop()


class EuclideanAlignmentTransform(InstanceTransform):

    def __init__(self, reference_matrices, inds):
        super(EuclideanAlignmentTransform, self).__init__(only_trial_data=False)
        self.reference_matrices = reference_matrices
        self.inds = inds

    def __call__(self, *x):
        """
        Applies a Euclidean alignment transformation to a subset of channels in the input tensor.
        Selects the appropriate reference matrix and channel indices based on the input, supporting both single and multiple subject/session cases.

        Args:
            *x: Input tensors, where the first element is the data tensor and subsequent elements may include thinker/session IDs.

        Returns:
            list: The input tensors with the specified channels transformed by the reference matrix.
        """
        # Check that we have at least x, thinker_id, session_id, label if more than one referecing matrix
        # TODO expand so that this works for a single subject and multiple sessions
        if not isinstance(self.reference_matrices, dict):
            xform = torch.transpose(self.reference_matrices, 0, 1)
            inds = self.inds
        else:
            # Missing thinker and session index
            if len(x) == 3:
                thid = 0
                sid = 0
            elif len(x) == 4:
                thid = 0
                sid = int(x[-2])
            else:
                thid = int(x[-3])
                sid = int(x[-2])

            xform = torch.transpose(self.reference_matrices[thid][sid], 0, 1)
            inds = self.inds[thid][sid]

        x[0][inds, :] = torch.matmul(xform, x[0][inds, :])
        return x
