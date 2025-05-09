import torch
import numpy as np
from collections import OrderedDict

# Not crazy about this approach..
from mne.utils._bunch import NamedInt
from mne.io.constants import FIFF
# Careful this doesn't overlap with future additions to MNE, might have to coordinate
DEEP_1010_SCALE_CH = NamedInt('DN3_DEEP1010_SCALE_CH', 3000)
DEEP_1010_EXTRA_CH = NamedInt('DN3_DEEP1010_EXTRA_CH', 3001)

_LEFT_NUMBERS = list(reversed(range(1, 9, 2)))
_RIGHT_NUMBERS = list(range(2, 10, 2))

_EXTRA_CHANNELS = 5

DEEP_1010_CHS_LISTING = [
    # EEG
    "NZ",
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F9", *[f"F{n}" for n in _LEFT_NUMBERS], "FZ", *[f"F{n}" for n in _RIGHT_NUMBERS], "F10",

    "FT9", "FT7", *[f"FC{n}" for n in _LEFT_NUMBERS[1:]], "FCZ",
    *[f"FC{n}" for n in _RIGHT_NUMBERS[:-1]], "FT8", "FT10",

    "T9", "T7", "T3",  *[f"C{n}" for n in _LEFT_NUMBERS[1:]], "CZ",
    *[f"C{n}" for n in _RIGHT_NUMBERS[:-1]], "T4", "T8", "T10",

    "TP9", "TP7", *[f"CP{n}" for n in _LEFT_NUMBERS[1:]], "CPZ",
    *[f"CP{n}" for n in _RIGHT_NUMBERS[:-1]], "TP8", "TP10",

    "P9", "P7", "T5",  *[f"P{n}" for n in _LEFT_NUMBERS[1:]], "PZ",
    *[f"P{n}" for n in _RIGHT_NUMBERS[:-1]],  "T6", "P8", "P10",

    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1",  "OZ", "O2",
    "IZ",
    # EOG
    "VEOGL", "VEOGR", "HEOGL", "HEOGR",

    # Ear clip references
    "A1", "A2", "REF",
    # SCALING
    "SCALE",
    # Extra
    *["EX{}".format(n) for n in range(1, _EXTRA_CHANNELS+1)]
]
EEG_INDS = list(range(DEEP_1010_CHS_LISTING.index('VEOGL')))
EOG_INDS = [DEEP_1010_CHS_LISTING.index(ch) for ch in ["VEOGL", "VEOGR", "HEOGL", "HEOGR"]]
REF_INDS = [DEEP_1010_CHS_LISTING.index(ch) for ch in ["A1", "A2", "REF"]]
EXTRA_INDS = list(range(len(DEEP_1010_CHS_LISTING) - _EXTRA_CHANNELS, len(DEEP_1010_CHS_LISTING)))
SCALE_IND = -len(EXTRA_INDS) + len(DEEP_1010_CHS_LISTING)
_NUM_EEG_CHS = len(DEEP_1010_CHS_LISTING) - len(EOG_INDS) - len(REF_INDS) - len(EXTRA_INDS) - 1

DEEP_1010_CH_TYPES = ([FIFF.FIFFV_EEG_CH] * _NUM_EEG_CHS) + ([FIFF.FIFFV_EOG_CH] * len(EOG_INDS)) + \
                     ([FIFF.FIFFV_EEG_CH] * len(REF_INDS)) + [DEEP_1010_SCALE_CH] + \
                     ([DEEP_1010_EXTRA_CH] * _EXTRA_CHANNELS)


def _deep_1010(map, names, eog, ear_ref, extra):
    """
    Generates a mapping matrix from provided channel names to the Deep1010 standard layout.
    This function normalizes the mapping to account for multiple values mapped to a single location.

    Args:
        map (np.ndarray): Initial mapping matrix of shape (num_channels, num_deep1010_channels).
        names (list): List of channel names to be mapped.
        eog (list): List of EOG channel names.
        ear_ref (list): List of ear reference channel names.
        extra (list): List of extra channel names.

    Returns:
        torch.Tensor: Normalized mapping matrix as a float tensor.
    """

    for i, ch in enumerate(names):
        if ch not in eog and ch not in ear_ref and ch not in extra:
            try:
                map[i, DEEP_1010_CHS_LISTING.index(str(ch).upper())] = 1.0
            except ValueError:
                print(f"Warning: channel {ch} not found in standard layout. Skipping...")
                continue

    # Normalize for when multiple values are mapped to single location
    summed = map.sum(axis=0)[np.newaxis, :]
    mapping = torch.from_numpy(np.divide(map, summed, out=np.zeros_like(map), where=summed != 0)).float()
    mapping.requires_grad_(False)
    return mapping


def _valid_character_heuristics(name, informative_characters):
    """
    Extracts valid characters from a channel name based on a set of informative characters.
    Returns a cleaned channel name or None if no valid characters are found.

    Args:
        name (str): The original channel name.
        informative_characters (iterable): Characters considered informative for the mapping.

    Returns:
        str or None: The cleaned channel name, or None if no valid characters are found.
    """
    possible = ''.join(c for c in name.upper() if c in informative_characters).replace(' ', '')
    if not possible:
        print(f"Could not use channel {name}. Could not resolve its true label, rename first.")
        return None
    return possible


def _check_num_and_get_types(type_dict: OrderedDict):
    """
    Ensures the number of EOG and reference channels does not exceed the Deep1010 standard.
    Returns lists of valid EOG and reference channel names, removing any excess channels.

    Args:
        type_dict (OrderedDict): Dictionary mapping channel names to their types.

    Returns:
        tuple: Two lists containing valid EOG and reference channel names, respectively.
    """
    type_lists = []
    for ch_type, max_num in zip(('eog', 'ref'), (len(EOG_INDS), len(REF_INDS))):
        channels = [ch_name for ch_name, _type in type_dict.items() if _type == ch_type]

        for name in channels[max_num:]:
            print(f"Losing assumed {ch_type} channel {name} because there are too many.")
            type_dict[name] = None
        type_lists.append(channels[:max_num])
    return type_lists[0], type_lists[1]


def _heuristic_eog_resolution(eog_channel_name):
    """
    Resolves an EOG channel name to a standardized format using informative characters.
    Returns a cleaned EOG channel name suitable for Deep1010 mapping.

    Args:
        eog_channel_name (str): The original EOG channel name.

    Returns:
        str or None: The cleaned EOG channel name, or None if no valid characters are found.
    """
    return _valid_character_heuristics(eog_channel_name, "VHEOGLR")


def _heuristic_ref_resolution(ref_channel_name: str):
    """
    Resolves a reference channel name to a standardized format.

    Args:
        ref_channel_name (str): The original reference channel name.

    Returns:
        str: The resolved reference channel name. Defaults to "REF" if no specific identifier is found.
    """
    # Remove common identifiers
    ref_channel_name = ref_channel_name.replace('EAR', '').replace('REF', '')

    # Direct mapping based on specific identifiers
    if 'A1' in ref_channel_name or 'L' in ref_channel_name:
        return 'A1'
    
    return 'A2' if 'A2' in ref_channel_name or 'R' in ref_channel_name else "REF"


def _heuristic_eeg_resolution(eeg_ch_name: str):
    """
    Resolves an EEG channel name to a standardized format using informative characters.
    Returns a cleaned EEG channel name suitable for Deep1010 mapping.

    Args:
        eeg_ch_name (str): The original EEG channel name.

    Returns:
        str or None: The cleaned EEG channel name, or None if no valid characters are found.
    """
    eeg_ch_name = eeg_ch_name.upper()
    # remove some common garbage
    eeg_ch_name = eeg_ch_name.replace('EEG', '').replace('REF', '')

    informative_characters = {
        c for name in DEEP_1010_CHS_LISTING[:_NUM_EEG_CHS] for c in name
    }
    return _valid_character_heuristics(eeg_ch_name, informative_characters)


def _likely_eeg_channel(name):
    """
    Determines if a channel name likely corresponds to an EEG channel in the Deep1010 standard.
    Returns True if the name matches any standard EEG channel, otherwise returns False.

    Args:
        name (str): The channel name to check.

    Returns:
        bool: True if the channel is likely an EEG channel, False otherwise.
    """
    if name is not None:
        for ch in DEEP_1010_CHS_LISTING[:_NUM_EEG_CHS]:
            if ch in name.upper():
                return True
    return False


def _heuristic_resolution(old_type_dict: OrderedDict):
    """
    Resolves channel names in a type dictionary to standardized Deep1010 names using heuristics.
    Returns a new OrderedDict with resolved channel names, handling duplicates and unresolvable names.

    Args:
        old_type_dict (OrderedDict): Dictionary mapping original channel names to their types.

    Returns:
        OrderedDict: Dictionary with resolved channel names as keys and their types as values.
    """
    resolver = {'eeg': _heuristic_eeg_resolution, 'eog': _heuristic_eog_resolution, 'ref': _heuristic_ref_resolution,
                'extra': lambda x: x, None: lambda x: x}

    new_type_dict = OrderedDict()

    for old_name, ch_type in old_type_dict.items():
        if ch_type is None:
            new_type_dict[old_name] = None
            continue

        new_name = resolver[ch_type](old_name)
        if new_name is None:
            new_type_dict[old_name] = None
        else:
            while new_name in new_type_dict.keys():
                print(
                    f'Deep1010 Heuristics resulted in duplicate entries for {new_name}, incrementing name, but will be lost in mapping'
                )
                new_name = f"{new_name}-COPY"
            new_type_dict[new_name] = old_type_dict[old_name]

    assert len(new_type_dict) == len(old_type_dict)
    return new_type_dict


def map_named_channels_deep_1010(channel_names: list, EOG=None, ear_ref=None, extra_channels=None):
    """
    Maps channel names to the Deep1010 format, will automatically map EOG and extra channels if they have been
    named according to standard convention. Otherwise provide as keyword arguments.

    Parameters
    ----------
    channel_names : list
                   List of channel names from dataset
    EOG : list, str
         Must be a single channel name, or left and right EOG channels, optionally vertical L/R then horizontal
         L/R for four channels.
    ear_ref : Optional, str, list
               One or two channels to be used as references. If two, should be left and right in that order.
    extra_channels : list, None
                     Up to 6 extra channels to include. Currently not standardized, but could include ECG, respiration,
                     EMG, etc.

    Returns
    -------
    mapping : torch.Tensor
              Mapping matrix from previous channel sequence to Deep1010.
    """
    mapping_matrix = np.zeros((len(channel_names), len(DEEP_1010_CHS_LISTING)))

    if isinstance(EOG, str):
        EOG = [EOG] * 4
    elif not EOG:
        EOG = []
    elif len(EOG) == 1:
        EOG = EOG * 4
    elif len(EOG) == 2:
        EOG = EOG * 2
    else:
        assert len(EOG) == 4, f"EOG channels must be 1, 2, or 4 channels. Got {len(EOG)}"

    for eog_map, eog_std in zip(EOG, EOG_INDS):
        try:
            mapping_matrix[channel_names.index(eog_map), eog_std] = 1.0
        except ValueError as e:
            raise ValueError(f"EOG channel {eog_map} not found in provided channels.") from e

    if isinstance(ear_ref, str):
        ear_ref = [ear_ref] * 2
    elif not ear_ref:
        ear_ref = []
    else:
        assert len(ear_ref) <= len(REF_INDS), "Too many ear reference channels provided."
    for ref_map, ref_std in zip(ear_ref, REF_INDS):
        try:
            mapping_matrix[channel_names.index(ref_map), ref_std] = 1.0
        except ValueError as e:
            raise ValueError(f"Reference channel {ref_map} not found in provided channels.") from e

    if isinstance(extra_channels, str):
        extra_channels = [extra_channels]
    elif extra_channels is None:
        extra_channels = []
    assert len(extra_channels) <= _EXTRA_CHANNELS
    
    for ch, place in zip(extra_channels, EXTRA_INDS):
        if ch is not None:
            mapping_matrix[channel_names.index(ch), place] = 1.0

    return _deep_1010(mapping_matrix, channel_names, EOG, ear_ref, extra_channels)


def map_dataset_channels_deep_1010(channels: np.ndarray, exclude_stim=True):
    """
    Maps channels as stored by a :any:`DN3ataset` to the Deep1010 format, will automatically map EOG and extra channels
    by type.

    Parameters
    ----------
    channels : np.ndarray
               Channels that remain a 1D sequence (they should not have been projected into 2 or 3D grids) of name and
               type. This means the array has 2 dimensions:
               ..math:: N_{channels} \by 2
               With the latter dimension containing name and type respectively, as is constructed by default in most
               cases.
    exclude_stim : bool
                   This option allows the stim channel to be added as an *extra* channel. The default (True) will not do
                   this, and it is very rare if ever where this would be needed.

    Warnings
    --------
    If for some reason the stim channel is labelled with a label from the `DEEP_1010_CHS_LISTING` it will be included
    in that location and result in labels bleeding into the observed data.

    Returns
    -------
    mapping : torch.Tensor
              Mapping matrix from previous channel sequence to Deep1010.
    """
    if len(channels.shape) != 2 or channels.shape[1] != 2:
        raise ValueError(
            f"Deep1010 Mapping: channels must be a 2 dimensional array with dim0 = num_channels, dim1 = 2. Got {channels.shape}"
        )
    channel_types = OrderedDict()

    # Use this for some semblance of order in the "extras"
    extra = [None for _ in range(_EXTRA_CHANNELS)]
    extra_idx = 0

    for name, ch_type in channels:
        # Annoyingly numpy converts them to strings...
        ch_type = int(ch_type)
        if ch_type == FIFF.FIFFV_EEG_CH and _likely_eeg_channel(name):
            channel_types[name] = 'eeg'
        elif ch_type == FIFF.FIFFV_EOG_CH or name in [DEEP_1010_CHS_LISTING[idx] for idx in EOG_INDS]:
            channel_types[name] = 'eog'
        elif ch_type == FIFF.FIFFV_STIM_CH:
            if exclude_stim:
                channel_types[name] = None
                continue
            # if stim, always set as last extra
            channel_types[name] = 'extra'
            extra[-1] = name
        elif 'REF' in name.upper() or 'A1' in name.upper() or 'A2' in name.upper() or 'EAR' in name.upper():
            channel_types[name] = 'ref'
        else:
            if extra_idx == _EXTRA_CHANNELS - 1 and not exclude_stim:
                print(f"Stim channel overwritten by {name} in Deep1010 mapping.")
            elif extra_idx == _EXTRA_CHANNELS:
                print(f"No more room in extra channels for {name}")
                continue
            channel_types[name] = 'extra'
            extra[extra_idx] = name
            extra_idx += 1

    revised_channel_types = _heuristic_resolution(channel_types)
    eog, ref = _check_num_and_get_types(revised_channel_types)

    return map_named_channels_deep_1010(list(revised_channel_types.keys()), eog, ref, extra)


def stringify_channel_mapping(original_names: list, mapping: np.ndarray):
    """
    Creates a human-readable summary of the mapping from original channel names to Deep1010 standard names.
    Returns a string showing the mapping for each channel type and highlights channels that were heuristically assigned.

    Args:
        original_names (list): List of original channel names.
        mapping (np.ndarray): Mapping matrix from original channels to Deep1010 channels.

    Returns:
        str: A formatted string summarizing the channel mapping and heuristic assignments.
    """
    result = ''
    heuristically_mapped = []

    def match_old_new_idx(old_idx, new_idx_set: list):
        """
        Finds the Deep1010 standard channel names that correspond to a given original channel index.
        Returns a comma-separated string of mapped Deep1010 channel names for the specified index set.

        Args:
            old_idx (int): Index of the original channel.
            new_idx_set (list): List of Deep1010 channel indices to consider.

        Returns:
            str: Comma-separated Deep1010 channel names mapped from the original channel.
        """
        new_names = [DEEP_1010_CHS_LISTING[i] for i in np.nonzero(mapping[old_idx, :])[0] if i in new_idx_set]
        return ','.join(new_names)

    for inds, label in zip([list(range(_NUM_EEG_CHS)), EOG_INDS, REF_INDS, EXTRA_INDS],
                           ['EEG', 'EOG', 'REF', 'EXTRA']):
        result += "{} (original(new)): ".format(label)
        for idx, name in enumerate(original_names):
            news = match_old_new_idx(idx, inds)
            if len(news) > 0:
                result += '{}({}) '.format(name, news)
                if news != name.upper():
                    heuristically_mapped.append('{}({}) '.format(name, news))
        result += '\n'

    result += 'Heuristically Assigned: ' + ' '.join(heuristically_mapped)

    return result
