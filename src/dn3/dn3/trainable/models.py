from abc import ABCMeta

import math

from copy import deepcopy

import numpy as np

from ..data.dataset import DN3ataset
from .layers import *


class DN3BaseModel(nn.Module):
    """
    This is a base model used by the provided models in the library that is meant to make those included in this
    library as powerful and multi-purpose as is reasonable.

    It is not strictly necessary to have new modules inherit from this, any nn.Module should suffice, but it provides
    some integrated conveniences...

    The premise of this model is that deep learning models can be understood as *learned pipelines*. These
    :any:`DN3BaseModel` objects, are re-interpreted as a two-stage pipeline, the two stages being *feature extraction*
    and *classification*.
    """
    def __init__(self, samples, channels, return_features=True):
        """
        Initializes the DN3BaseModel with the specified number of samples, channels, and feature return option.
        Sets up the model for use as a two-stage pipeline for feature extraction and classification.

        Args:
            samples (int): Number of samples per input instance.
            channels (int): Number of input channels.
            return_features (bool, optional): If True, the model returns features in addition to predictions. Defaults to True.
        """
        super().__init__()
        self.samples = samples
        self.channels = channels
        self.return_features = return_features

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Must be implemented by subclasses to specify how input data is processed.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output of the model.
        """
        raise NotImplementedError

    def internal_loss(self, forward_pass_tensors):
        """
        Computes an internal loss for the model, if applicable.
        By default, this method returns None and should be overridden by subclasses if needed.

        Args:
            forward_pass_tensors: The outputs from the model's forward pass.

        Returns:
            None: By default, no internal loss is computed.
        """
        return None

    def clone(self):
        """
        Creates and returns a deep copy of the model instance.
        This allows for safe duplication of the model's state and parameters.

        Returns:
            DN3BaseModel: A deep copy of the current model instance.
        """
        return deepcopy(self)

    def load(self, filename, strict=True):
        """
        Loads model parameters from a file, automatically selecting the appropriate device.
        Supports loading on CUDA, MPS, or CPU devices and updates the model's state dictionary.

        Args:
            filename (str): Path to the file containing the saved model state.
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the model. Defaults to True.
        """
        if torch.cuda.is_available():
            map_location = torch.device("cuda")
        elif torch.backends.mps.is_available():
            map_location = torch.device("mps")
        else:
            map_location = torch.device("cpu")

        state_dict = torch.load(filename, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        """
        Saves the model's state dictionary to a file.
        The saved file can later be loaded to restore the model's parameters.

        Args:
            filename (str): Path to the file where the model state will be saved.
        """
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        """
        Freezes or unfreezes all model parameters except for the classifier layer.
        This is useful for transfer learning, allowing only the classifier to be trained while keeping feature extraction layers fixed.

        Args:
            unfreeze (bool, optional): If True, unfreezes all parameters; if False, freezes them. Defaults to False.
        """
        for param in self.parameters():
            param.requires_grad = unfreeze

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, **modelargs):
        """
        Instantiates a model from a DN3ataset, automatically configuring input dimensions.
        Prints dataset information and asserts the input is a DN3ataset.

        Args:
            dataset (DN3ataset): The dataset to use for model configuration.
            **modelargs: Additional keyword arguments for model initialization.

        Returns:
            DN3BaseModel: An instance of the model configured for the dataset.
        """
        print(
            f"Creating {cls.__name__} using: {len(dataset.channels)} channels with trials of {dataset.sequence_length} samples at {dataset.sfreq}Hz"
        )
        assert isinstance(dataset, DN3ataset)
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs)


class Classifier(DN3BaseModel):
    """
    A generic Classifier container. This container breaks operations up into feature extraction and feature
    classification to enable convenience in transfer learning and more.
    """

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, **modelargs):
        """
        Create a classifier from a dataset.

        Parameters
        ----------
        dataset
        modelargs: dict
                   Options to construct the dataset, if dataset does not have listed targets, targets must be specified
                   in the keyword arguments or will fall back to 2.

        Returns
        -------
        model: Classifier
               A new `Classifier` ready to classifiy data from `dataset`
        """
        if hasattr(dataset, 'get_targets'):
            targets = len(np.unique(dataset.get_targets()))
        elif dataset.info is not None and isinstance(dataset.info.targets, int):
            targets = dataset.info.targets
        else:
            targets = 2
        modelargs.setdefault('targets', targets)
        print(
            f"Creating {cls.__name__} using: {len(dataset.channels)} channels x {dataset.sequence_length} samples at {dataset.sfreq}Hz | {modelargs['targets']} targets"
        )
        assert isinstance(dataset, DN3ataset)
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs)

    def __init__(self, targets, samples, channels, return_features=True):
        """
        Initializes the Classifier with the specified number of targets, samples, and channels.
        Sets up the classification layer and stores the initial state for resetting.

        Args:
            targets (int): Number of target classes for classification.
            samples (int): Number of samples per input instance.
            channels (int): Number of input channels.
            return_features (bool, optional): If True, the model returns features in addition to predictions. Defaults to True.
        """
        super(Classifier, self).__init__(samples, channels, return_features=return_features)
        self.targets = targets
        self.make_new_classification_layer()
        self._init_state = self.state_dict()

    def reset(self):
        """
        Resets the model's parameters to their initial state after construction.
        This is useful for reinitializing the model without reconstructing the object.

        Returns:
            None
        """
        self.load_state_dict(self._init_state)

    def forward(self, *x):
        """
        Performs a forward pass through the classifier, separating feature extraction and classification.
        Returns both the classification output and features if return_features is True, otherwise returns only the classification output.

        Args:
            *x: Input tensors to the model.

        Returns:
            tuple or torch.Tensor: (classification output, features) if return_features is True, else classification output only.
        """
        features = self.features_forward(*x)
        classification_output = self.classifier_forward(features)
        return (classification_output, features) if self.return_features else classification_output

    def make_new_classification_layer(self):
        """
        This allows for a distinction between the classification layer(s) and the rest of the network. Using a basic
        formulation of a network being composed of two parts feature_extractor & classifier.

        This method is for implementing the classification side, so that methods like :py:meth:`freeze_features` works
        as intended.

        Anything besides a layer that just flattens anything incoming to a vector and Linearly weights this to the
        target should override this method, and there should be a variable called `self.classifier`

        """
        classifier = nn.Linear(self.num_features_for_classification, self.targets)
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)

    def freeze_features(self, unfreeze=False, freeze_classifier=False):
        """
        In many cases, the features learned by a model in one domain can be applied to another case.

        This method freezes (or un-freezes) all but the `classifier` layer. So that any further training does not (or
        does if unfreeze=True) affect these weights.

        Parameters
        ----------
        unfreeze : bool
                   To unfreeze weights after a previous call to this.
        freeze_classifier: bool
                   Commonly, the classifier layer will not be frozen (default). Setting this to `True` will freeze this
                   layer too.
        """
        super(Classifier, self).freeze_features(unfreeze=unfreeze)

        if isinstance(self.classifier, nn.Module) and not freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    @property
    def num_features_for_classification(self):
        """
        Returns the number of features produced by the feature extraction part of the model.
        This property must be implemented by subclasses to specify the feature dimensionality.

        Returns:
            int: The number of features for classification.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    def classifier_forward(self, features):
        """
        Applies the classification layer(s) to the extracted features.
        This method returns the output of the classifier given the input features.

        Args:
            features (torch.Tensor): The feature tensor to classify.

        Returns:
            torch.Tensor: The classification output.
        """
        return self.classifier(features)

    def features_forward(self, x):
        """
        Extracts features from the input tensor for use in classification.
        This method must be implemented by subclasses to define the feature extraction process.

        Args:
            x (torch.Tensor): The input tensor to extract features from.

        Returns:
            torch.Tensor: The extracted feature tensor.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    def load(self, filename, include_classifier=False, freeze_features=True):
        """
        Loads model parameters from a file, with options to include or exclude the classifier and to freeze feature layers.
        Automatically selects the appropriate device for loading and updates the model's state dictionary.

        Args:
            filename (str): Path to the file containing the saved model state.
            include_classifier (bool, optional): If False, removes classifier weights from the loaded state. Defaults to False.
            freeze_features (bool, optional): If True, freezes feature extraction layers after loading. Defaults to True.
        """
        if torch.cuda.is_available():
            map_location = torch.device("cuda")
        elif torch.backends.mps.is_available():
            map_location = torch.device("mps")
        else:
            map_location = torch.device("cpu")
        state_dict = torch.load(filename, map_location=map_location)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        if freeze_features:
            self.freeze_features()

    def save(self, filename, ignore_classifier=False):
        """
        Saves the model's state dictionary to a file, with an option to exclude the classifier weights.
        Prints a message indicating the save location.

        Args:
            filename (str): Path to the file where the model state will be saved.
            ignore_classifier (bool, optional): If True, removes classifier weights from the saved state. Defaults to False.
        """
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        print(f"Saving to {filename} ...")
        torch.save(state_dict, filename)


class StrideClassifier(Classifier, metaclass=ABCMeta):

    def __init__(self, targets, samples, channels, stride_width=2, return_features=False):
        """
        Instead of summarizing the entire temporal dimension into a single prediction, a prediction kernel is swept over
        the final sequence representation and generates predictions at each step.

        Parameters
        ----------
        targets
        samples
        channels
        stride_width
        return_features
        """
        self.stride_width = stride_width
        super(StrideClassifier, self).__init__(targets, samples, channels, return_features=return_features)

    def make_new_classification_layer(self):
        """
        Creates a convolutional classification layer that generates predictions at each stride along the temporal dimension.
        Initializes the classifier weights using Xavier normal initialization and zeros the bias.

        Returns:
            None
        """
        self.classifier = torch.nn.Conv1d(self.num_features_for_classification, self.targets,
                                          kernel_size=self.stride_width)
        torch.nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()


class LogRegNetwork(Classifier):
    """
    In effect, simply an implementation of linear kernel (multi)logistic regression
    """
    def features_forward(self, x):
        """
        Returns the input tensor as features without modification.
        This is used in models where the input itself serves as the feature representation.

        Args:
            x (torch.Tensor): The input tensor to extract features from.

        Returns:
            torch.Tensor: The input tensor, unchanged.
        """
        return x

    @property
    def num_features_for_classification(self):
        """
        Returns the number of features produced by the feature extraction part of the model.
        For this model, it is the product of the number of samples and channels.

        Returns:
            int: The number of features for classification.
        """
        return self.samples * self.channels


class TIDNet(Classifier):
    """
    The Thinker Invariant Densenet from Kostas & Rudzicz 2020, https://doi.org/10.1088/1741-2552/abb7a7

    This alone is not strictly "thinker invariant", but on average outperforms shallower models at inter-subject
    prediction capability.
    """

    def __init__(self, targets, samples, channels, s_growth=24, t_filters=32, do=0.4, pooling=20,
                 activation=nn.LeakyReLU, temp_layers=2, spat_layers=2, temp_span=0.05, bottleneck=3,
                 summary=-1, return_features=False):
        """
        Initializes the TIDNet model for EEG classification with configurable temporal and spatial filtering layers.
        Sets up the temporal convolutional layers, spatial dense filters, and feature extraction pooling.

        Args:
            targets (int): Number of target classes for classification.
            samples (int): Number of samples per input instance.
            channels (int): Number of input channels.
            s_growth (int, optional): Growth rate for spatial dense layers. Defaults to 24.
            t_filters (int, optional): Number of temporal filters. Defaults to 32.
            do (float, optional): Dropout probability. Defaults to 0.4.
            pooling (int, optional): Pooling size for temporal max pooling. Defaults to 20.
            activation (callable, optional): Activation function for layers. Defaults to nn.LeakyReLU.
            temp_layers (int, optional): Number of temporal convolutional layers. Defaults to 2.
            spat_layers (int, optional): Number of spatial dense layers. Defaults to 2.
            temp_span (float, optional): Fraction of samples for temporal kernel length. Defaults to 0.05.
            bottleneck (int, optional): Bottleneck size for spatial dense layers. Defaults to 3.
            summary (int, optional): Output size after adaptive pooling. Defaults to -1 (computed from samples and pooling).
            return_features (bool, optional): If True, returns features in addition to predictions. Defaults to False.
        """
        self.temp_len = math.ceil(temp_span * samples)
        summary = samples // pooling if summary == -1 else summary
        self._num_features = (t_filters + s_growth * spat_layers) * summary
        super().__init__(targets, samples, channels, return_features=return_features)

        self.temporal = nn.Sequential(
            Expand(axis=1),
            TemporalFilter(1, t_filters, depth=temp_layers, temp_len=self.temp_len, activation=activation),
            nn.MaxPool2d((1, pooling)),
            nn.Dropout2d(do),
        )

        self.spatial = DenseSpatialFilter(self.channels, s_growth, spat_layers, in_ch=t_filters, dropout_rate=do,
                                          bottleneck=bottleneck, activation=activation)
        self.extract_features = nn.Sequential(
            nn.AdaptiveAvgPool1d(int(summary)),
        )

    @property
    def num_features_for_classification(self):
        """
        Returns the number of features produced by the feature extraction part of the model.
        For TIDNet and EEGNet models, this is determined by the architecture and stored in self._num_features.

        Returns:
            int: The number of features for classification.
        """
        return self._num_features

    def features_forward(self, x, **kwargs):
        """
        Extracts features from the input tensor using temporal and spatial filtering followed by pooling.
        Applies the temporal convolutional layers, spatial dense filters, and adaptive pooling to produce the feature representation.

        Args:
            x (torch.Tensor): The input tensor to extract features from.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The extracted feature tensor.
        """
        x = self.temporal(x)
        x = self.spatial(x)
        return self.extract_features(x)


class BaseEEGNet(Classifier):
    """
    A base class for EEGNet and EEGNetStrided to avoid code duplication.
    The EEGNet is a DN3 re-implementation of Lawhern et. al.'s EEGNet from:
    https://iopscience.iop.org/article/10.1088/1741-2552/aace8c

    Notes
    -----
    The implementation below is in no way officially sanctioned by the original authors, and in fact is missing the
    constraints the original authors have on the convolution kernels, and may or may not be missing more...
    That being said, in *our own personal experience*, this implementation has fared no worse when compared to
    implementations that include this constraint (albeit, those were *also not written* by the original authors).
    """

    def __init__(self, targets, samples, channels, do=0.25, pooling=8, F1=8, D=2, t_len=65, F2=16,
                 return_features=False):
        """
        Initializes the BaseEEGNet model with common parameters and layers.
        """
        if t_len >= samples:
            print("Warning: EEGNet `t_len` too long for sample length, reverting to 0.25 sample length")
            t_len = samples // 4
            t_len = t_len if t_len % 2 else t_len + 1

        self._num_features = F2 * (samples // (pooling // 2) // pooling)
        super().__init__(targets, samples, channels, return_features=return_features)

        self.init_conv = nn.Sequential(
            Expand(1),
            nn.Conv2d(1, F1, (1, t_len), padding=(0, t_len // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(F1, D * F1, (channels, 1), bias=False, groups=F1),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling // 2)),
            nn.Dropout(do)
        )

        self.sep_conv = nn.Sequential(
            # Separate into two convs, one that doesnt operate across filters, one isolated to filters
            nn.Conv2d(D * F1, D * F1, (1, 17), bias=False, padding=(0, 8), groups=D * F1),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling)),
            nn.Dropout(do)
        )

    @property
    def num_features_for_classification(self):
        """
        Returns the number of features produced by the feature extraction part of the model.
        For TIDNet and EEGNet models, this is determined by the architecture and stored in self._num_features.

        Returns:
            int: The number of features for classification.
        """
        return self._num_features

    def features_forward(self, x):
        """
        Extracts features from the input tensor using the EEGNet architecture.
        Applies initial, depthwise, and separable convolutional layers to produce the feature representation.

        Args:
            x (torch.Tensor): The input tensor to extract features from.

        Returns:
            torch.Tensor: The extracted feature tensor.
        """
        x = self.init_conv(x)
        x = self.depth_conv(x)
        return self.sep_conv(x)


class EEGNet(BaseEEGNet):
    """
    EEGNet implementation using the BaseEEGNet class.
    """
    def __init__(self, targets, samples, channels, do=0.25, pooling=8, F1=8, D=2, t_len=65, F2=16,
                 return_features=False):
        super().__init__(targets, samples, channels, do, pooling, F1, D, t_len, F2, return_features)


class EEGNetStrided(BaseEEGNet, StrideClassifier):
    """
    EEGNetStrided implementation using the BaseEEGNet and StrideClassifier classes.
    This model uses a strided classifier to generate predictions at each stride along the temporal dimension.
    The strided classifier is initialized with a stride width of 2, which can be adjusted as needed.
    The model is designed to work with EEG data and includes dropout, pooling, and convolutional layers for feature extraction.
    The features are extracted using the EEGNet architecture, and the output is squeezed to remove the temporal dimension.
    The model is initialized with parameters for the number of targets, samples, channels, dropout rate, pooling size,
    number of filters, depth, temporal length, and the number of features for classification.
    """
    def __init__(self, targets, samples, channels, do=0.25, pooling=8, F1=8, D=2, t_len=65, F2=16,
                 return_features=False, stride_width=2):
        """
        Initializes the EEGNetStrided model for EEG classification with a strided classifier and configurable convolutional layers.
        Sets up the initial, depthwise, and separable convolutional layers according to the EEGNet architecture, and configures strided output.

        Args:
            targets (int): Number of target classes for classification.
            samples (int): Number of samples per input instance.
            channels (int): Number of input channels.
            do (float, optional): Dropout probability. Defaults to 0.25.
            pooling (int, optional): Pooling size for temporal pooling. Defaults to 8.
            F1 (int, optional): Number of temporal filters in the initial convolution. Defaults to 8.
            D (int, optional): Depth multiplier for depthwise convolution. Defaults to 2.
            t_len (int, optional): Length of the temporal kernel. Defaults to 65.
            F2 (int, optional): Number of filters in the separable convolution. Defaults to 16.
            return_features (bool, optional): If True, returns features in addition to predictions. Defaults to False.
            stride_width (int, optional): Width of the stride for the classifier. Defaults to 2.
        """
        BaseEEGNet.__init__(self, targets, samples, channels, do, pooling, F1, D, t_len, F2, return_features)
        StrideClassifier.__init__(self, targets, samples, channels, stride_width=stride_width, return_features=return_features)
        self._num_features = F2  # Adjust for strided output

    def features_forward(self, x):
        """
        Extracts features and squeezes the temporal dimension for strided output.
        """
        x = super().features_forward(x)
        return x.squeeze(-2)


class BENDRClassifier(Classifier):
    """
    Implements a BENDR-based classifier for EEG data using a convolutional encoder and transformer-based contextualizer.
    The model extracts contextualized feature representations from input signals for downstream classification tasks.
    """

    def __init__(self, targets, samples, channels,
                 return_features=True,
                 encoder_h=256,
                 encoder_w=(3, 2, 2, 2, 2, 2),
                 encoder_do=0.,
                 projection_head=False,
                 encoder_stride=(3, 2, 2, 2, 2, 2),
                 hidden_feedforward=3076,
                 heads=8,
                 context_layers=8,
                 context_do=0.15,
                 activation='gelu',
                 position_encoder=25,
                 layer_drop=0.0,
                 mask_p_t=0.1,
                 mask_p_c=0.004,
                 mask_t_span=6,
                 mask_c_span=64,
                 start_token=-5,
                 **kwargs):
        """
        Initializes the BENDRClassifier with a convolutional encoder and transformer-based contextualizer.

        Args:
            targets (int): Number of target classes for classification.
            samples (int): Number of samples per input instance.
            channels (int): Number of input channels.
            return_features (bool, optional): If True, returns features in addition to predictions. Defaults to True.
            encoder_h (int, optional): Hidden size for the encoder and contextualizer. Defaults to 256.
            encoder_w (tuple, optional): Widths for the convolutional encoder layers. Defaults to (3, 2, 2, 2, 2, 2).
            encoder_do (float, optional): Dropout rate for the encoder. Defaults to 0.0.
            projection_head (bool, optional): Whether to use a projection head in the encoder. Defaults to False.
            encoder_stride (tuple, optional): Stride for the encoder layers. Defaults to (3, 2, 2, 2, 2, 2).
            hidden_feedforward (int, optional): Hidden size for the contextualizer feedforward layers. Defaults to 3076.
            heads (int, optional): Number of attention heads in the contextualizer. Defaults to 8.
            context_layers (int, optional): Number of transformer layers in the contextualizer. Defaults to 8.
            context_do (float, optional): Dropout rate for the contextualizer. Defaults to 0.15.
            activation (str, optional): Activation function for the contextualizer. Defaults to 'gelu'.
            position_encoder (int, optional): Size of the positional encoder. Defaults to 25.
            layer_drop (float, optional): Layer drop probability for the contextualizer. Defaults to 0.0.
            mask_p_t (float, optional): Probability of temporal masking. Defaults to 0.1.
            mask_p_c (float, optional): Probability of channel masking. Defaults to 0.004.
            mask_t_span (int, optional): Span of temporal masking. Defaults to 6.
            mask_c_span (int, optional): Span of channel masking. Defaults to 64.
            start_token (int, optional): Start token index for the contextualizer. Defaults to -5.
            **kwargs: Additional keyword arguments.
        """
        self._context_features = encoder_h
        super(BENDRClassifier, self).__init__(targets, samples, channels, return_features=return_features)
        self.encoder = ConvEncoderBENDR(in_features=channels, encoder_h=encoder_h, enc_width=encoder_w,
                                        dropout=encoder_do, projection_head=projection_head,
                                        enc_downsample=encoder_stride)
        self.contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=hidden_feedforward, heads=heads,
                                                  layers=context_layers, dropout=context_do, activation=activation,
                                                  position_encoder=position_encoder, layer_drop=layer_drop,
                                                  mask_p_t=mask_p_t, mask_p_c=mask_p_c, mask_t_span=mask_t_span,
                                                  finetuning=True)


    @property
    def num_features_for_classification(self):
        """
        Returns the number of features produced by the contextualizer for classification.

        Returns:
            int: The number of contextualized features for classification.
        """
        return self._context_features

    def easy_parallel(self):
        """
        Wraps the encoder, contextualizer, and classifier in DataParallel for multi-GPU training.
        """
        self.encoder = nn.DataParallel(self.encoder)
        self.contextualizer = nn.DataParallel(self.contextualizer)
        self.classifier = nn.DataParallel(self.classifier)

    def features_forward(self, x):
        """
        Extracts contextualized features from the input tensor using the encoder and contextualizer.
        Returns the first output of the contextualizer as the feature representation.

        Args:
            x (torch.Tensor): The input tensor to extract features from.

        Returns:
            torch.Tensor: The contextualized feature tensor.
        """
        x = self.encoder(x)
        x = self.contextualizer(x)
        return x[0]


