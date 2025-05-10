from .layers import *
from dn3.transforms.instance import InstanceTransform
from dn3.transforms.channels import map_named_channels_deep_1010, DEEP_1010_CHS_LISTING, SCALE_IND

from .processes import StandardClassification
from .models import Classifier


class TVector(Classifier):
    """
    Implements a temporal vector (T-Vector) extractor and classifier for EEG or similar time-series data.
    This class provides a feature extraction pipeline using temporal convolutional layers and supports classification and feature export.

    Args:
        num_target_people (int, optional): Number of target classes for classification. If None, disables classification. Defaults to None.
        channels (int, optional): Number of input channels. Defaults to len(DEEP_1010_CHS_LISTING).
        hidden_size (int, optional): Size of the hidden feature dimension. Defaults to 384.
        dropout (float, optional): Dropout probability for all layers. Defaults to 0.1.
        ignored_inds (tuple, optional): Indices of channels to ignore (e.g., scale channels). Defaults to (SCALE_IND,).
        incoming_channels (list, optional): List of incoming channel names for mapping. Defaults to None.
        norm_groups (int, optional): Number of groups for GroupNorm. Defaults to 16.
        return_tvectors (bool, optional): If True, returns T-vectors as features. Defaults to False.
    """

    def __init__(self, num_target_people=None, channels=len(DEEP_1010_CHS_LISTING), hidden_size=384, dropout=0.1,
                 ignored_inds=(SCALE_IND,), incoming_channels=None, norm_groups=16, return_tvectors=False):
        self.hidden_size = hidden_size
        self.num_target_people = num_target_people
        self.dropout = dropout
        super(TVector, self).__init__(num_target_people, None, channels, return_features=return_tvectors)
        self.ignored_ids = ignored_inds
        self.mapping = None if incoming_channels is None else map_named_channels_deep_1010(incoming_channels)

        def _make_td_layer(in_ch, out_ch, kernel, dilation):
            """
            Creates a temporal convolutional layer with ReLU activation, group normalization, and dropout.
            This helper function is used to construct the temporal feature extraction layers in the T-Vector model.

            Args:
                in_ch (int): Number of input channels.
                out_ch (int): Number of output channels.
                kernel (int): Size of the convolutional kernel.
                dilation (int): Dilation factor for the convolution.

            Returns:
                nn.Sequential: The constructed temporal convolutional layer.
            """
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation),
                nn.ReLU(),
                nn.GroupNorm(norm_groups, out_ch),
                nn.Dropout(dropout),
            )

        self.td_net = nn.Sequential(
            _make_td_layer(channels, hidden_size, 5, 1),
            _make_td_layer(hidden_size, hidden_size, 3, 2),
            _make_td_layer(hidden_size, hidden_size, 3, 3),
            _make_td_layer(hidden_size, hidden_size, 1, 1),
            _make_td_layer(hidden_size, hidden_size * 3, 1, 1),
        )

        # 3 * 2 -> td_net bottlenecks width at 3, 2 for mean and std pooling
        self.t_vector = self._make_ff_layer(hidden_size * 3 * 2, hidden_size)

    def _make_ff_layer(self, in_ch, out_ch):
        """
        Creates a feedforward layer consisting of a linear transformation, ReLU activation, layer normalization, and dropout.
        This helper function is used to construct the fully connected layers in the T-Vector model.

        Args:
            in_ch (int): Number of input features.
            out_ch (int): Number of output features.

        Returns:
            nn.Sequential: The constructed feedforward layer.
        """
        return nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(),
            nn.LayerNorm(out_ch),
            nn.Dropout(self.dropout),
        )

    def make_new_classification_layer(self):
        """
        Creates a new classification layer for the T-Vector model based on the number of target classes.
        If classification is enabled, sets up a feedforward network; otherwise, uses an identity mapping.

        Returns:
            None
        """
        if self.num_target_people is not None:
            self.classifier = nn.Sequential(
                self._make_ff_layer(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.num_target_people)
            )
        else:
            self.classifier = lambda x: x

    def load(self, filename, include_classifier=False, freeze_features=True):
        """
        Loads model parameters from a saved state dictionary file, with options to include the classifier and freeze features.
        This method updates the model's state and optionally removes classifier weights and freezes feature extraction layers.

        Args:
            filename (str): Path to the saved state dictionary file.
            include_classifier (bool, optional): Whether to include classifier weights. Defaults to False.
            freeze_features (bool, optional): Whether to freeze feature extraction layers. Defaults to True.
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
        self.freeze_features(unfreeze=not freeze_features)

    def save(self, filename, ignore_classifier=True):
        """
        Saves the model's state dictionary to a file, optionally excluding the classifier weights.
        This method serializes the model parameters for later loading or transfer.

        Args:
            filename (str): Path to the file where the state dictionary will be saved.
            ignore_classifier (bool, optional): If True, excludes classifier weights from the saved file. Defaults to True.
        """
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        torch.save(state_dict, filename)

    @property
    def num_features_for_classification(self):
        """
        Returns the number of features used for classification in the T-Vector model.
        This property provides the dimensionality of the feature vector output by the model.

        Returns:
            int: The number of features for classification.
        """
        return self.hidden_size

    def features_forward(self, x):
        """
        Extracts feature vectors (T-vectors) from the input data using the temporal convolutional network and pooling.
        This method processes the input, applies channel mapping and masking, and returns the pooled feature representation.

        Args:
            x (torch.Tensor): The input tensor of shape (channels, sequence_length) or (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: The extracted T-vector features.
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # Ignore e.g. scale values for the dataset to avoid easy identification
        if self.mapping is not None:
            x = (x.permute([0, 2, 1]) @ self.mapping).permute([0, 2, 1])
        x[:, self.ignored_ids, :] = 0
        for_pooling = self.td_net(x)
        pooled = torch.cat([for_pooling.mean(dim=-1), for_pooling.std(dim=-1)], dim=1)
        return self.t_vector(pooled)


class ClassificationWithTVectors(StandardClassification):
    """
    Combines a classifier and a T-Vector model for enhanced classification using both feature sets.
    This class builds a meta-classifier that integrates T-vectors and classifier features for improved performance.

    Args:
        classifier (Classifier): The main classifier model.
        tvector_model (TVector): The T-Vector feature extractor model.
        loss_fn (callable, optional): Loss function for training. Defaults to None.
        cuda (bool, optional): Whether to use CUDA for computation. Defaults to False.
        metrics (dict, optional): Metrics for evaluation. Defaults to None.
        learning_rate (float, optional): Learning rate for training. Defaults to None.
    """

    def __init__(self, classifier: Classifier, tvector_model: TVector, loss_fn=None, cuda=False, metrics=None,
                 learning_rate=None,):
        super(ClassificationWithTVectors, self).__init__(classifier=classifier, tvector_model=tvector_model,
                                                         loss_fn=loss_fn, cuda=cuda, metrics=metrics,
                                                         learning_rate=learning_rate)

    def build_network(self, **kwargs):
        """
        Builds the meta-classifier network that combines T-vector and classifier features for final prediction.
        This method initializes the attention parameter and meta-classifier layers, and ensures the T-vector model is in evaluation mode.

        Args:
            **kwargs: Additional keyword arguments for network building.
        """
        super(ClassificationWithTVectors, self).build_network(**kwargs)
        self.tvector_model.train(False)
        incoming = self.classifier.num_features_for_classification
        self.attn_tvect = torch.nn.Parameter(torch.ones((self.tvector_model.num_features_for_classification,
                                                         self.classifier.num_features_for_classification),
                                                        requires_grad=True, device=self.device))
        self.meta_classifier = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(incoming),
            # nn.Linear(incoming, incoming // 4),
            # nn.Dropout(0.6),
            # nn.Sigmoid(),
            nn.Linear(incoming, self.classifier.targets)
        )

    def train_step(self, *inputs):
        """
        Performs a single training step for the meta-classifier, ensuring it is in training mode.
        Delegates the actual training logic to the base StandardClassification class.

        Args:
            *inputs: Input data for the training step.

        Returns:
            The result of the training step as returned by the base class.
        """
        self.meta_classifier.train(True)
        return super(StandardClassification, self).train_step(*inputs)

    def evaluate(self, dataset, **loader_kwargs):
        """
        Evaluates the model on the provided dataset, ensuring T-Vector and meta-classifier are in evaluation mode.
        Returns the evaluation results from the base StandardClassification class.

        Args:
            dataset: The dataset to evaluate on.
            **loader_kwargs: Additional keyword arguments for the data loader.

        Returns:
            The evaluation results as returned by the base class.
        """
        self.tvector_model.train(False)
        self.meta_classifier.train(False)
        return super(ClassificationWithTVectors, self).evaluate(dataset)

    def parameters(self):
        """
        Returns an iterator over all parameters of the model, including the meta-classifier and attention parameters.
        This method ensures that all trainable parameters are included for optimization.

        Returns:
            Iterator[torch.nn.Parameter]: An iterator over model parameters.
        """
        yield from super(ClassificationWithTVectors, self).parameters()
        yield from self.meta_classifier.parameters()
        yield self.attn_tvect

    def forward(self, *inputs):
        """
        Combines classifier features and T-vectors, applies attention, and passes the result through the meta-classifier.
        Returns the final classification output for the input batch.

        Args:
            *inputs: Input data for the model, typically a batch of samples.

        Returns:
            torch.Tensor: The output of the meta-classifier after combining features.
        """
        batch_size = inputs[0].shape[0]
        _, classifier_features = self.classifier(inputs[0].clone())
        _, t_vectors = self.tvector_model(inputs[0])
        added = classifier_features.view(batch_size, -1) + t_vectors.view(batch_size, -1) @ self.attn_tvect
        return self.meta_classifier(added)


class TVectorConcatenation(InstanceTransform):
    """
    Concatenates T-vector features to the input data for each instance, augmenting the input with learned representations.
    This transform is useful for enriching input data with T-vectors before passing to downstream models.

    Args:
        t_vector_model (TVector or str): A TVector instance or path to a saved TVector model.
    """

    def __init__(self, t_vector_model):
        if isinstance(t_vector_model, TVector):
            self.tvect = t_vector_model
        elif isinstance(t_vector_model, str):
            print(f"Loading T-Vectors model from path: {t_vector_model}")
            self.tvect = TVector()
            self.tvect.load(t_vector_model)
        # ensure not in training mode
        self.tvect.train(False)
        for p in self.tvect.parameters():
            p.requires_grad = False
        super(TVectorConcatenation, self).__init__()

    def __call__(self, x):
        """
        Concatenates the T-vector features to the input tensor along the channel dimension.
        Returns the augmented tensor with T-vectors prepended to the original data.

        Args:
            x (torch.Tensor): Input tensor of shape (channels, sequence_length).

        Returns:
            torch.Tensor: Augmented tensor with T-vectors concatenated.
        """
        channels, sequence_length = x.shape
        tvector = self.tvect.features_forward(x).view(-1, 1)
        return torch.cat((tvector.expand(-1, sequence_length), x), dim=0)

    def new_channels(self, old_channels):
        """
        Returns the updated list of channel names after T-vector concatenation.
        Appends T-vector channel names to the original channel list.

        Args:
            old_channels (list): List of original channel names.

        Returns:
            list: Updated list of channel names including T-vectors.
        """
        return old_channels + [f'T-vectors-{i+1}' for i in range(self.tvect.num_features_for_classification)]

