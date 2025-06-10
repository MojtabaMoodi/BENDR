import copy
import torch
import numpy as np
from torch import nn
from dn3.trainable.utils import _make_mask


class _SingleAxisOperation(nn.Module):
    """
    Base class for operations that act along a single axis of a tensor.
    Stores the axis to operate on and requires subclasses to implement the forward method.

    Args:
        axis (int, optional): The axis along which to operate. Defaults to -1.
    """
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        raise NotImplementedError

# Some general purpose convenience layers
# ---------------------------------------


class Expand(_SingleAxisOperation):
    """
    Expands the input tensor by adding a new dimension at the specified axis.
    This is useful for preparing tensors for operations that require an additional dimension, such as convolution.

    Args:
        x (torch.Tensor): The input tensor to expand.

    Returns:
        torch.Tensor: The expanded tensor with a new dimension at the specified axis.
    """
    def forward(self, x):
        return x.unsqueeze(self.axis)


class Squeeze(_SingleAxisOperation):
    """
    Removes a dimension from the input tensor at the specified axis.
    This is useful for reducing the dimensionality of tensors after operations that add singleton dimensions.

    Args:
        x (torch.Tensor): The input tensor to squeeze.

    Returns:
        torch.Tensor: The squeezed tensor with the specified axis removed.
    """
    def forward(self, x):
        return x.squeeze(self.axis)


class Permute(nn.Module):
    """
    Permutes the dimensions of the input tensor according to the specified axes order.
    This is useful for rearranging tensor dimensions to match the requirements of different layers or operations.

    Args:
        axes (tuple or list): The desired order of axes for permutation.

    Returns:
        torch.Tensor: The permuted tensor.
    """
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)


class Concatenate(_SingleAxisOperation):
    """
    Concatenates multiple input tensors along a specified axis.
    This is useful for combining tensors along a particular dimension, such as stacking feature maps or batch elements.

    Args:
        *x: Input tensors to concatenate. Can be passed as separate arguments or as a single tuple.

    Returns:
        torch.Tensor: The concatenated tensor along the specified axis.
    """
    def forward(self, *x):
        if len(x) == 1 and isinstance(x[0], tuple):
            x = x[0]
        return torch.cat(x, dim=self.axis)


class IndexSelect(nn.Module):
    """
    Selects elements from the input tensors based on specified indices.
    Returns either a single element or a list of elements, depending on the number of indices provided.

    Args:
        indices (int, list, or tuple): Indices of elements to select from the input.

    Returns:
        list or object: The selected elements from the input tensors.
    """
    def __init__(self, indices):
        super().__init__()
        assert isinstance(indices, (int, list, tuple))
        self.indices = list(indices) if isinstance(indices, (list, tuple)) else [indices]

    def forward(self, *x):
        if len(x) == 1 and isinstance(x[0], tuple):
            x = x[0]
        return [x[i] for i in self.indices] if len(self.indices) > 1 else x[self.indices[0]]


class Flatten(nn.Module):
    """
    Flattens the input tensor except for the batch dimension.
    This is useful for preparing tensors for fully connected layers by collapsing all non-batch dimensions into one.

    Args:
        x (torch.Tensor): The input tensor to flatten.

    Returns:
        torch.Tensor: The flattened tensor with shape (batch_size, -1).
    """
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class ConvBlock2D(nn.Module):
    """
    Implements a 2D convolutional block with optional batch normalization, activation, dropout, and residual connection.
    This block is commonly used for feature extraction in deep learning models, especially for EEG and time-series data.

    Args:
        in_filters (int): Number of input channels.
        out_filters (int): Number of output channels.
        kernel (tuple): Size of the convolutional kernel.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1).
        padding (int or tuple, optional): Padding added to both sides of the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        do_rate (float, optional): Dropout probability. Defaults to 0.5.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        activation (callable, optional): Activation function class. Defaults to nn.LeakyReLU.
        residual (bool, optional): Whether to add a residual connection. Defaults to False.

    Returns:
        torch.Tensor: The output tensor after applying convolution, activation, dropout, batch normalization, and optional residual connection.
    """
    def __init__(self, in_filters, out_filters, kernel, stride=(1, 1), padding=0, dilation=1, groups=1, do_rate=0.5,
                 batch_norm=True, activation=nn.LeakyReLU, residual=False):
        super().__init__()
        self.kernel = kernel
        self.activation = activation()
        self.residual = residual

        self.conv = nn.Conv2d(in_filters, out_filters, kernel, stride=stride, padding=padding, dilation=dilation,
                           groups=groups, bias=not batch_norm)
        self.dropout = nn.Dropout2d(p=do_rate)
        self.batch_norm = nn.BatchNorm2d(out_filters) if batch_norm else nn.Identity()

    def forward(self, x, **kwargs):
        res = x
        x = self.conv(x, **kwargs)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x + res if self.residual else x


class DenseFilter(nn.Module):
    """
    Implements a dense convolutional filter block for feature extraction along a specified dimension.
    Applies batch normalization, activation, bottleneck convolution, and dropout, then concatenates the result with the input.
    This DenseNet-inspired filter block features in the TIDNet network from Kostas & Rudzicz 2020 (Thinker
    Invariance). 2D convolution is used, but with a kernel that only spans one of the dimensions. In TIDNet it is
    used to develop channel operations independently of temporal changes.

    Args:
        in_features (int): Number of input channels.
        growth_rate (int): Number of output channels to add per layer.
        filter_len (int, optional): Length of the convolutional filter. Defaults to 5.
        do (float, optional): Dropout probability. Defaults to 0.5.
        bottleneck (int, optional): Bottleneck size for intermediate convolution. Defaults to 2.
        activation (callable, optional): Activation function class. Defaults to nn.LeakyReLU.
        dim (int, optional): Dimension along which to apply the filter. Defaults to -2.

    Returns:
        torch.Tensor: The concatenated tensor of input and new features.
    """
    def __init__(self, in_features, growth_rate, filter_len=5, do=0.5, bottleneck=2, activation=nn.LeakyReLU, dim=-2):
        super().__init__()
        dim = dim if dim > 0 else dim + 4
        if dim < 2 or dim > 3:
            raise ValueError('Only last two dimensions supported')
        kernel = (filter_len, 1) if dim == 2 else (1, filter_len)

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_features),
            activation(),
            nn.Conv2d(in_features, bottleneck * growth_rate, 1),
            nn.BatchNorm2d(bottleneck * growth_rate),
            activation(),
            nn.Conv2d(bottleneck * growth_rate, growth_rate, kernel, padding=tuple((k // 2 for k in kernel))),
            nn.Dropout2d(do)
        )

    def forward(self, x):
        return torch.cat((x, self.net(x)), dim=1)


class DenseSpatialFilter(nn.Module):
    """
    Implements a dense spatial filtering block using a sequence of DenseFilter layers.
    Optionally collapses the channel dimension at the end using a convolutional block.
    This extends the :any:`DenseFilter` to specifically operate in channel space and collapse this dimension
    over the course of `depth` layers.

    Args:
        channels (int): Number of input channels.
        growth (int): Growth rate for each DenseFilter layer.
        depth (int): Number of DenseFilter layers to stack.
        in_ch (int, optional): Number of input channels to the first DenseFilter. Defaults to 1.
        bottleneck (int, optional): Bottleneck size for DenseFilter layers. Defaults to 4.
        dropout_rate (float, optional): Dropout probability for DenseFilter layers. Defaults to 0.0.
        activation (callable, optional): Activation function class. Defaults to nn.LeakyReLU.
        collapse (bool, optional): Whether to collapse the channel dimension at the end. Defaults to True.

    Returns:
        torch.Tensor: The output tensor after dense spatial filtering and optional channel collapse.
    """
    def __init__(self, channels, growth, depth, in_ch=1, bottleneck=4, dropout_rate=0.0, activation=nn.LeakyReLU,
                 collapse=True):
        super().__init__()
        self.net = nn.Sequential(*[
            DenseFilter(in_ch + growth * d, growth, bottleneck=bottleneck, do=dropout_rate,
                        activation=activation) for d in range(depth)
        ])
        n_filters = in_ch + growth * depth
        self.collapse = collapse
        self.channel_collapse = ConvBlock2D(n_filters, n_filters, (channels, 1), do_rate=0) if collapse else None

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.net(x)
        return self.channel_collapse(x).squeeze(-2) if self.collapse else x


class SpatialFilter(nn.Module):
    """
    Implements a spatial filtering block using a sequence of ConvBlock2D layers.
    Optionally applies a residual connection using a 1D convolution if specified.

    Args:
        channels (int): Number of input channels.
        filters (int): Number of output filters for each ConvBlock2D.
        depth (int): Number of ConvBlock2D layers to stack.
        in_ch (int, optional): Number of input channels to the first ConvBlock2D. Defaults to 1.
        dropout_rate (float, optional): Dropout probability for ConvBlock2D layers. Defaults to 0.0.
        activation (callable, optional): Activation function class. Defaults to nn.LeakyReLU.
        batch_norm (bool, optional): Whether to use batch normalization in ConvBlock2D layers. Defaults to True.
        residual (bool, optional): Whether to add a residual connection. Defaults to False.

    Returns:
        torch.Tensor: The output tensor after spatial filtering and optional residual connection.
    """
    def __init__(self, channels, filters, depth, in_ch=1, dropout_rate=0.0, activation=nn.LeakyReLU, batch_norm=True,
                 residual=False):
        super().__init__()
        kernels = [(channels // depth, 1) for _ in range(depth-1)]
        kernels += [(channels - sum(x[0] for x in kernels) + depth-1, 1)]
        self.filter = nn.Sequential(
            ConvBlock2D(in_ch, filters, kernels[0], do_rate=dropout_rate/depth, activation=activation,
                        batch_norm=batch_norm),
            *[ConvBlock2D(filters, filters, kernel, do_rate=dropout_rate/depth, activation=activation,
                          batch_norm=batch_norm)
              for kernel in kernels[1:]]
        )
        self.residual = nn.Conv1d(channels * in_ch, filters, 1) if residual else None

    def forward(self, x):
        res = x
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        elif self.residual:
            res = res.contiguous().view(res.shape[0], -1, res.shape[3])
        x = self.filter(x).squeeze(-2)
        return x + self.residual(res) if self.residual else x


class TemporalFilter(nn.Module):
    """
    Implements a temporal filtering block using a sequence of dilated Conv2d layers.
    Supports both netwise residual and dense connection styles for temporal feature extraction.
    (This implements the dilated temporal-only spanning convolution from TIDNet.)

    Args:
        channels (int): Number of input channels.
        filters (int): Number of output filters for each Conv2d layer.
        depth (int): Number of Conv2d layers to stack.
        temp_len (int): Length of the temporal convolutional kernel.
        dropout (float, optional): Dropout probability for Conv2d layers. Defaults to 0.0.
        activation (callable, optional): Activation function class. Defaults to nn.LeakyReLU.
        residual (str, optional): Residual connection style, either 'netwise' or 'dense'. Defaults to 'netwise'.

    Returns:
        torch.Tensor: The output tensor after temporal filtering and optional residual or dense connections.
    """

    def __init__(self, channels, filters, depth, temp_len, dropout=0., activation=nn.LeakyReLU, residual='netwise'):
        super().__init__()
        temp_len = temp_len + 1 - temp_len % 2
        self.residual_style = str(residual)
        net = []

        for i in range(depth):
            dil = depth - i
            conv = nn.utils.weight_norm(nn.Conv2d(channels if i == 0 else filters, filters, kernel_size=(1, temp_len),
                                      dilation=dil, padding=(0, dil * (temp_len - 1) // 2)))
            net.append(nn.Sequential(
                conv,
                activation(),
                nn.Dropout2d(dropout)
            ))
        if self.residual_style.lower() == 'netwise':
            self.net = nn.Sequential(*net)
            self.residual = nn.Conv2d(channels, filters, (1, 1))
        elif residual.lower() == 'dense':
            self.net = net

    def forward(self, x):
        if self.residual_style.lower() == 'netwise':
            return self.net(x) + self.residual(x)
        elif self.residual_style.lower() == 'dense':
            for l in self.net:
                x = torch.cat((x, l(x)), dim=1)
            return x



class _BENDREncoder(nn.Module):
    """
    Base class for BENDR encoder modules that provides loading, saving, and feature freezing utilities.
    This class is intended to be subclassed for specific encoder architectures.

    Args:
        in_features (int): Number of input features.
        encoder_h (int, optional): Hidden dimension size for the encoder. Defaults to 256.
    """
    def __init__(self, in_features, encoder_h=256):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h

    def load(self, filename, strict=True):
        if torch.cuda.is_available():
            map_location = torch.device("cuda")
        elif torch.backends.mps.is_available():
            map_location = torch.device("mps")
        else:
            map_location = torch.device("cpu")
        state_dict = torch.load(filename, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze


class ConvEncoderBENDR(_BENDREncoder):
    """
    Implements a convolutional encoder for BENDR models using a configurable stack of 1D convolutional layers.
    This encoder extracts hierarchical temporal features from input sequences and supports optional projection heads.

    Args:
        in_features (int): Number of input features.
        encoder_h (int, optional): Hidden dimension size for the encoder. Defaults to 256.
        enc_width (tuple or list, optional): Kernel widths for each convolutional layer. Defaults to (3, 2, 2, 2, 2, 2).
        dropout (float, optional): Dropout probability for convolutional layers. Defaults to 0.0.
        projection_head (bool, optional): Whether to add a projection head at the end. Defaults to False.
        enc_downsample (tuple or list, optional): Downsampling factors for each convolutional layer. Defaults to (3, 2, 2, 2, 2, 2).
    """
    def __init__(self, in_features, encoder_h=256, enc_width=(3, 2, 2, 2, 2, 2),
                 dropout=0., projection_head=False, enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__(in_features, encoder_h)
        self.encoder_h = encoder_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        # Centerable convolutions make life simpler
        enc_width = [e if e % 2 else e+1 for e in enc_width]
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module(f"Encoder_{i}", nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout2d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h

        if projection_head:
            self.encoder.add_module("projection-1", nn.Sequential(
                nn.Conv1d(in_features, in_features, 1),
                nn.Dropout2d(dropout*2),
                nn.GroupNorm(in_features // 2, in_features),
                nn.GELU()
            ))

    def description(self, sfreq=None, sequence_len=None):
        """
        Returns a string describing the receptive field, downsampling, and overlap characteristics of the encoder.
        This method provides a summary of how the encoder processes input sequences and the resulting output dimensions.

        Args:
            sfreq (float, optional): Sampling frequency of the input data. Defaults to None.
            sequence_len (int, optional): Length of the input sequence. Defaults to None.

        Returns:
            str: A description of the encoder's receptive field, downsampling factor, overlap, and output samples.
        """
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = f"Receptive field: {rf} samples"
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += f" | Downsampled by {ds_factor}"
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += f" | Overlap of {rf - ds_factor} samples"
        if sequence_len is not None:
            desc += f" | {sequence_len // ds_factor} encoded samples/trial"
        return desc

    def downsampling_factor(self, samples):
        """
        Calculates the total downsampling factor applied by the encoder to the input sequence.
        Returns the number of samples remaining after all downsampling operations.

        Args:
            samples (int or float): The original number of input samples.

        Returns:
            float: The number of samples after downsampling.
        """
        for factor in self._downsampling:
            samples = np.ceil(samples / factor)
        return samples

    def forward(self, x):
        """
        Passes the input through the encoder network and returns the encoded output.
        This method applies the sequential convolutional layers to extract features from the input.

        Args:
            x (torch.Tensor): The input tensor to encode.

        Returns:
            torch.Tensor: The encoded output tensor.
        """
        return self.encoder(x)


# FIXME this is redundant with part of the contextualizer
class EncodingAugment(nn.Module):
    """
    Applies data augmentation to encoded representations by masking temporal and channel dimensions and adding relative positional encoding.
    This module is useful for regularizing transformer-based models by simulating missing data and enhancing positional awareness.

    Args:
        in_features (int): Number of input features.
        mask_p_t (float, optional): Probability of masking a temporal position. Defaults to 0.1.
        mask_p_c (float, optional): Probability of masking a channel. Defaults to 0.01.
        mask_t_span (int, optional): Span of temporal masking. Defaults to 6.
        mask_c_span (int, optional): Span of channel masking. Defaults to 64.
        dropout (float, optional): Dropout probability for input conditioning. Defaults to 0.1.
        position_encoder (int, optional): Kernel size for relative position encoding. Defaults to 25.
    """
    def __init__(self, in_features, mask_p_t=0.1, mask_p_c=0.01, mask_t_span=6, mask_c_span=64, dropout=0.1,
                 position_encoder=25):
        super().__init__()
        self.mask_replacement = torch.nn.Parameter(torch.zeros(in_features), requires_grad=True)
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        transformer_dim = 3 * in_features

        conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
        nn.init.normal_(conv.weight, mean=0, std=2 / transformer_dim)
        nn.init.constant_(conv.bias, 0)
        conv = nn.utils.weight_norm(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, transformer_dim, 1),
        )

    def forward(self, x, mask_t=None, mask_c=None):
        """
        Applies temporal and channel masking, relative positional encoding, and input conditioning to the input tensor.
        This method augments the encoded representations for regularization and improved positional awareness.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, features, sequence_length).
            mask_t (torch.Tensor, optional): Boolean mask for temporal positions. If None, generated during training.
            mask_c (torch.Tensor, optional): Boolean mask for channels. If None, generated during training.

        Returns:
            torch.Tensor: The augmented and conditioned tensor.
        """
        bs, feat, seq = x.shape

        if self.training:
            if mask_t is None and self.p_t > 0 and self.mask_t_span > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0 and self.mask_c_span > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        x = self.input_conditioning(x + self.relative_position(x))
        return x

    def init_from_contextualizer(self, filename):
        """
        Initializes the mask embedding and position encoder from a saved contextualizer checkpoint.
        Loads parameters from the specified file and freezes them to prevent further training.

        Args:
            filename (str): Path to the saved contextualizer checkpoint file.
        """
        if torch.cuda.is_available():
            map_location = torch.device("cuda")
        elif torch.backends.mps.is_available():
            map_location = torch.device("mps")
        else:
            map_location = torch.device("cpu")
        state_dict = torch.load(filename, map_location=map_location)
        self.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        print("Initialized mask embedding and position encoder from ", filename)


class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BENDRContextualizer(nn.Module):
    """
    Implements a transformer-based contextualizer for BENDR models, supporting masking, relative positional encoding, and layer dropout.
    This class is designed to process sequential data with self-attention and is suitable for pretraining and finetuning on EEG or similar time-series data.

    Args:
        in_features (int): Number of input features.
        hidden_feedforward (int, optional): Size of the feedforward layer in the transformer. Defaults to 3076.
        heads (int, optional): Number of attention heads. Defaults to 8.
        layers (int, optional): Number of transformer encoder layers. Defaults to 8.
        dropout (float, optional): Dropout probability for transformer and input conditioning. Defaults to 0.15.
        activation (str, optional): Activation function for the transformer. Defaults to 'gelu'.
        position_encoder (int, optional): Kernel size for relative position encoding. Defaults to 25.
        layer_drop (float, optional): Probability of dropping a transformer layer during training. Defaults to 0.0.
        mask_p_t (float, optional): Probability of masking a temporal position. Defaults to 0.1.
        mask_p_c (float, optional): Probability of masking a channel. Defaults to 0.004.
        mask_t_span (int, optional): Span of temporal masking. Defaults to 6.
        mask_c_span (int, optional): Span of channel masking. Defaults to 64.
        start_token (int, optional): Value for the start token. Defaults to -5.
        finetuning (bool, optional): Whether the model is in finetuning mode. Defaults to False.
    """

    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8, dropout=0.15, activation='gelu',
                 position_encoder=25, layer_drop=0.0, mask_p_t=0.1, mask_p_c=0.004, mask_t_span=6, mask_c_span=64,
                 start_token=-5, finetuning=False):
        super(BENDRContextualizer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3

        encoder = nn.TransformerEncoderLayer(d_model=in_features * 3, nhead=heads, dim_feedforward=hidden_feedforward,
                                             dropout=dropout, activation=activation)
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim)

        # self.norm_layers = nn.ModuleList([copy.deepcopy(norm) for _ in range(layers)])
        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # Initialize replacement vector with 0's
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)),
                                                   requires_grad=True)

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        """
        Initializes the parameters of BERT-style layers using Xavier uniform initialization and T-Fixup scaling.
        This method is applied to each module in the model to ensure proper initialization for stable training.

        Args:
            module (nn.Module): The module whose parameters will be initialized.
        """
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

        # if isinstance(module, nn.Conv1d):
        #     # std = np.sqrt((4 * (1.0 - self.dropout)) / (self.in_features * self.in_features))
        #     # module.weight.data.normal_(mean=0.0, std=std)
        #     nn.init.xavier_uniform_(module.weight.data)
        #     module.bias.data.zero_()

    def forward(self, x, mask_t=None, mask_c=None):
        """
        Processes the input tensor through the BENDR transformer-based contextualizer, applying masking, positional encoding, and transformer layers.
        Returns the contextualized output suitable for downstream tasks such as classification or regression.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, features, sequence_length).
            mask_t (torch.Tensor, optional): Boolean mask for temporal positions. If None, generated during training and finetuning.
            mask_c (torch.Tensor, optional): Boolean mask for channels. If None, generated during training and finetuning.

        Returns:
            torch.Tensor: The contextualized output tensor after transformer processing.
        """
        bs, feat, seq = x.shape
        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        # Multi-gpu workaround, wastes memory
        x = x.clone()

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(device=x.device, dtype=x.dtype).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))

    def freeze_features(self, unfreeze=False, finetuning=False):
        """
        Sets the requires_grad attribute for all parameters to control feature freezing during training or finetuning.
        Optionally keeps the mask replacement parameter frozen if in finetuning mode.

        Args:
            unfreeze (bool, optional): If True, unfreezes all parameters. If False, freezes them. Defaults to False.
            finetuning (bool, optional): If True, keeps the mask replacement parameter frozen. Defaults to False.
        """
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False

    def load(self, filename, strict=True):
        """
        Loads model parameters from a saved state dictionary file.
        Automatically selects the appropriate device for loading and updates the model's state.

        Args:
            filename (str): Path to the saved state dictionary file.
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
        This method serializes the model parameters for later loading or transfer.

        Args:
            filename (str): Path to the file where the state dictionary will be saved.
        """
        torch.save(self.state_dict(), filename)
