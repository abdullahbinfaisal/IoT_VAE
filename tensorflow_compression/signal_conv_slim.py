# ==============================================================================
"""Slimmable signal processing convolution layers (just for 2D image signal).
Modified for TensorFlow 2.x compatibility.
"""

import tensorflow as tf
import numpy as np
import types

# --- Minimal vendored parameterizers + padding ------------------------------
# Identity parameterizer (RDFT left as plain weight)  
class RDFTParameterizer:
    def __init__(self, **kwargs):
        pass
    def __call__(self, name, shape, dtype, getter, **kwargs):
        return getter(name, shape=shape, dtype=dtype, **kwargs)

# Nonnegative parameterizer  
class NonnegativeParameterizer:
    def __init__(self, minimum=0.0, **kwargs):
        self.minimum = minimum
    def __call__(self, name, shape, dtype, getter, **kwargs):
        v = getter(name, shape=shape, dtype=dtype, **kwargs)
        return tf.nn.relu(v) + self.minimum

# same padding function is still exposed under ops.padding_ops  
from tensorflow_compression.python.ops.padding_ops import same_padding_for_kernel

parameterizers = types.SimpleNamespace(
    RDFTParameterizer=RDFTParameterizer,
    NonnegativeParameterizer=NonnegativeParameterizer,
)
padding_ops = types.SimpleNamespace(
    same_padding_for_kernel=same_padding_for_kernel
)
# ----------------------------------------------------------------------------- 
# -----------------------------------------------------------------------------


class _SignalConv(tf.keras.layers.Layer):
    """N-D convolution layer for signal processing.

    This layer creates a filter kernel that is convolved or cross correlated with
    the layer input to produce an output tensor. The main difference of this class
    to `tf.keras.layers.Conv?D` is how padding, up- and downsampling, and alignment is
    handled.

    In general, the outputs are equivalent to a composition of:
    1. an upsampling step (if `strides_up > 1`)
    2. a convolution or cross correlation
    3. a downsampling step (if `strides_down > 1`)
    4. addition of a bias vector (if `use_bias == True`)
    5. a pointwise nonlinearity (if `activation is not None`)

    Arguments:
        filters: Integer. If `not channel_separable`, specifies the total number of
            filters, which is equal to the number of output channels. Otherwise,
            specifies the number of filters per channel, which makes the number of
            output channels equal to `filters` times the number of input channels.
        kernel_support: An integer or iterable of `rank` integers, specifying the
            length of the convolution/correlation window in each dimension.
        corr: Boolean. If True, compute cross correlation. If False, convolution.
        strides_down: An integer or iterable of `rank` integers, specifying an
            optional downsampling stride after the convolution/correlation.
        strides_up: An integer or iterable of `rank` integers, specifying an
            optional upsampling stride before the convolution/correlation.
        padding: String. One of the supported padding modes (see class docstring).
        extra_pad_end: Boolean. When upsampling, use extra skipped samples at the
            end of each dimension (default).
        channel_separable: Boolean. If `False` (default), each output channel is
            computed by summing over all filtered input channels. If `True`, each
            output channel is computed from only one input channel, and `filters`
            specifies the number of filters per channel.
        data_format: String, one of `channels_last` (default) or `channels_first`.
            The ordering of the input dimensions.
        activation: Activation function or `None`.
        use_bias: Boolean, whether an additive constant will be applied to each
            output channel.
        kernel_initializer: An initializer for the filter kernel.
        bias_initializer: An initializer for the bias vector.
        kernel_regularizer: Optional regularizer for the filter kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        kernel_parameterizer: Reparameterization applied to filter kernel.
        bias_parameterizer: Reparameterization applied to bias.
    """

    def __init__(self, rank, filters, kernel_support,
                 corr=False, strides_down=1, strides_up=1, padding="valid",
                 extra_pad_end=True, channel_separable=False,
                 data_format="channels_last",
                 activation=None, use_bias=False,
                 kernel_initializer='variance_scaling',  # Updated default initializer
                 bias_initializer='zeros',  # Updated default initializer
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_parameterizer=None,  # Changed default to None
                 bias_parameterizer=None,
                 **kwargs):
        # Call parent constructor
        super(_SignalConv, self).__init__(**kwargs)
        
        # Store parameters
        self._rank = int(rank)
        self._filters = int(filters)
        self._kernel_support = self._normalize_tuple(
            kernel_support, self._rank, "kernel_support")
        self._corr = bool(corr)
        self._strides_down = self._normalize_tuple(
            strides_down, self._rank, "strides_down")
        self._strides_up = self._normalize_tuple(
            strides_up, self._rank, "strides_up")
        if self._rank == 2:
            self._strides_up_list = self._pad_strides(self.strides_up)
        self._padding = str(padding).lower()
        
        # Check padding mode
        try:
            self._pad_mode = {
                "valid": None,
                "same_zeros": "CONSTANT",
                "same_reflect": "REFLECT",
            }[self.padding]
        except KeyError:
            raise ValueError(f"Unsupported padding mode: '{padding}'")
            
        self._extra_pad_end = bool(extra_pad_end)
        self._channel_separable = bool(channel_separable)
        self._data_format = self._normalize_data_format(data_format)
        self._activation = activation
        self._use_bias = bool(use_bias)
        
        # Convert string initializers to objects if needed
        if isinstance(kernel_initializer, str):
            self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        else:
            self._kernel_initializer = kernel_initializer
            
        if isinstance(bias_initializer, str):
            self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        else:
            self._bias_initializer = bias_initializer
            
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        
        # If no parameterizer is provided, use the default from TFC
        if kernel_parameterizer is None and hasattr(parameterizers, 'RDFTParameterizer'):
            self._kernel_parameterizer = parameterizers.RDFTParameterizer()
        else:
            self._kernel_parameterizer = kernel_parameterizer
            
        self._bias_parameterizer = bias_parameterizer
        
        # Initialize member variables
        self._kernel = None
        self._bias = None
        self.input_spec = tf.keras.layers.InputSpec(ndim=self._rank + 2)

    def _normalize_tuple(self, value, rank, name):
        """Converts `value` to a tuple of length `rank`."""
        if isinstance(value, int):
            return (value,) * rank
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(f"The {name} argument must be a tuple of length {rank}, "
                            f"but it is {value}.")
        if len(value_tuple) != rank:
            raise ValueError(f"The {name} argument must be a tuple of length {rank}, "
                            f"but it is {value_tuple}.")
        return value_tuple

    def _normalize_data_format(self, data_format):
        """Normalize data format string."""
        data_format = data_format.lower()
        if data_format not in ["channels_first", "channels_last"]:
            raise ValueError(f"Unknown data_format: {data_format}")
        return data_format

    @property
    def filters(self):
        return self._filters

    @property
    def kernel_support(self):
        return self._kernel_support

    @property
    def corr(self):
        return self._corr

    @property
    def strides_down(self):
        return self._strides_down

    @property
    def strides_up(self):
        return self._strides_up

    @property
    def padding(self):
        return self._padding

    @property
    def extra_pad_end(self):
        return self._extra_pad_end

    @property
    def channel_separable(self):
        return self._channel_separable

    @property
    def data_format(self):
        return self._data_format

    @property
    def activation(self):
        return self._activation

    @property
    def use_bias(self):
        return self._use_bias

    @property
    def kernel_initializer(self):
        return self._kernel_initializer

    @property
    def bias_initializer(self):
        return self._bias_initializer

    @property
    def kernel_regularizer(self):
        return self._kernel_regularizer

    @property
    def bias_regularizer(self):
        return self._bias_regularizer

    @property
    def kernel_parameterizer(self):
        return self._kernel_parameterizer

    @property
    def bias_parameterizer(self):
        return self._bias_parameterizer

    @property
    def kernel(self):
        return self._kernel

    @property
    def bias(self):
        return self._bias

    @property
    def _channel_axis(self):
        return {"channels_first": 1, "channels_last": -1}[self.data_format]

    def _pad_strides(self, strides):
        """Return a plain Python list for tf.nn.conv2d_transpose strides."""
        # strides may be an int or a 2‑tuple
        if isinstance(strides, int):
            sh = sw = strides
        else:
            sh, sw = strides
        # For NHWC conv2d_transpose we need [1, sh, sw, 1]
        return [1, int(sh), int(sw), 1]

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        channel_axis = self._channel_axis
        input_channels = input_shape[channel_axis]
        if input_channels is None:
            raise ValueError("The channel dimension of the inputs must be defined.")
            
        kernel_shape = self.kernel_support + (input_channels, self.filters)
        if self.channel_separable:
            output_channels = self.filters * input_channels
        else:
            output_channels = self.filters

        # Create kernel variable with potential parameterization
        if self.kernel_parameterizer is None:
            self._kernel = self.add_weight(
                name="kernel", 
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True)
        else:
            # For TF 2.x compatible parameterizers
            self._kernel = self.kernel_parameterizer(
                name="kernel", 
                shape=kernel_shape,
                dtype=self.dtype,
                getter=lambda name, *args, **kwargs: self.add_weight(
                    name=name, *args, **kwargs))

        # Create bias variable with potential parameterization
        if self.use_bias:
            if self.bias_parameterizer is None:
                self._bias = self.add_weight(
                    name="bias",
                    shape=(output_channels,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=True)
            else:
                # For TF 2.x compatible parameterizers
                self._bias = self.bias_parameterizer(
                    name="bias", 
                    shape=(output_channels,),
                    dtype=self.dtype,
                    getter=lambda name, *args, **kwargs: self.add_weight(
                        name=name, *args, **kwargs))
        else:
            self._bias = None

        self.input_spec = tf.keras.layers.InputSpec(
            ndim=self._rank + 2, axes={channel_axis: input_channels})
        
        # Mark layer as built
        self.built = True

    def call(self, inputs, mask_in=None, mask_out=None, training=None):
        """Forward pass.

        Args:
            inputs: Input tensor
            mask_in: Number of input channels to use (for slimmable networks)
            mask_out: Number of output channels to use (for slimmable networks)
            training: Whether in training mode (not used, included for compatibility)

        Returns:
            Output tensor
        """
        # Make defaults for masks if not provided
        if mask_in is None:
            if self.data_format == "channels_last":
                mask_in = inputs.shape[-1]
            else:
                mask_in = inputs.shape[1]
                
        if mask_out is None:
            mask_out = self.filters
        
        # Apply input mask by slicing the appropriate dimension
        if self.data_format == "channels_last":
            sliced_inputs = inputs[..., :mask_in]
        else:
            sliced_inputs = inputs[:, :mask_in, ...]
            
        # Convert to tensor
        outputs = tf.convert_to_tensor(sliced_inputs, dtype=self.dtype)
        input_shape = tf.shape(outputs)

        # First, perform any requested padding.
        if self.padding in ("same_zeros", "same_reflect"):
            padding = padding_ops.same_padding_for_kernel(
                self.kernel_support, self.corr, self.strides_up)
            if self.data_format == "channels_last":
                padding = [[0, 0]] + list(padding) + [[0, 0]]
            else:
                padding = [[0, 0], [0, 0]] + list(padding)
            outputs = tf.pad(outputs, padding, self._pad_mode)

        # Set up for convolution
        kernel = self.kernel
        corr = self.corr

        # If a convolution with no upsampling is desired, we flip the kernels and
        # use cross correlation to implement it
        if (not corr and
            all(s == 1 for s in self.strides_up) and
            all(s % 2 == 1 for s in self.kernel_support)):
            corr = True
            slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
            kernel = kernel[slices]

        # Similarly, implement cross correlation with no downsampling using convolutions
        if (corr and
            all(s == 1 for s in self.strides_down) and
            any(s != 1 for s in self.strides_up) and
            all(s % 2 == 1 for s in self.kernel_support)):
            corr = False
            slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
            kernel = kernel[slices]

        # Convert data format string to TF format
        data_format = "NHWC" if self.data_format == "channels_last" else "NCHW"
        
        # Apply appropriately masked kernel
        if self.data_format == "channels_last":
            masked_kernel = kernel[..., :mask_in, :mask_out]
        else:
            masked_kernel = kernel[..., :mask_in, :mask_out]

        # Perform the actual convolution or correlation operation
        if (corr and
            self.channel_separable and
            self._rank == 2 and
            all(s == 1 for s in self.strides_up) and
            all(s == self.strides_down[0] for s in self.strides_down)):
            # Use depthwise convolution for channel-separable correlations
            outputs = tf.nn.depthwise_conv2d(
                outputs, masked_kernel, 
                strides=self._pad_strides(self.strides_down),
                padding="VALID", data_format=data_format)
        elif (corr and
              all(s == 1 for s in self.strides_up) and
              not self.channel_separable):
            # Use convolution for standard correlations with potential downsampling
            outputs = tf.nn.convolution(
                outputs, masked_kernel, strides=self.strides_down, padding="VALID",
                data_format=data_format)
        elif (not corr and
              all(s == 1 for s in self.strides_down) and
              ((not self.channel_separable and 1 <= self._rank <= 3) or
               (self.channel_separable and self.filters == 1 and self._rank == 2 and
                all(s == self.strides_up[0] for s in self.strides_up)))):
            # Use transpose convolution for convolutions with potential upsampling
            
            # Transpose convolutions expect the output and input channels in reversed order
            if not self.channel_separable:
                kernel = tf.transpose(
                    masked_kernel, list(range(self._rank)) + [self._rank + 1, self._rank])
            else:
                kernel = masked_kernel

            # Compute shape of temporary output
            pad_shape = tf.shape(outputs)
            temp_shape = [pad_shape[0]] + (self._rank + 1) * [None]
            if self.data_format == "channels_last":
                spatial_axes = range(1, self._rank + 1)
                temp_shape[-1] = mask_out
            else:
                spatial_axes = range(2, self._rank + 2)
                temp_shape[1] = mask_out
                
            # Calculate output dimensions based on upsampling
            if self.extra_pad_end:
                get_length = lambda l, s, k: l * s + (k - 1)
            else:
                get_length = lambda l, s, k: l * s + (k - s)
                
            for i, a in enumerate(spatial_axes):
                temp_shape[a] = get_length(
                    pad_shape[a], self.strides_up[i], self.kernel_support[i])

            # Compute convolution based on rank
            if self._rank == 1 and not self.channel_separable:
                # Handle 1D case by using 2D ops
                extradim = 2 if self.data_format == "channels_first" else 1
                strides = self._pad_strides(self.strides_up)
                temp = tf.squeeze(
                    tf.nn.conv2d_transpose(
                        tf.expand_dims(outputs, extradim),
                        tf.expand_dims(kernel, 0),
                        tf.constant(temp_shape[:extradim] + [1] + temp_shape[extradim:]),
                        strides=strides[:extradim] + (1,) + strides[extradim:],
                        padding="VALID", data_format=data_format.replace("W", "HW")),
                    [extradim])
            elif self._rank == 2 and self.channel_separable:
                # Handle channel-separable 2D case
                temp = tf.nn.conv2d_transpose(
                    outputs, kernel, tf.constant(temp_shape),
                    strides=self._strides_up_list, padding="VALID",
                    data_format=data_format)
            elif self._rank == 2 and not self.channel_separable:
                # Handle standard 2D case
                # Build a vector of Tensors for output_shape:
                # pad_shape: [batch, H_after_pad, W_after_pad, C]
                # spatial_axes = [1,2] if channels_last else [2,3]
                if self.data_format == "channels_last":
                    # batch, height, width, channels
                    out_h = temp_shape[1]   # already computed as a Tensor
                    out_w = temp_shape[2]
                    output_shape = tf.stack([pad_shape[0], out_h, out_w, mask_out])
                else:
                    # NCHW: batch, channels, height, width
                    out_h = temp_shape[2]
                    out_w = temp_shape[3]
                    output_shape = tf.stack([pad_shape[0], mask_out, out_h, out_w])

                temp = tf.nn.conv2d_transpose(
                    outputs, kernel, output_shape,
                    strides=self._strides_up_list, padding="VALID",
                    data_format=data_format)
            elif self._rank == 3 and not self.channel_separable:
                # Handle 3D case
                temp = tf.nn.conv3d_transpose(
                    outputs, kernel, tf.constant(temp_shape),
                    strides=self._pad_strides(self.strides_up), padding="VALID",
                    data_format=data_format)
            else:
                raise NotImplementedError("Unsupported configuration")

            # Perform crop
            slices = [slice(None)] * (self._rank + 2)
            if self.padding == "valid":
                # Take kernel_support - 1 samples away from both sides
                for i, a in enumerate(spatial_axes):
                    slices[a] = slice(
                        self.kernel_support[i] - 1,
                        None if self.kernel_support[i] == 1 else
                        1 - self.kernel_support[i])
            else:
                # Take kernel_support // 2 plus padding away from beginning
                for i, a in enumerate(spatial_axes):
                    offset = padding[a][0] * self.strides_up[i]
                    offset += self.kernel_support[i] // 2
                    length = get_length(input_shape[a], self.strides_up[i], offset + 1)
                    slices[a] = slice(offset, length)
            outputs = temp[tuple(slices)]
        else:
            raise NotImplementedError(
                "The provided combination of SignalConv arguments is not implemented "
                f"(kernel_support={self.kernel_support}, corr={self.corr}, "
                f"strides_down={self.strides_down}, strides_up={self.strides_up}, "
                f"channel_separable={self.channel_separable}, filters={self.filters}).")

        # Add bias if requested
        if self.bias is not None:
            masked_bias = self.bias[:mask_out]
            if self.data_format == "channels_first":
                # Handle bias addition for channels_first format
                if self._rank == 1:
                    outputs = tf.expand_dims(outputs, 2)
                    outputs = tf.nn.bias_add(outputs, masked_bias, data_format="NCHW")
                    outputs = tf.squeeze(outputs, [2])
                elif self._rank == 2:
                    outputs = tf.nn.bias_add(outputs, masked_bias, data_format="NCHW")
                elif self._rank >= 3:
                    shape = tf.shape(outputs)
                    outputs = tf.reshape(outputs, shape[:3] + [-1])
                    outputs = tf.nn.bias_add(outputs, masked_bias, data_format="NCHW")
                    outputs = tf.reshape(outputs, shape)
            else:
                outputs = tf.nn.bias_add(outputs, masked_bias)

        # Apply activation function if requested
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        """Compute output tensor shape from input tensor shape."""
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(self._rank + 2)
        batch = input_shape[0]
        
        if self.data_format == "channels_first":
            spatial = input_shape[2:].dims
            channels = input_shape[1]
        else:
            spatial = input_shape[1:-1].dims
            channels = input_shape[-1]

        # Calculate spatial dimensions
        for i, s in enumerate(spatial):
            if s is not None:
                if self.extra_pad_end:
                    s *= self.strides_up[i]
                else:
                    s = (s - 1) * self.strides_up[i] + 1
                if self.padding == "valid":
                    s -= self.kernel_support[i] - 1
                s = (s - 1) // self.strides_down[i] + 1
                spatial[i] = s

        # Calculate channel dimension
        if self.channel_separable:
            channels *= self.filters
        else:
            channels = self.filters

        # Return shape
        if self.data_format == "channels_first":
            return tf.TensorShape([batch, channels] + spatial)
        else:
            return tf.TensorShape([batch] + spatial + [channels])
            
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super(_SignalConv, self).get_config()
        config.update({
            'rank': self._rank,
            'filters': self.filters,
            'kernel_support': self.kernel_support,
            'corr': self.corr,
            'strides_down': self.strides_down,
            'strides_up': self.strides_up,
            'padding': self.padding,
            'extra_pad_end': self.extra_pad_end,
            'channel_separable': self.channel_separable,
            'data_format': self.data_format,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            # Note: parameterizers might require custom serialization logic
        })
        return config


def _conv_class_factory(name, rank):
    """Subclass from _SignalConv, fixing convolution rank."""
    def init(self, *args, **kwargs):
        return _SignalConv.__init__(self, rank, *args, **kwargs)
    
    clsdict = {
        "__init__": init,
        "__doc__": _SignalConv.__doc__
    }
    return type(name, (_SignalConv,), clsdict)


# Create the specific SignalConv classes
SignalConv2D_slim = _conv_class_factory("SignalConv2D_slim", 2)