"""SlimGDN+ layer for TensorFlow 2.x."""

import tensorflow as tf
import numpy as np
import types

# --- Vendored parameterizer for TFC 2.x ----------------------
class NonnegativeParameterizer:
    def __init__(self, minimum=0.0):
        self.minimum = minimum
    def __call__(self, name, shape, dtype, getter, **kwargs):
        v = getter(name, shape=shape, dtype=dtype, **kwargs)
        return tf.nn.relu(v) + self.minimum

parameterizers = types.SimpleNamespace(
    NonnegativeParameterizer=NonnegativeParameterizer,
)
# ----------------------------------------------------------------

def _get_default_parameterizers():
    """Create default parameterizers with appropriate settings."""
    _default_beta_param = parameterizers.NonnegativeParameterizer(minimum=1e-6)
    _default_gamma_param = parameterizers.NonnegativeParameterizer()
    _default_scale_gamma = parameterizers.NonnegativeParameterizer()
    _default_bias_gamma = parameterizers.NonnegativeParameterizer()
    _default_scale_beta = parameterizers.NonnegativeParameterizer()
    _default_bias_beta = parameterizers.NonnegativeParameterizer()
    return (_default_beta_param, _default_gamma_param, _default_scale_gamma,
            _default_bias_gamma, _default_scale_beta, _default_bias_beta)

class GDN(tf.keras.layers.Layer):
    """Generalized divisive normalization layer for slimmable networks.
    
    Based on the papers:

    > "Density modeling of images using a generalized normalization
    > transformation"<br />
    > J. Ballé, V. Laparra, E.P. Simoncelli<br />
    > https://arxiv.org/abs/1511.06281

    > "End-to-end optimized image compression"<br />
    > J. Ballé, V. Laparra, E.P. Simoncelli<br />
    > https://arxiv.org/abs/1611.01704

    This version extends GDN to support slimmable networks with multiple
    width configurations through the idx and mask parameters.

    Arguments:
        inverse: Boolean. If `False` (default), compute GDN response. If `True`,
            compute IGDN response (the division is replaced by multiplication).
        num_deform: The number of subnetworks in the slimmable autoencoder
        rectify: Boolean. If `True`, apply a `relu` nonlinearity to the inputs
            before calculating GDN response.
        gamma_init: The gamma matrix will be initialized as the identity matrix
            multiplied with this value.
        bias_init: Initial value for the bias parameters.
        data_format: Format of input tensor ('channels_first' or 'channels_last').
        beta_parameterizer: Reparameterization for beta parameter.
        gamma_parameterizer: Reparameterization for gamma parameter.
        scale_gamma_parameterizer: Reparameterization for gamma scaling factor.
        bias_gamma_parameterizer: Reparameterization for gamma bias.
        scale_beta_parameterizer: Reparameterization for beta scaling factor.
        bias_beta_parameterizer: Reparameterization for beta bias.
    """

    def __init__(self,
                inverse=False,
                num_deform=5,
                rectify=False,
                gamma_init=.1,
                bias_init=0.00001,
                data_format="channels_last",
                beta_parameterizer=None,
                gamma_parameterizer=None,
                scale_gamma_parameterizer=None,
                bias_gamma_parameterizer=None,
                scale_beta_parameterizer=None,
                bias_beta_parameterizer=None,
                activity_regularizer=None,
                **kwargs):
        # Call parent constructor
        super(GDN, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)
        
        # Get default parameterizers if not provided
        param_defaults = _get_default_parameterizers()
        
        # Store parameters
        self._num_deform = num_deform
        self.inverse = bool(inverse)
        self.rectify = bool(rectify)
        self._gamma_init = float(gamma_init)
        self._bias_init = float(bias_init)
        self.data_format = data_format
        
        # Set parameterizers, using defaults if not provided
        self._beta_parameterizer = beta_parameterizer or param_defaults[0]
        self._gamma_parameterizer = gamma_parameterizer or param_defaults[1]
        self._scale_gamma_parameterizer = scale_gamma_parameterizer or param_defaults[2]
        self._bias_gamma_parameterizer = bias_gamma_parameterizer or param_defaults[3]
        self._scale_beta_parameterizer = scale_beta_parameterizer or param_defaults[4]
        self._bias_beta_parameterizer = bias_beta_parameterizer or param_defaults[5]
        
        # Validate data format early to catch errors
        self._get_channel_axis()
        
        # Set up input spec
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        
        # Initialize member variables
        self.beta = None
        self.gamma = None
        self.beta_scale = None
        self.beta_bias = None
        self.gamma_scale = None
        self.gamma_bias = None

    def _get_channel_axis(self):
        """Get the channel dimension axis based on data_format."""
        try:
            return {"channels_first": 1, "channels_last": -1}[self.data_format]
        except KeyError:
            raise ValueError(f"Unsupported `data_format` for GDN layer: {self.data_format}")

    def build(self, input_shape):
        """Build the layer by creating variables."""
        channel_axis = self._get_channel_axis()
        
        # Convert to TensorShape and get dimensions
        input_shape = tf.TensorShape(input_shape)
        self._input_rank = input_shape.rank
        
        # In the original code, num_channels was hardcoded to 192
        # We could either keep that or derive it from input_shape
        # Let's keep the original behavior for compatibility
        num_channels = 192
        
        # Update input spec with correct dimensions
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=input_shape.rank, axes={channel_axis: num_channels})

        # Create beta parameter (per-channel bias term)
        # Using parameterizer with add_weight instead of add_variable
        if self._beta_parameterizer is None:
            self.beta = self.add_weight(
                name="beta",
                shape=[num_channels],
                initializer=tf.keras.initializers.Ones(),
                trainable=True)
        else:
            self.beta = self._beta_parameterizer(
                name="beta", 
                shape=[num_channels],
                dtype=self.dtype,
                getter=lambda name, *args, **kwargs: self.add_weight(
                    name=name, *args, **kwargs),
                initializer=tf.keras.initializers.Ones())

        # Create gamma parameter (channel interaction terms)
        if self._gamma_parameterizer is None:
            gamma_initializer = tf.keras.initializers.Identity(gain=self._gamma_init)
            self.gamma = self.add_weight(
                name="gamma",
                shape=[num_channels, num_channels],
                initializer=gamma_initializer,
                trainable=True)
        else:
            self.gamma = self._gamma_parameterizer(
                name="gamma", 
                shape=[num_channels, num_channels],
                dtype=self.dtype,
                getter=lambda name, *args, **kwargs: self.add_weight(
                    name=name, *args, **kwargs),
                initializer=tf.keras.initializers.Identity(gain=self._gamma_init))

        # Create beta scale parameter (per-subnetwork scaling for beta)
        if self._scale_beta_parameterizer is None:
            self.beta_scale = self.add_weight(
                name="beta_scale",
                shape=[self._num_deform],
                initializer=tf.keras.initializers.Ones(),
                trainable=True)
        else:
            self.beta_scale = self._scale_beta_parameterizer(
                name="beta_scale", 
                shape=[self._num_deform],
                dtype=self.dtype,
                getter=lambda name, *args, **kwargs: self.add_weight(
                    name=name, *args, **kwargs),
                initializer=tf.keras.initializers.Ones())

        # Create beta bias parameter (per-subnetwork bias for beta)
        if self._bias_beta_parameterizer is None:
            self.beta_bias = self.add_weight(
                name="beta_bias",
                shape=[self._num_deform],
                initializer=tf.keras.initializers.Zeros(),
                trainable=True)
        else:
            self.beta_bias = self._bias_beta_parameterizer(
                name="beta_bias", 
                shape=[self._num_deform],
                dtype=self.dtype,
                getter=lambda name, *args, **kwargs: self.add_weight(
                    name=name, *args, **kwargs),
                initializer=tf.keras.initializers.Zeros())

        # Create gamma scale parameter (per-subnetwork scaling for gamma)
        if self._scale_gamma_parameterizer is None:
            self.gamma_scale = self.add_weight(
                name="gamma_scale",
                shape=[self._num_deform],
                initializer=tf.keras.initializers.Ones(),
                trainable=True)
        else:
            self.gamma_scale = self._scale_gamma_parameterizer(
                name="gamma_scale", 
                shape=[self._num_deform],
                dtype=self.dtype,
                getter=lambda name, *args, **kwargs: self.add_weight(
                    name=name, *args, **kwargs),
                initializer=tf.keras.initializers.Ones())

        # Create gamma bias parameter (per-subnetwork bias for gamma)
        if self._bias_gamma_parameterizer is None:
            self.gamma_bias = self.add_weight(
                name="gamma_bias",
                shape=[self._num_deform],
                initializer=tf.keras.initializers.Zeros(),
                trainable=True)
        else:
            self.gamma_bias = self._bias_gamma_parameterizer(
                name="gamma_bias", 
                shape=[self._num_deform],
                dtype=self.dtype,
                getter=lambda name, *args, **kwargs: self.add_weight(
                    name=name, *args, **kwargs),
                initializer=tf.keras.initializers.Zeros())

        # Mark the layer as built
        self.built = True

    def call(self, inputs, idx, mask, training=None):
        """Apply GDN normalization to the inputs.
        
        Args:
            inputs: Input tensor
            idx: Index of subnetwork to use
            mask: Channel mask size
            training: Whether in training mode (not used, included for Keras compatibility)
            
        Returns:
            Normalized tensor
        """
        # Apply channel masking to inputs
        inputs = inputs[:,:,:,:mask]
        
        # Convert inputs to tensors of the right type
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        ndim = self._input_rank
        
        # Apply rectification if requested
        if self.rectify:
            inputs = tf.nn.relu(inputs)

        # Compute normalization pool based on input rank and data format
        channel_axis = self._get_channel_axis()
        
        if ndim == 2:
            # Special case for 2D inputs (no spatial dimensions)
            norm_pool = tf.matmul(tf.square(inputs), self.gamma)
            norm_pool = tf.nn.bias_add(norm_pool, self.beta)
        elif self.data_format == "channels_last" and ndim <= 5:
            # Efficient implementation for common cases (channels_last, rank 3-5)
            shape = self.gamma.shape.as_list()
            
            # Reshape gamma to match input dimensions
            gamma = tf.reshape(self.gamma, (ndim - 2) * [1] + shape)
            
            # Apply scaling and biasing for the specific subnetwork (idx), with masking
            masked_gamma = self.gamma_scale[idx] * gamma[:,:,:mask,:mask] + self.gamma_bias[idx]
            
            # Compute the normalization pool via convolution
            norm_pool = tf.nn.convolution(
                tf.square(inputs),
                masked_gamma,
                padding="VALID"
            )
            
            # Add beta with appropriate scaling and bias
            norm_pool = tf.nn.bias_add(
                norm_pool, self.beta_scale[idx] * self.beta[:mask] + self.beta_bias[idx])
        else:
            # Generic implementation for other cases
            # Use tensordot to compute the channel-wise normalization
            masked_gamma = self.gamma_scale[idx] * self.gamma[:mask,:mask] + self.gamma_bias[idx]
            
            norm_pool = tf.tensordot(
                tf.square(inputs), masked_gamma, [[channel_axis], [0]])
            
            # Add beta with appropriate scaling and bias
            norm_pool += self.beta_scale[idx] * self.beta[:mask] + self.beta_bias[idx]
            
            # Handle channels_first format
            if self.data_format == "channels_first":
                # Transpose result for channels_first format
                axes = list(range(ndim - 1))
                axes.insert(1, ndim - 1)
                norm_pool = tf.transpose(norm_pool, axes)

        # Apply inverse (IGDN) or regular (GDN) normalization
        if self.inverse:
            # For IGDN, multiply by square root of normalization pool
            norm_pool = tf.sqrt(norm_pool)
        else:
            # For GDN, divide by square root of normalization pool
            norm_pool = tf.math.rsqrt(norm_pool)
            
        # Apply normalization
        outputs = inputs * norm_pool

        return outputs

    def compute_output_shape(self, input_shape):
        """Compute the output shape from the input shape."""
        return tf.TensorShape(input_shape)
        
    def get_config(self):
        """Return layer configuration for serialization."""
        # Start with the base config
        config = super(GDN, self).get_config()
        
        # Add GDN_slim_plus specific configuration
        config.update({
            'inverse': self.inverse,
            'num_deform': self._num_deform,
            'rectify': self.rectify,
            'gamma_init': self._gamma_init,
            'bias_init': self._bias_init,
            'data_format': self.data_format,
            # Parameterizers might need custom serialization - omitted for simplicity
        })
        
        return config