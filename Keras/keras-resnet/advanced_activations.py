# -*- coding: utf-8 -*-
"""Layers that act as activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras.legacy import interfaces


class LeakyReLU(Layer):
    """Leaky version of a Rectified Linear Unit.
    It allows a small gradient when the unit is not active:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha: float >= 0. Negative slope coefficient.
    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """

    def __init__(self, alpha=0.3, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.relu(inputs, alpha=self.alpha)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(LeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class PReLU(Layer):
    """Parametric Rectified Linear Unit.
    It follows:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alpha` is a learned array with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_initializer: initializer function for the weights.
        alpha_regularizer: regularizer for the weights.
        alpha_constraint: constraint for the weights.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
    """

    @interfaces.legacy_prelu_support
    def __init__(self, alpha_initializer='zeros',
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True
        self.alpha = self.add_weight(shape=param_shape,
                                     name='alpha',
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, mask=None):
        pos = K.relu(inputs)
        if K.backend() == 'theano':
            neg = (K.pattern_broadcast(self.alpha, self.param_broadcast) *
                   (inputs - K.abs(inputs)) * 0.5)
        else:
            neg = -self.alpha * K.relu(-inputs)
        return pos + neg

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class ELU(Layer):
    """Exponential Linear Unit.
    It follows:
    `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
    `f(x) = x for x >= 0`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha: scale for the negative factor.
    # References
        - [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289v1)
    """

    def __init__(self, alpha=1.0, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.elu(inputs, self.alpha)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(ELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class ThresholdedReLU(Layer):
    """Thresholded Rectified Linear Unit.
    It follows:
    `f(x) = x for x > theta`,
    `f(x) = 0 otherwise`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        theta: float >= 0. Threshold location of activation.
    # References
        - [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/abs/1402.3337)
    """

    def __init__(self, theta=1.0, **kwargs):
        super(ThresholdedReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.theta = K.cast_to_floatx(theta)

    def call(self, inputs, mask=None):
        return inputs * K.cast(K.greater(inputs, self.theta), K.floatx())

    def get_config(self):
        config = {'theta': float(self.theta)}
        base_config = super(ThresholdedReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Softmax(Layer):
    """Softmax activation function.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        axis: Integer, axis along which the softmax normalization is applied.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return activations.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class ReLU(Layer):
    """Rectified Linear Unit activation function.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        max_value: Float, the maximum output value.
    """

    def __init__(self, max_value=None, **kwargs):
        super(ReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.max_value = max_value

    def call(self, inputs):
        return activations.relu(inputs, max_value=self.max_value)

    def get_config(self):
        config = {'max_value': self.max_value}
        base_config = super(ReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class ParametricLog(Layer):
    """Parametric Log Activation Unit.

    It follows:
    `f(x) = log(X + C)`,
    where `C` is a learned array with the same shape as x.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        const_initializer: initializer function for the weights.

    # Usaage
    # References
        - TODO
    """

    def __init__(self, const_initializer='ones', **kwargs):
        super(ParametricLog, self).__init__(**kwargs)
        self.supports_masking = True
        self.const_initializer = initializers.get(const_initializer)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.posConst = self.add_weight(shape=param_shape, name='posConst', initializer=self.const_initializer, regularizer=None, constraint=None)
        self.negConst = self.add_weight(shape=param_shape, name='negConst', initializer=self.const_initializer, regularizer=None, constraint=None)

        self.input_spec = InputSpec(ndim=len(input_shape), axes={})
        self.built = True

    def call(self, inputs):
        #inputs = inputs + K.abs(K.min(inputs)) + 1
        pos = K.log(K.relu(inputs + self.posConst) + 1)
        neg =  - K.log(K.relu( -(inputs - K.abs(inputs)) * 0.5 + self.negConst ) +  1)
        return pos + neg

    def get_config(self):
        config = {'const_initializer': initializers.serialize(self.const_initializer),}
        base_config = super(ParametricLog, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
