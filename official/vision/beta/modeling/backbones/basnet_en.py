# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Encoder of BASNet.

Boundary-Awar network (BASNet) were proposed in:
[1] Qin, Xuebin, et al. 
    Basnet: Boundary-aware salient object detection.
"""


# Import libraries
import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_blocks
from official.vision.beta.modeling.layers import nn_layers

layers = tf.keras.layers

# Specifications for BASNet encoder.
# Each element in the block configuration is in the following format:
# (block_fn, num_filters, stride, block_repeats)

BASNET_EN_SPECS = [
        ('residual', 64,  1, 3),    #ResNet-34
        ('residual', 128, 2, 4),   #ResNet-34
        ('residual', 256, 2, 6),   #ResNet-34
        ('residual', 512, 2, 3),   #ResNet-34
        ('residual', 512, 1, 3),   #BASNet
        ('residual', 512, 1, 3),   #BASNet
    ]

@tf.keras.utils.register_keras_serializable(package='Vision')
class BASNet_En(tf.keras.Model):

  def __init__(self,
               input_specs=layers.InputSpec(shape=[None, None, None, 3]),
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """BASNet_En initialization function.

    Args:
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                          Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                        Default to None.
      **kwargs: keyword arguments to be passed.
    """
    self._input_specs = input_specs
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Build BASNet.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    x = layers.Conv2D(
        # (gunho) for BASNet
        filters=64, kernel_size=3, strides=1, use_bias=False, padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            inputs)
    x = self._norm(
        axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
            x)
    x = tf_utils.get_activation(activation)(x)
    # (gunho) for BASNet
    #x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)


    endpoints = {}

    for i, spec in enumerate(BASNET_EN_SPECS):
      if spec[0] == 'residual':
        block_fn = nn_blocks.ResidualBlock
      elif spec[0] == 'bottleneck':
        block_fn = nn_blocks.BottleneckBlock
      else:
        raise ValueError('Block fn `{}` is not supported.'.format(spec[0]))
      x = self._block_group(
          inputs=x,
          filters=spec[1],
          strides=spec[2],
          block_fn=block_fn,
          block_repeats=spec[3],
          name='block_group_l{}'.format(i + 2))
      if i == 4 or i == 5:
        x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
      endpoints[str(i)] = x

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(BASNet_En, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def _block_group(self,
                   inputs,
                   filters,
                   strides,
                   block_fn,
                   block_repeats=1,
                   name='block_group'):
    """Creates one group of blocks for the ResNet model.

    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first convolution of the layer.
      strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
      block_fn: Either `nn_blocks.ResidualBlock` or `nn_blocks.BottleneckBlock`.
      block_repeats: `int` number of blocks contained in the layer.
      name: `str`name for the block.

    Returns:
      The output `Tensor` of the block layer.
    """
    x = block_fn(
        filters=filters,
        strides=strides,
        use_projection=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)(
            inputs)

    for _ in range(1, block_repeats):
      x = block_fn(
          filters=filters,
          strides=1,
          use_projection=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._activation,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)(
              x)

    return tf.identity(x, name=name)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs
