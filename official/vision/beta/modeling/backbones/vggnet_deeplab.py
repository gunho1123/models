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
"""Contains definitions of VGGNet

VGGNet were proposed in:
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolution Networks for Large-Scale Image Recognition. arXiv:1409.1556
"""

# Import libraries
import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.backbones import factory
from official.vision.beta.modeling.layers import nn_blocks
from official.vision.beta.modeling.layers import nn_layers

layers = tf.keras.layers

# Specifications for different ResNet variants.
# Each entry specifies block configurations of the particular ResNet variant.
# Each element in the block configuration is in the following format:
# (num_filters, maxpool_stride, dilation_rate, block_repeats)
VGGNET_SPECS = {
    16: [
        (64,  2,  1,  2),
        (128, 2,  1,  2),
        (256, 2,  1,  3),
        (512, 1,  1,  3),
        (512, 1,  2,  3),
    ],
    19: [
        (64,  2,  1,  2),
        (128, 2,  1,  2),
        (256, 2,  1,  4),
        (512, 1,  1,  4),
        (512, 1,  2,  4),
    ],
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class DilatedVGGNet(tf.keras.Model):
  """Class to build VGGNet model with Deeplabv1 modifications.
  
  This backbone is suitable for semantic segmentation. It was proposed in:
  [2] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille
      Semantic image segmentation with deep convolutional nets and fully connected crfs.
      arXiv:1412.7062 
  """

  def __init__(self,
               model_id,
               input_specs=layers.InputSpec(shape=[None, None, None, 3]),
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """VGGNet with Deeplab modification initialization function.

    Args:
      model_id: `int` depth of VGGNet backbone model.
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
    self._model_id = model_id
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

    # Build VGGNet.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    x = inputs

    endpoints = {}
    #endpoints['0'] = x
    for i, spec in enumerate(VGGNET_SPECS[model_id]):
      x = self._block_group(
          inputs=x,
          filters=spec[0],
          strides=1,
          maxpool_stride=spec[1],
          dilation_rate=spec[2],
          block_repeats=spec[3],
          name='block_group_l{}'.format(i + 1))
      endpoints[str(i)] = x

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(DilatedVGGNet, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def _block_group(self,
                   inputs,
                   filters,
                   strides,
                   maxpool_stride,
                   dilation_rate,
                   block_repeats=1,
                   name='block_group'):
    """Creates one group of blocks for the ResNet model.

    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first convolution of the layer.
      strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
      block_repeats: `int` number of blocks contained in the layer.
      name: `str`name for the block.

    Returns:
      The output `Tensor` of the block layer.
    """
    block_fn = nn_blocks.ConvBlock
    
    x = inputs

    for _ in range(0, block_repeats):
      x = block_fn(
          filters=filters,
          strides=strides,
          dilation_rate=dilation_rate,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._activation,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)(
              x)

    x = layers.MaxPooling2D(pool_size=(3,3), strides=maxpool_stride, padding='same')(x)

    return tf.identity(x, name=name)

  def get_config(self):
    config_dict = {
        'model_id': self._model_id,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('dilated_vggnet')
def build_dilated_vggnet(
    input_specs: tf.keras.layers.InputSpec,
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds VGGNet backbone from a config."""
  backbone_type = model_config.backbone.type
  backbone_cfg = model_config.backbone.get()
  norm_activation_config = model_config.norm_activation
  assert backbone_type == 'dilated_vggnet', (f'Inconsistent backbone type '
                                     f'{backbone_type}')

  return DilatedVGGNet(
      model_id=backbone_cfg.model_id,
      input_specs=input_specs,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)
