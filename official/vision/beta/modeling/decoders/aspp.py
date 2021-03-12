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
"""Contains definitions of Atrous Spatial Pyramid Pooling (ASPP) decoder."""

# Import libraries
import tensorflow as tf

from official.vision import keras_cv


@tf.keras.utils.register_keras_serializable(package='Vision')
class ASPP(tf.keras.layers.Layer):
  """Creates an Atrous Spatial Pyramid Pooling (ASPP) layer."""

  def __init__(self,
               level,
               dilation_rates,
               stem_type='v3',
               num_filters=256,
               pool_kernel_size=None,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation='relu',
               dropout_rate=0.0,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               interpolation='bilinear',
               **kwargs):
    """Initializes an Atrous Spatial Pyramid Pooling (ASPP) layer.

    Args:
      level: An `int` level to apply ASPP.
      dilation_rates: A `list` of dilation rates.
      num_filters: An `int` number of output filters in ASPP.
      pool_kernel_size: A `list` of [height, width] of pooling kernel size or
        None. Pooling size is with respect to original image size, it will be
        scaled down by 2**level. If None, global average pooling is used.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      activation: A `str` activation to be used in ASPP.
      dropout_rate: A `float` rate for dropout regularization.
      kernel_initializer: A `str` name of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      interpolation: A `str` of interpolation method. It should be one of
        `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`,
        `gaussian`, or `mitchellcubic`.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(ASPP, self).__init__(**kwargs)
    self._config_dict = {
        'level': level,
        'dilation_rates': dilation_rates,
        'stem_type': stem_type,
        'num_filters': num_filters,
        'pool_kernel_size': pool_kernel_size,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'activation': activation,
        'dropout_rate': dropout_rate,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'interpolation': interpolation,
    }

  def build(self, input_shape):
    pool_kernel_size = None
    if self._config_dict['pool_kernel_size']:
      pool_kernel_size = [
          int(p_size // 2**self._config_dict['level'])
          for p_size in self._config_dict['pool_kernel_size']
      ]
    self.aspp = keras_cv.layers.SpatialPyramidPooling(
        output_channels=self._config_dict['num_filters'],
        dilation_rates=self._config_dict['dilation_rates'],
        stem_type=self._config_dict['stem_type'],
        pool_kernel_size=pool_kernel_size,
        use_sync_bn=self._config_dict['use_sync_bn'],
        batchnorm_momentum=self._config_dict['norm_momentum'],
        batchnorm_epsilon=self._config_dict['norm_epsilon'],
        activation=self._config_dict['activation'],
        dropout=self._config_dict['dropout_rate'],
        kernel_initializer=self._config_dict['kernel_initializer'],
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        interpolation=self._config_dict['interpolation'])

  def call(self, inputs):
    """Calls the Atrous Spatial Pyramid Pooling (ASPP) layer on an input.

    The output of ASPP will be a dict of {`level`, `tf.Tensor`} even if only one
    level is present. Hence, this will be compatible with the rest of the
    segmentation model interfaces.

    Args:
      inputs: A `dict` of `tf.Tensor` where
        - key: A `str` of the level of the multilevel feature maps.
        - values: A `tf.Tensor` of shape [batch, height_l, width_l,
          filter_size].

    Returns:
      A `dict` of `tf.Tensor` where
        - key: A `str` of the level of the multilevel feature maps.
        - values: A `tf.Tensor` of output of ASPP module.
    """
    outputs = {}
    level = str(self._config_dict['level'])
    filters = self._config_dict['num_filters']
    if self._config_dict['stem_type'] == 'v2':
      output = self.aspp(inputs[level])
      for i in range(len(self._config_dict['dilation_rates'])):
        outputs['dlv2_{}'.format(i)] = output[:,:,:,filters*i:filters*(i+1)]
    else:
      outputs[level] = self.aspp(inputs[level])
    return outputs

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
