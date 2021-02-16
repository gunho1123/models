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
"""FOV decoder."""

# Import libraries
import tensorflow as tf

from official.modeling import tf_utils
from official.vision import keras_cv

layers = tf.keras.layers

@tf.keras.utils.register_keras_serializable(package='Vision')
class FOV(tf.keras.layers.Layer):
  """FOV."""

  def __init__(self,
               level,
               dilation_rates,
               kernel_size,
               num_filters=1024,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation='relu',
               dropout_rate=0.5,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               **kwargs):
    """ASPP initialization function.

    Args:
      level: `int` level to apply ASPP.
      dilation_rates: `list` of dilation rates.
      num_filters: `int` number of output filters in ASPP.
      pool_kernel_size: `list` of [height, width] of pooling kernel size or
        None. Pooling size is with respect to original image size, it will be
        scaled down by 2**level. If None, global average pooling is used.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      activation: `str` activation to be used in ASPP.
      dropout_rate: `float` rate for dropout regularization.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      interpolation: interpolation method, one of bilinear, nearest, bicubic,
        area, lanczos3, lanczos5, gaussian, or mitchellcubic.
      **kwargs: keyword arguments to be passed.
    """
    super(FOV, self).__init__(**kwargs)
    self._config_dict = {
        'level': level,
        'dilation_rates': dilation_rates,
        'kernel_size': kernel_size,
        'num_filters': num_filters,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'activation': activation,
        'dropout_rate': dropout_rate,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
    }
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1
    
    self._activation_fn = tf_utils.get_activation(activation)
    self._norm = self._norm(
        axis= bn_axis,
        momentum=norm_momentum,
        epsilon=norm_epsilon)
    self._dropout = layers.Dropout(dropout_rate)


  def build(self, input_shape):
    self._conv0 = layers.Conv2D(
        filters=self._config_dict['num_filters'],
        kernel_size=self._config_dict['kernel_size'],
        dilation_rate=self._config_dict['dilation_rates'],
        strides=1, use_bias=False, padding='same',
        kernel_initializer=self._config_dict['kernel_initializer'],
        kernel_regularizer=self._config_dict['kernel_regularizer']
        )
    self._conv1 = layers.Conv2D(
        filters=self._config_dict['num_filters'],
        kernel_size=1,
        dilation_rate=1,
        strides=1, use_bias=False, padding='same',
        kernel_initializer=self._config_dict['kernel_initializer'],
        kernel_regularizer=self._config_dict['kernel_regularizer']
        )



  def call(self, inputs):
    """FOV call method.

    The output of ASPP will be a dict of level, Tensor even if only one
    level is present. Hence, this will be compatible with the rest of the
    segmentation model interfaces..

    Args:
      inputs: A dict of tensors
        - key: `str`, the level of the multilevel feature maps.
        - values: `Tensor`, [batch, height_l, width_l, filter_size].
    Returns:
      A dict of tensors
        - key: `str`, the level of the multilevel feature maps.
        - values: `Tensor`, output of ASPP module.
    """
    outputs = {}
    
    level = str(self._config_dict['level'])
    x = self._conv0(inputs[level])
    x = self._norm(x)
    x = self._activation_fn(x)
    x = self._dropout(x)

    x = self._conv1(x)
    x = self._norm(x)
    x = self._activation_fn(x)
    x = self._dropout(x)

    outputs[level] = x
    return outputs

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
