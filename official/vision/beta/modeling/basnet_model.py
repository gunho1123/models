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
"""Build BASNet models."""

# Import libraries
import tensorflow as tf

layers = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package='Vision')
class BASNetModel(tf.keras.Model):
  """A Segmentation class model.

  Input images are passed through backbone first. Decoder network is then
  applied, and finally, segmentation head is applied on the output of the
  decoder network. Layers such as ASPP should be part of decoder.
  """

  def __init__(self,
               backbone,
               decoder,
               refinement,
               #head,
               **kwargs):
    """BASNet initialization function.

    Args:
      backbone: a backbone network.
      decoder: a decoder network. E.g. FPN.
      refinement: a module for salient map refinement
      #head: segmentation head.
      **kwargs: keyword arguments to be passed.
    """
    super(BASNetModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'refinement': refinement,
        #'head': head,
    }
    self.backbone = backbone
    self.decoder = decoder
    self.refinement = refinement

    #self.head = head

  def call(self, inputs, training=None):
    features = self.backbone(inputs)

    if self.decoder:
      features = self.decoder(features)
    
    if self.refinement:
      features['7'] = self.refinement(features['6'])
    

    return features

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone)
    if self.decoder is not None:
      items.update(decoder=self.decoder)
    if self.refinement is not None:
      items.update(refinement=self.refinement)
    return items

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
