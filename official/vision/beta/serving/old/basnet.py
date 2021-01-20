# Lint as: python3
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
"""Detection input and model functions for serving/inference."""

import tensorflow as tf

from official.vision.beta.modeling import factory
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.serving import export_base
from official.core import exp_factory
import numpy as np
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class BASNetModule(export_base.ExportModule):
  """basnet Module."""

  def build_model(self):
    input_specs = tf.keras.layers.InputSpec(
        shape=[self._batch_size] + self._input_image_size + [3])

    self._model = factory.build_basnet_model(
        input_specs=input_specs,
        model_config=self._params.task.model,
        l2_regularizer=None)

    return self._model

  def _build_inputs(self, image):
    """Builds classification model inputs for serving."""
    # Center crops and resizes image.
    image = preprocess_ops.center_crop_image(image)

    image = tf.image.resize(
        image, self._input_image_size, method=tf.image.ResizeMethod.BILINEAR)

    image = tf.reshape(
        image, [self._input_image_size[0], self._input_image_size[1], 3])

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)
    return image

  def _run_inference_on_image_tensors(self, images):
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      Tensor holding classification output logits.
    """
    with tf.device('cpu:0'):
      images = tf.cast(images, dtype=tf.float32)

      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._build_inputs, elems=images,
              fn_output_signature=tf.TensorSpec(
                  shape=self._input_image_size + [3], dtype=tf.float32),
              parallel_iterations=32
              )
          )

    logits = self._model(images, training=False)

    return dict(outputs=logits)


params = exp_factory.get_exp_config('basnet_duts')

image = tf.keras.preprocessing.image.load_img('./img1_test.jpg')
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])

#print(input_arr.shape)



#input_arr = tf.reshape(input_arr, [1, input_arr.shape[0], input_arr.shape[1], 3])

module = BASNetModule(
    params, batch_size=1, input_image_size=[224, 224])
model = module.build_model()

ckpt = tf.train.Checkpoint(**model.checkpoint_items)
status = ckpt.restore("/home/ghpark/ckpt_basnet/ckpt-68588")
status.assert_consumed()

"""
processed_images = tf.nest.map_structure(
    tf.stop_gradient,
    tf.map_fn(
        module._build_inputs,
        elems=input_arr,
        fn_output_signature=tf.TensorSpec(
            shape=[224, 224, 3], dtype=tf.float32)
    )
)
"""


processed_images = tf.image.resize(input_arr, [224, 224])

print("processed_image")
print(processed_images.shape)
print(processed_images)

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
processed_images = processed_images/255
print(processed_images)

outputs = model(processed_images, training=False)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(outputs)

output = outputs['ref']
output = output*255
output = tf.cast(output, tf.uint8)
print(output)


output = output[0,:,:,0]

print(output.shape)


img = np.zeros((224,224,3))
img[:,:,0] = output
img[:,:,1] = output
img[:,:,2] = output



tf.keras.preprocessing.image.save_img("./output.png", img, data_format="channels_last", scale=True)
