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
"""Metrics for segmentation."""

import tensorflow as tf
import numpy as np


class BASNetEvaluator(tf.keras.metrics.Metric):
  """Mean Absolute Error metric for BASNet.

  This class utilizes tf.keras.metrics.MeanAbsoluteError to perform batched mean absolute error when
  both input images and groundtruth masks are resized to the same size
  (rescale_predictions=False). It doesn't support computing mean absolute error on groundtruth original
  sizes, in which case, each prediction is rescaled back to the original image
  size.
  """

  def __init__(
      self, name=None, dtype=None):
    """Constructs BASNet evaluator class.

    Args:
      name: `str`, name of the metric instance..
      dtype: data type of the metric result.
    """
    self._thresholds=0.9
    self._mae = tf.keras.metrics.MeanAbsoluteError(dtype=dtype)
    self._precision = tf.keras.metrics.Precision(thresholds=self._thresholds)
    self._recall = tf.keras.metrics.Recall(thresholds=self._thresholds)
    self._beta_square = 0.3

    super(BASNetEvaluator, self).__init__(
        name=name, dtype=dtype)

  def update_state(self, y_true, y_pred):
    """Updates metic state.

    Args:
      y_true: Tensor [batch, width, height, 1], groundtruth masks.
      y_pred: Tensor [batch, width_p, height_p, num_classes], predicated masks.
    """
    self._precision.update_state(y_true, y_pred)
    self._recall.update_state(y_true, y_pred)
    self._mae.update_state(y_true, y_pred)



  def result(self):
    precision = self._precision.result().numpy()
    recall = self._recall.result().numpy()
    self.mae = self._mae.result().numpy()


    self.f_beta = (1+self._beta_square)*precision*recall / (self._beta_square*precision+recall)

    metrics_dict = {}
    metrics_dict['mae'] = self.mae
    metrics_dict['F_beta'] = self.f_beta

    temp_array = [self.mae, self.f_beta]
    temp_array = np.array(temp_array)

    #return metrics_dict
    return self.mae, self.f_beta
    #return temp_array

  def reset_states(self):
    self._precision.reset_states()
    self._recall.reset_states()
    self._mae.reset_states()





class MeanAbsoluteError(tf.keras.metrics.MeanAbsoluteError):
  """Mean Absolute Error metric for BASNet.

  This class utilizes tf.keras.metrics.MeanAbsoluteError to perform batched mean absolute error when
  both input images and groundtruth masks are resized to the same size
  (rescale_predictions=False). It doesn't support computing mean absolute error on groundtruth original
  sizes, in which case, each prediction is rescaled back to the original image
  size.
  """

  def __init__(
      self, rescale_predictions=False, name=None, dtype=None):
    """Constructs Segmentation evaluator class.

    Args:
      rescale_predictions: `bool`, whether to scale back prediction to original
        image sizes. If True, y_true['image_info'] is used to rescale
        predictions.
      name: `str`, name of the metric instance..
      dtype: data type of the metric result.
    """
    self._rescale_predictions = rescale_predictions
    super(MeanAbsoluteError, self).__init__(
        name=name, dtype=dtype)

  def update_state(self, y_true, y_pred):
    """Updates metic state.

    Args:
      y_true: Tensor [batch, width, height, 1], groundtruth masks.
      y_pred: Tensor [batch, width_p, height_p, num_classes], predicated masks.
    """
    predictions = y_pred
    masks = y_true

    """
    if isinstance(predictions, tuple) or isinstance(predictions, list):
      predictions = tf.concat(predictions, axis=0)
      masks = tf.concat(masks, axis=0)
    """
    # Ignore mask elements is set to zero for argmax op.
    
    """
    if self._rescale_predictions:
      # This part can only run on cpu/gpu due to dynamic image resizing.
      flatten_predictions = []
      flatten_masks = []
      flatten_valid_masks = []
      for mask, valid_mask, predicted_mask, image_info in zip(
          masks, valid_masks, predictions, images_info):

        rescale_size = tf.cast(
            tf.math.ceil(image_info[1, :] / image_info[2, :]), tf.int32)
        image_shape = tf.cast(image_info[0, :], tf.int32)
        offsets = tf.cast(image_info[3, :], tf.int32)

        predicted_mask = tf.image.resize(
            predicted_mask,
            rescale_size,
            method=tf.image.ResizeMethod.BILINEAR)

        predicted_mask = tf.image.crop_to_bounding_box(predicted_mask,
                                                       offsets[0], offsets[1],
                                                       image_shape[0],
                                                       image_shape[1])
        mask = tf.image.crop_to_bounding_box(mask, 0, 0, image_shape[0],
                                             image_shape[1])
        valid_mask = tf.image.crop_to_bounding_box(valid_mask, 0, 0,
                                                   image_shape[0],
                                                   image_shape[1])

        predicted_mask = tf.argmax(predicted_mask, axis=2)
        flatten_predictions.append(tf.reshape(predicted_mask, shape=[1, -1]))
        flatten_masks.append(tf.reshape(mask, shape=[1, -1]))
        flatten_valid_masks.append(tf.reshape(valid_mask, shape=[1, -1]))
      flatten_predictions = tf.concat(flatten_predictions, axis=1)
      flatten_masks = tf.concat(flatten_masks, axis=1)
      flatten_valid_masks = tf.concat(flatten_valid_masks, axis=1)

    else:
      predictions = tf.image.resize(
          predictions,
          tf.shape(masks)[1:3],
          method=tf.image.ResizeMethod.BILINEAR)
      predictions = tf.argmax(predictions, axis=3)
      flatten_predictions = tf.reshape(predictions, shape=[-1])
      flatten_masks = tf.reshape(masks, shape=[-1])
      flatten_valid_masks = tf.reshape(valid_masks, shape=[-1])
    predictions = tf.image.resize(
        predictions,
        tf.shape(masks)[1:3],
        method=tf.image.ResizeMethod.BILINEAR)
    predictions = tf.argmax(predictions, axis=3)
    flatten_predictions = tf.reshape(predictions, shape=[-1])
    flatten_masks = tf.reshape(masks, shape=[-1])
    flatten_valid_masks = tf.reshape(valid_masks, shape=[-1])
 
    """
    super(MeanAbsoluteError, self).update_state(
        masks, predictions)

