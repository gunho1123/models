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
"""Losses used for BASNet models."""

# Import libraries
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

EPSILON = 1e-5

class BASNetLoss:
  """BASNet loss."""

  def __init__(self, label_smoothing, class_weights,
               ignore_label, use_groundtruth_dimension):
    self._class_weights = class_weights
    self._ignore_label = ignore_label
    self._use_groundtruth_dimension = use_groundtruth_dimension
    self._label_smoothing = label_smoothing
    self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM, from_logits=False)
    self._ssim = tf.image.ssim

  def __call__(self, sigmoids, labels):
    #_, height, width, num_classes = logits.get_shape().as_list()
    """
    if self._use_groundtruth_dimension:
      # TODO(arashwan): Test using align corners to match deeplab alignment.
      logits = tf.image.resize(
          logits, tf.shape(labels)[1:3],
          method=tf.image.ResizeMethod.BILINEAR)
    else:
      labels = tf.image.resize(
          labels, (height, width),
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    """

    print("TEST_label_shape")
    print(labels.get_shape()) # [Batch/GPU, 244, 244]
    
    #print("TEST_sigmoid_shape")
    #print(sigmoids['ref'].get_shape()) # [Batch/GPU, 244, 244, None]
    

    levels = sorted(sigmoids.keys())
    
    labels = tf.cast(labels, tf.float32)
    labels_bce = labels
    labels = tf.expand_dims(labels, axis=-1)


    bce_losses = []
    ssim_losses = []
    iou_losses = []




    for level in levels:
      bce_losses.append(
          #self._binary_crossentropy(sigmoids[level], labels_temp))
          self._binary_crossentropy(labels_bce, sigmoids[level]))
      ssim_losses.append(
          1 - self._ssim(sigmoids[level], labels, max_val=1.0))
      iou_losses.append(
          self._iou_loss(sigmoids[level], labels))
    

    total_bce_loss = tf.math.add_n(bce_losses)
    total_ssim_loss = tf.math.add_n(ssim_losses)
    total_iou_loss = tf.math.add_n(iou_losses)


    
    total_loss = total_bce_loss + total_ssim_loss + total_iou_loss
    #total_loss = total_bce_loss + total_ssim_loss
    #total_loss = total_iou_loss


    return total_loss
    



    """
    bce = tf.keras.losses.BinaryCrossentropy()
    bce_loss = []
    for logit, label in zip(logits, labels):
      bce_loss.append(bce(logit, label).numpy())

    bce_loss = np.array(bce_loss)
    total_bce_loss = np.sum(bce_loss)
    print("total_bce_loss")
    print(total_bce_loss)

    ssim = tf.image.ssim_multiscale(logits, labels, max_val=1.0)
    ssim_loss = 1-ssim
    total_ssim_loss = np.sum(ssim_loss)
    print("total_ssim_loss")
    print(total_ssim_loss)
    
    total_iou_loss = 0.0
    for logit, label in zip(logits, labels):
      
      Iand1 = tf.reduce_sum(logit[:,:,:]*label[:,:,:])
      Ior1 = tf.reduce_sum(logit[:,:,:])+tf.reduce_sum(label[:,:,:])-Iand1
      IoU = Iand1/Ior1

      total_iou_loss += (1-IoU)

    #total_iou_loss = total_iou_loss/logits.get_shape()[0]

    print("total_iou_loss")
    print(total_iou_loss)
    """
    #loss = total_bce_loss + total_ssim_loss + total_iou_loss
    #loss = total_bce_loss
 
    """
    valid_mask = tf.not_equal(labels, self._ignore_label) #_ignore_label = 255, white : object = 0 & background = 1
    normalizer = tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + EPSILON
    # Assign pixel with ignore label to class 0 (background). The loss on the
    # pixel will later be masked out.
    labels = tf.where(valid_mask, labels, tf.zeros_like(labels))

    labels = tf.squeeze(tf.cast(labels, tf.int32), axis=3)
    valid_mask = tf.squeeze(tf.cast(valid_mask, tf.float32), axis=3)
    onehot_labels = tf.one_hot(labels, num_classes)
    onehot_labels = onehot_labels * (
        1 - self._label_smoothing) + self._label_smoothing / num_classes
    #cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
        #labels=onehot_labels, logits=logits)
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=onehot_labels, logits=logits)


    if not self._class_weights:
      class_weights = [1] * num_classes
    else:
      class_weights = self._class_weights

    if num_classes != len(class_weights):
      raise ValueError(
          'Length of class_weights should be {}'.format(num_classes))

    weight_mask = tf.einsum('...y,y->...',
                            tf.one_hot(labels, num_classes, dtype=tf.float32),
                            tf.constant(class_weights, tf.float32))
    valid_mask *= weight_mask
    cross_entropy_loss *= tf.cast(valid_mask, tf.float32)
    loss = tf.reduce_sum(cross_entropy_loss) / normalizer
    """

    #return loss

  """
  def _bce_loss(self, sigmoids, labels, normalizer=1.0):
    with tf.name_scope('bce_loss'):
      bce_loss = self._binary_crossentropy(labels, sigmoids)
      bce_loss /= normalizer
      return bce_loss
  """

  def _iou_loss(self, sigmoids, labels):
    total_iou_loss = 0
    
    Iand1 = tf.reduce_sum(sigmoids[:,:,:,:]*labels[:,:,:,:])
    Ior1 = tf.reduce_sum(sigmoids[:,:,:,:])+tf.reduce_sum(labels[:,:,:,:])-Iand1
    IoU = Iand1/Ior1
    total_iou_loss += 1-IoU

    """
    for sigmoid, label in zip(sigmoids, labels):
      Iand1 = tf.reduce_sum(sigmoid[:,:,:]*label[:,:,:])
      Ior1 = tf.reduce_sum(sigmoid[:,:,:])+tf.reduce_sum(label[:,:,:])-Iand1
      IoU = Iand1/Ior1
      total_iou_loss += 1-IoU
    """
    return total_iou_loss




