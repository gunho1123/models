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
"""Tests for aspp."""

# Import libraries
import tensorflow as tf

from official.vision.beta.modeling.backbones import basnet_en
from official.vision.beta.modeling.decoders import basnet_de
from official.vision.beta.modeling.modules import refunet
from official.vision.beta.losses import basnet_losses

from PIL import Image
import numpy as np
import dataclasses


layers = tf.keras.layers



def random_crop_and_pad_image_and_mask(image, mask, size):
  """Randomly crops `image` together with `mask`.

  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    mask: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [H, W] indicating the crop size.
  Returns:
    A tuple of (cropped_image, cropped_mask).
  """
  combined = tf.concat([image, mask], axis=2)
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined, 0, 0,
      tf.maximum(size[0], image_shape[0]),
      tf.maximum(size[1], image_shape[1]))
  last_mask_dim = tf.shape(mask)[-1]
  last_image_dim = tf.shape(image)[-1]
  combined_crop = tf.image.random_crop(
      combined_pad,
      size=tf.concat([size, [last_mask_dim + last_image_dim]],
                     axis=0))
  return (combined_crop[:, :, :last_image_dim],
          combined_crop[:, :, last_image_dim:])







image = np.array(Image.open("./image.jpg"))
image = tf.convert_to_tensor(image)
#print("image")
#print(image)
#print(image.shape)

image = tf.image.resize(image, [256, 256])
#image = tf.image.random_crop(image, [224, 224, 3])
#image = tf.reshape(image, [1, 224, 224, 3])

label = np.array(Image.open("./mask.png"))
label = tf.convert_to_tensor(label)


label = tf.reshape(label, [label.shape[0], label.shape[1], 1])
label = tf.image.resize(label, [256, 256])


image, label = random_crop_and_pad_image_and_mask(image, label, [224, 224])
print("CROPPPPPP")
print(image)
print(label)

pil_img1 = tf.keras.preprocessing.image.array_to_img(image)
pil_img2 = tf.keras.preprocessing.image.array_to_img(label)

#Image.fromarray(np.hstack((np.array(pil_img1),np.array(pil_img2)))).show()
pil_img1.show()
pil_img2.show()




image = image/255
label = label/255

#sess = tf.Session()
#value = sess.run(label)
#print(value)

#print(label)


input_size = 224
tf.keras.backend.set_image_data_format('channels_last')

inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)

image_chw = np.load(file="image_save.npy")
label_chw = np.load(file="label_save.npy")

#print(image_chw.shape) # (3, 224, 224)
#print(label_chw.shape) # (1, 224, 224)

image_hwc = np.zeros((224, 224, 3))

image_hwc[:,:,0] = image_chw[0,:,:]
image_hwc[:,:,1] = image_chw[1,:,:]
image_hwc[:,:,2] = image_chw[2,:,:]

label_hwc = np.zeros((224, 224, 1))
label_hwc[:,:,0] = label_chw[0,:,:]


#print(image_hwc.shape) #
image_bhwc = np.expand_dims(image_hwc, axis=0)
image = tf.convert_to_tensor(image_bhwc)
#print("IMAGE")
#print(image.get_shape())
#image = tf.reshape(image, [1, 244, 244, 3])
#image = image.reshape(1, 224, 224, 3)
label_bhwc = np.expand_dims(label_hwc, axis=0)
label = tf.convert_to_tensor(label_bhwc)
#label = tf.reshape(label, [1, 244, 244, 1])
#print("LABEL")
#print(label.get_shape())
#


backbone = basnet_en.BASNet_En()

network = basnet_de.BASNet_De(
      input_specs=backbone.output_specs  
  )

module = refunet.RefUnet()

endpoints = backbone(image)

#print(endpoints.get_shape())
#np.save("inconv_tf", arr=endpoints.numpy())
#print("SAVE DONE")
sups = network(endpoints)


sups['7'] = module(sups['6'])

outputs = []
for i in range(8):
  outputs.append(sups[str(i)])

outputs = np.array(outputs)

outputs = tf.squeeze(outputs, axis=1)


loss_fn = basnet_losses.BASNetLoss(
    0.1,
    dataclasses.field(default_factory=list),
    255,
    True)

total_loss = loss_fn(outputs, label)

print("TOTAL_LOSS")
print(total_loss)
