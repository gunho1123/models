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


"""

image = np.array(Image.open("./image.jpg"))
image = tf.convert_to_tensor(image)
#image = tf.image.resize(image, [256, 256])
image = tf.image.resize(image, [224, 224])
#image = tf.image.random_crop(image, [224, 224, 3])
image = tf.reshape(image, [1, 224, 224, 3])

label = np.array(Image.open("./mask.png"))
label = tf.convert_to_tensor(label)
label = tf.reshape(label, [300, 400, 1])
label = tf.image.resize(label, [224, 224])
label = tf.reshape(label, [1, 224, 224, 1])

#pil_img = tf.keras.preprocessing.image.array_to_img(image)
#pil_img.show()


image = image/255
label = label/255

#sess = tf.Session()
#value = sess.run(label)
#print(value)

#print(label)

"""

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


#outputs = outputs.numpy()
#np.save("outputs", arr=outputs)
"""
outputs = np.load(file="outputs.npy")
outputs = tf.convert_to_tensor(outputs)
"""
#print(outputs)

loss_fn = basnet_losses.BASNetLoss(
    0.1,
    dataclasses.field(default_factory=list),
    255,
    True)

total_loss = loss_fn(outputs, label)

print("TOTAL_LOSS")
print(total_loss)
