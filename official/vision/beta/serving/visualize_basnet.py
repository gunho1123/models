import tensorflow as tf
import numpy as np
from PIL import Image



export_dir_path = "/home/ghpark/export_basnet/saved_model"

input_images = np.array(Image.open('img7_test.jpg'))
input_images = tf.image.resize(input_images, [256, 256])
input_images = tf.reshape(input_images, [1, 256, 256, 3])
processed_images = tf.cast(input_images, tf.uint8)


print("processed_images.shape")
print(processed_images)
print(processed_images.shape)






imported = tf.saved_model.load(export_dir_path)



model_fn = imported.signatures['serving_default']
output = model_fn(processed_images)


output = output['outputs']
output = output*255
print(output)
print("Reduce_MAX")
print(tf.math.reduce_max(output))



output = tf.cast(output, tf.uint8)

output = output[0,:,:,0].numpy()

print(output)
print(output.shape)


img = np.zeros((256,256,3))
img[:,:,0] = output
img[:,:,1] = output
img[:,:,2] = output



tf.keras.preprocessing.image.save_img("./output.png", img, data_format="channels_last", scale=False)
#outputs = imported(processed_images, training=False)
