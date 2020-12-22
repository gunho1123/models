import tensorflow as tf


from official.vision.beta.modeling import backbones
from official.vision.beta.modeling import basnet_model
from official.vision.beta.modeling import decoders
from official.vision.beta.modeling.modules import refunet


config = {'backbone': {'basnet_en': {'model_id': 'BASNet_En'},
                       'type': 'basnet_en'},
          'decoder': {'basnet_de': {'use_separable_conv': False},
                       'type': 'basnet_de'},
          'input_size': [224, 224, 3],
          'norm_activation': {'activation': 'relu',
                        'norm_epsilon': 0.001,
                        'norm_momentum': 0.99,
                        'use_sync_bn': False},
          'num_classes': 0}
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(config)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

"""
a = type('', (), {})()
a.backbone = type('', (), {})()
a.backbone.basnet_en = type('', (), {})()
a.backbone.basnet_en.model_id = 'BASNet_En'
a.backbone.type = 'basnet_en'

a.decoder = type('', (), {})()
a.decoder.basnet_de = type('', (), {})()
a.decoder.basnet_de.use_separable_conv = False
a.decoder.type = 'basnet_de'
a.input_size = [224, 224, 3]
a.norm_activation = type('', (), {})()
a.norm_activation.activation = 'relu'
a.norm_activation.norm_epsilon = 0.001
a.norm_activation.norm_momentum = 0.99
a.norm_activation.use_sync_bn = False
a.num_classes = 0







print(a)
print(a.backbone.basnet_en.model_id)
"""


tf.keras.backend.set_image_data_format('channels_last')

input_specs = tf.keras.layers.InputSpec(
    shape=[None] + [224, 224, 3])

backbone = backbones.BASNet_En(
    input_specs=input_specs)
decoder = decoders.BASNet_De(
    input_specs=backbone.output_specs)
refinement = refunet.RefUnet()


backbone.model.load_weight('/home/ghpark/ckpt_basnet/ckpt-274352')
decoder.load_weight('/home/ghpark/ckpt_basnet/ckpt-274352')
refinement.load_weight('/home/ghpark/ckpt_basnet/ckpt-274352')


model = basnet_model.BASNetModel(
    backbone=backbone,
    decoder=decoder,
    refinement=refinement
)

#model.load_weight('/home/ghpark/ckpt_basnet/ckpt-274352')


model.summary()





"""Builds basnet model."""
input_specs = tf.keras.layers.InputSpec(
    shape=[None] + [224, 224, 3])

l2_weight_decay = 0
# Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
# (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
# (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
l2_regularizer = (tf.keras.regularizers.l2(
    l2_weight_decay / 2.0) if l2_weight_decay else None)

model = factory.build_basnet_model(
    input_specs=input_specs,
    model_config= a,
    l2_regularizer=l2_regularizer)

print(model)
