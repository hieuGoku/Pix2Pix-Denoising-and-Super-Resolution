import tensorflow as tf
from tensorflow.keras import layers, models
from models.blocks import *
import cfg

def DiscriminatorDenoise():
    # https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_discriminator

    inp = tf.keras.layers.Input(shape=[cfg.low_height, cfg.low_width, 3])
    tar = tf.keras.layers.Input(shape=[cfg.low_height, cfg.low_width, 3])
    x = tf.keras.layers.concatenate([inp, tar])
    
    x = downsample(filters=64,  size=4, apply_batchnorm=False)(x)
    x = downsample(filters=128, size=4, apply_batchnorm=True)(x)
    x = downsample(filters=256, size=4, apply_batchnorm=True)(x)
    
    initializer1 =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)
    initializer2 =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)
    
    x = layers.Conv2D(filters=256, kernel_size=3, kernel_initializer=initializer1,
                            strides=1, use_bias=False)(x)
    x = layers.Conv2D(filters=1, kernel_size=3, kernel_initializer=initializer2,
                            strides=1, use_bias=False)(x)
    
    model = models.Model(inputs=[inp,tar], outputs=x)
    
    return model
    
def DiscriminatorSR():
    # https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_discriminator

    inp = tf.keras.layers.Input(shape=[cfg.img_height, cfg.img_width, 3])
    
    x = downsample(filters=64,  size=4, apply_batchnorm=False)(inp)
    x = downsample(filters=128, size=4, apply_batchnorm=True)(x)
    x = downsample(filters=256, size=4, apply_batchnorm=True)(x)
    
    initializer1 =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)
    initializer2 =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)
    
    x = layers.Conv2D(filters=256, kernel_size=3, kernel_initializer=initializer1,
                            strides=1, use_bias=False)(x)
    x = layers.Conv2D(filters=1, kernel_size=3, kernel_initializer=initializer2,
                            strides=1, use_bias=False)(x)
    
    model = models.Model(inputs=inp, outputs=x)
    
    return model