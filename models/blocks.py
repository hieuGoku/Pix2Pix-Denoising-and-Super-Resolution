import tensorflow as tf
from tensorflow.keras import layers, models

class SubPixel(layers.Layer):
    def __init__(self, upscale_factor, channels, activation=None):
        super(SubPixel, self).__init__()
        self.upscale_factor = upscale_factor
        self.channels = channels
        self.activation = activation
    
    def __call__(self, inputs):
        x = layers.Conv2D(self.channels*(self.upscale_factor ** 2), 3, padding="same",
                            activation=self.activation, kernel_initializer="Orthogonal")(inputs)
        outputs = tf.nn.depth_to_space(x, self.upscale_factor)
        return outputs

def extract_first_features(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    initializer =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)
    
    result.add(layers.Conv2D(filters=filters, kernel_size=size, kernel_initializer=initializer,
                        strides=1, padding='same', use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    
    return result
    
def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    initializer =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)
    
    result.add(layers.Conv2D(filters=filters, kernel_size=size, kernel_initializer=initializer,
                            strides=2, padding='same', use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())
    
    return result

def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()

    initializer =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)

    # Conv2DTranspose stride=2
    result.add(layers.Conv2DTranspose(filters=filters, kernel_size=size, kernel_initializer=initializer,
                            strides=2, padding='same', use_bias=False))
    
    result.add(layers.BatchNormalization())
    
    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())
    
    return result

def downsample_sr(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    initializer =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)

    result.add(layers.Conv2D(filters=filters, kernel_size=size, kernel_initializer=initializer,
                            strides=1, padding='same', use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())

    return result

def upsample_sr(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()

    initializer =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)

    # Conv2DTranspose stride=2
    result.add(layers.Conv2DTranspose(filters=filters, kernel_size=size, kernel_initializer=initializer,
                            strides=2, padding='same', use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.LeakyReLU())
    
    return result