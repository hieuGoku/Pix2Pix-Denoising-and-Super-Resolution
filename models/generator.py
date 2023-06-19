import tensorflow as tf
from tensorflow.keras import layers, models
from models.blocks import *
import cfg

def UNet_process(x):
    """
    feature image input -> noise free image

    """
    down_stack = [
        downsample(filters=64, size=4, apply_batchnorm=False),
        downsample(filters=256, size=4, apply_batchnorm=True),
        downsample(filters=512, size=4, apply_batchnorm=True),
        downsample(filters=512, size=4, apply_batchnorm=True),
        downsample(filters=512, size=4, apply_batchnorm=True),
    ]

    up_stack = [
        upsample(filters=512, size=4, apply_dropout=True),
        upsample(filters=512, size=4, apply_dropout=True),
        upsample(filters=256, size=4, apply_dropout=True),
        upsample(filters=64, size=4, apply_dropout=False),
    ]

    initializer =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)
    last = layers.Conv2DTranspose(filters=3, kernel_size=4, kernel_initializer=initializer,
                            strides=2, padding='same', activation='tanh')
    
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)
    
    return x
        
def Generator():
    """
    model denoise and super resolution ((x4 size))

    """
    inputs = layers.Input(shape=(cfg.low_height, cfg.low_width, 3))
    x = extract_first_features(filters=64, size=3, apply_batchnorm=True)(inputs)
    x = UNet_process(x)
    fake_lr = x

    x = extract_first_features(filters=64, size=3, apply_batchnorm=True)(x)
    x = downsample_sr(filters=256, size=3, apply_batchnorm=True)(x)
    x = downsample_sr(filters=256, size=3, apply_batchnorm=True)(x)

    x = upsample_sr(filters=128, size=4, apply_dropout=False)(x)
    
    x = downsample_sr(filters=128, size=3, apply_batchnorm=True)(x)

    initializer =  tf.keras.initializers.RandomNormal(mean=0. , stddev=0.02)
    last = layers.Conv2DTranspose(filters=3, kernel_size=4, kernel_initializer=initializer,
                            strides=2, padding='same', activation='tanh')
    
    fake_hr = last(x)
    
    model = models.Model(inputs=inputs, outputs=[fake_lr, fake_hr])
    
    return model