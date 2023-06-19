import tensorflow as tf
from tensorflow.keras import layers, models
from models.blocks import *

def discriminator_loss(disc_real_output, disc_generated_output):

    loss_dsic_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.03)

    real_los = loss_dsic_object(tf.ones_like(disc_real_output), disc_real_output)
    
    fake_los = loss_dsic_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_loss = real_los + fake_los
    
    return total_loss

def generator_denoise_loss(disc_generated_output, gen_output, target):

    bce  = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    LAMBDA = 40

    gan_loss = bce(tf.ones_like(disc_generated_output), disc_generated_output)
    
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_loss = gan_loss + LAMBDA*l1_loss

    return total_loss, gan_loss, l1_loss

def generator_sr_loss(disc_generated_output, gen_output, target):

    bce  = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    LAMBDA = 40

    gan_loss = bce(tf.ones_like(disc_generated_output), disc_generated_output)
    
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    total_loss = gan_loss + LAMBDA*l1_loss
    
    return total_loss, gan_loss, l1_loss
