import tensorflow as tf
import matplotlib.pyplot as plt
import cfg

def load(image_path):
    """
    Parameters
    ----------
    image_path : string
    
    Returns
    -------
    image_lr: tf.Tensor (tf.float32) input low resolution and noise
    target_lr: tf.Tensor (tf.float32) target low resolution and not noise 
    target_hr: tf.Tensor (tf.float32) target high resolution 
    """
    
    # read file and decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image)

    # resize image_lr, target_lr with low_width and low_height
    image_lr = tf.image.resize(image, size=(cfg.low_height, cfg.low_width))
    target_lr = tf.image.resize(image, size=(cfg.low_height, cfg.low_width))
    
    # resize target_hr IMG_WIDTH and IMG_HEIGHT
    target_hr = tf.image.resize(image, size=(cfg.img_height, cfg.img_width))
    
    # create noise matrix has size with low_width and low_height
    noise_per = 0.20
    noise = tf.random.uniform(shape=(cfg.low_height, cfg.low_width, 3), minval=1-noise_per, maxval=1+noise_per)
    
    # Convert image to float32 tensors
    image_lr = tf.cast(image_lr, dtype=tf.float32)
    target_lr = tf.cast(target_lr, dtype=tf.float32)
    target_hr = tf.cast(target_hr, dtype=tf.float32)
    
    # image_lr multiply noise matrix -> noise image
    image_lr = image_lr*noise
    image_lr = tf.clip_by_value(image_lr, 0, 255)
    
    return image_lr, target_lr, target_hr

def normalize(image_lr, target_lr, target_hr):
    """
    normalizing the images to [-1, 1]    

    """
    image_lr  = (image_lr/127.5) - 1  
    target_lr = (target_lr/127.5) - 1 
    target_hr = (target_hr/127.5) - 1 
    
    return image_lr, target_lr, target_hr

#@tf.function()
def random_jitter(image_lr, target_lr, target_hr):
    """
    There is a 50% chance of flipping the image from left to right
    
    """
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        image_lr  = tf.image.flip_left_right(image_lr)
        target_lr  = tf.image.flip_left_right(target_lr)
        target_hr  = tf.image.flip_left_right(target_hr)
        
    return image_lr, target_lr, target_hr
    
def load_image_train(image_path):
    """ 
    load image, random_jitter and normalize image 

    """
    image_lr, target_lr, target_hr = load(image_path)

    image_lr, target_lr, target_hr = random_jitter(image_lr, target_lr, target_hr)

    image_lr, target_lr, target_hr = normalize(image_lr, target_lr, target_hr)

    return image_lr, target_lr, target_hr

def load_image_test(image_path):
    """ 
    load image and normalize image 

    """

    image_lr, target_lr, target_hr = load(image_path)

    image_lr, target_lr, target_hr = normalize(image_lr, target_lr, target_hr)

    return image_lr, target_lr, target_hr


def evaluate(model, epoch, data):        
    psnr_mean = 0.0
    for image_lr, target_lr, target_hr in data:
        fake_lr, fake_hr = model(image_lr, training=False)
        psnr_b = tf.image.psnr(fake_hr, target_hr, max_val=1.0)
        psnr_mean = tf.math.reduce_mean(psnr_b)
    
#     psnr_mean = psnr_mean/10.0
    print('-------- psnr: ', psnr_mean.numpy(), '   ----- epoch: ', epoch)
    return psnr_mean
    

def generate_images(model, image_lr, target_sr, is_fake_lr=False):
    fake_lr, fake_hr = model([image_lr], training=False)
    if is_fake_lr:
        display_list = [image_lr[0], fake_lr[0], target_sr[0], fake_hr[0]]
        title_l = ['Input', 'Denoise', 'Real', 'Generated']    
        plt.figure(figsize=(25,30))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.title(title_l[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
    else:
        display_list = [image_lr[0], target_sr[0], fake_hr[0]]
        title_l = ['Input', 'Real', 'Generated']    
        plt.figure(figsize=(25,30))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title_l[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.show()