import tensorflow as tf
from sklearn.cross_validation import train_test_split
import glob
import cv2
import functools
import glob
import numpy as np
import time

T1 = time.time()
tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS  
tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_string("gpu_id", "0", "idx of GPU using")

import os  # see issue #152
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

from tensorflow.python.client import device_lib
# print (device_lib.list_local_devices())

# experiment code
def _img_string_to_tensor(image_string, is_aug, image_size):
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Convert from full range of uint8 to range [0,1] of float32.
    image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    # Resize to expected
    if is_aug:
        image = tf.image.resize_images(image_decoded_as_float, size=(256,256))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_contrast(image, 0.4, 0.6)
        image = tf.image.random_brightness(image, 0.2)
        
    else:
        image = tf.image.resize_images(image_decoded_as_float, size=image_size)
        return image
    return image

# def our input function
def make_input_fn(filenames, labels, is_aug, image_size=(256,256), shuffle=False, batch_size=64, num_epochs=None, buffer_size=4096):
    
    def _path_to_img(path, label, is_aug=is_aug):
        # Read in the image from disk
        image_string = tf.read_file(path)
        image_resized = _img_string_to_tensor(image_string, image_size, is_aug)
        
        return {'input_1': image_resized}, label
    
    def _input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.prefetch(buffer_size=batch_size)    
        if shuffle:
            # Shuffle the records. Note that we shuffle before repeating to ensure
            # that the shuffling respects epoch boundaries.
            dataset = dataset.shuffle(buffer_size=buffer_size)

          # If we are training over multiple epochs before evaluating, repeat the
          # dataset for the appropriate number of epochs.
            dataset = dataset.repeat(num_epochs)

          # Parse the raw records into images and labels. Testing has shown that setting
          # num_parallel_batches > 1 produces no improvement in throughput, since
          # batch_size is almost always much greater than the number of CPU cores.
            dataset = dataset.apply(
              tf.contrib.data.map_and_batch(_path_to_img,
                  batch_size=batch_size,
                  num_parallel_batches=1))

          # Operations between the final prefetch and the get_next call to the iterator
          # will happen synchronously during run time. We prefetch here again to
          # background all of the above processing work and keep it out of the
          # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
          # allows DistributionStrategies to adjust how many batches to fetch based
          # on how many devices are present.
        dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        return dataset

    return _input_fn

# def our model by tf.keras
import keras.backend as K 
tf.keras.backend.clear_session()
keras_resnet = tf.keras.applications.ResNet50(
    input_shape=(256, 256, 3),
    include_top=False,
    pooling='avg',
    weights='imagenet')

print("CREATING MODEL.....")
logits =  tf.keras.layers.Dense(2, 'softmax')(keras_resnet.layers[-1].output)
model = tf.keras.models.Model(keras_resnet.inputs, logits)

# Compile model with the optimizer, loss, and metrics you'd like to train with.
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                          loss='categorical_crossentropy',
                          metric='accuracy')

# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_resnet = tf.keras.estimator.model_to_estimator(keras_model=model)

data_dir = '/data/jimmy15923/dogscats/'
x_train = glob.glob(os.path.join(data_dir, 'train', '**/*.jpg'))
y_train = np.zeros(23000)
y_train[:11500] = 1
y_train = tf.keras.utils.to_categorical(y_train, 2)

# build input function
train_input_fn = make_input_fn(x_train, y_train, batch_size=64, shuffle=True, is_aug=True)

print("START TRANING.....")
# time_cal._run()
est_resnet.train(input_fn=train_input_fn, steps=1000)
# time_cal.stop()
print("DONE TRAINING!!")
T2 = time.time()

res = (T2 - T1) / 60.0

print("-"*10)
print("EVALUATING...")
x_val = glob.glob(os.path.join(data_dir, 'valid', '**/*.jpg'))
y_val = np.zeros(2000)
y_val[:1000] = 1
y_val = tf.keras.utils.to_categorical(y_val, 2)
val_input_fn = make_input_fn(x_val, y_val, batch_size=64, shuffle=True, is_aug=False)

est_resnet.evaluate(val_input_fn, steps=40)

print("TOTAL time = {} mins".format(str(res)[:4]))

# time_cal.save2plot("usage.png")
