import os
import tensorflow as tf
from glob import glob
from PIL import Image
from unet import get_unet
from utils import make_patches, get_dataset

PATCH_SIZE = 256
STEP = 64
ALPHA = 0.01
BATCH_SIZE = 64
BUFFER_SIZE = 10**5
LEARNING_RATE = 1e-4
LOG_NUM = 1

# set directories
train_data_dir = '../data/train_data'
validation_data_dir = '../data/validation_data'

train_dataset = get_tf_dataset(train_data_dir, PATCH_SIZE, STEP, BATCH_SIZE,
							   BUFFER_SIZE, ALPHA)
validation_dataset = get_tf_dataset(validation_data_dir, PATCH_SIZE, PATCH_SIZE,
									BATCH_SIZE, )

# get unet model
model = get_unet()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
              loss='binary_crossentropy', metrics=['accuracy'])

# set checkpoint
ckpt_dir = 'ckpt_{}'.format(LOG_NUM)
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_file = os.path.join(ckpt_dir, 'cp.ckpt')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_file,
                                                 save_weights_only=True,
                                                 verbose=1)

# train model
model.fit(train_images,
          train_masks,
          epochs=10,
          validation_data=(test_images, test_masks),
          callbacks=[cp_callback])
