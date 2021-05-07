import os
import tensorflow as tf
from glob import glob
from PIL import Image
from unet import get_unet
from utils import make_train_patches, get_train_dataset

PATCH_SIZE = 128
DROPOUT_RATE = 0.2
STEP = 16
ALPHA = 0.001
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LOG_NUM = 1
EPOCHS = 10

# set paths
train_data_dir = '../data/train_data'
validation_data_dir = '../data/validation_data'

train_dataset = get_train_dataset(train_data_dir, PATCH_SIZE, STEP, BATCH_SIZE,
                                  ALPHA)
validation_dataset = get_train_dataset(validation_data_dir, PATCH_SIZE,
                                       PATCH_SIZE, BATCH_SIZE, ALPHA)

# get unet model
model = get_unet(PATCH_SIZE, DROPOUT_RATE)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
              loss='binary_crossentropy', metrics=['accuracy'])

# set checkpoint
ckpt_dir = 'ckpt_{}'.format(LOG_NUM)
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_file = os.path.join(ckpt_dir, 'cp.ckpt')

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_file,
                                                   save_weights_only=True,
                                                   verbose=1)

# train model
model.fit(train_dataset,
          epochs=EPOCHS,
          validation_data=validation_dataset,
          callbacks=[ckpt_callback])
