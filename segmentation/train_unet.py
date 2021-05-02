import os
import tensorflow as tf
from glob import glob
from PIL import Image
from unet import get_unet
from utils import make_patches, get_tf_dataset

PATCH_SIZE = 64
STEP = 32
ALPHA = 0.01
BUFFER_SIZE = 10**5
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LOG_NUM = 3
EPOCHS = 50

# set directories
train_data_dir = '../data/train_data'
validation_data_dir = '../data/validation_data'

train_dataset = get_tf_dataset(train_data_dir, PATCH_SIZE, STEP, BUFFER_SIZE,
							   BATCH_SIZE, ALPHA)
validation_dataset = get_tf_dataset(validation_data_dir, PATCH_SIZE, PATCH_SIZE,
									BUFFER_SIZE, BATCH_SIZE, threshold=False)

# get unet model
model = get_unet(PATCH_SIZE)
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
