import os
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from unet import get_unet
from utils import make_block_patches, aggregate_block_patches

PATCH_SIZE = 64
LOG_NUM = 3

# load unet model
ckpt_dir = 'ckpt_{}'.format(LOG_NUM)
latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

model = get_unet(PATCH_SIZE)
model.load_weights(latest_ckpt)

# set paths for data
test_data_dir = '../data/test_data'
image_files = glob(os.path.join(test_data_dir, 'images/*'))
mask_files = glob(os.path.join(test_data_dir, 'masks/*'))

#
