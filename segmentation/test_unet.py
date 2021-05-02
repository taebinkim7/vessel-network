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

for image_file, mask_file in zip(image_files, mask_files):
	image = np.array(Image.open(image_file))/255 # rescale the image
	mask = np.array(Image.open(mask_file))
    old_size = image.shape[0:2]
    image_block_patches = make_block_patches(image, PATCH_SIZE)
    mask_block_patches = make_block_patches(mask, PATCH_SIZE)
    mask_block_coords = list(mask_block_patches.keys())
    test_patches = np.array(list(image_block_patches.values()))
    predicted_patches = model(test_patches, training=False)
    predicted_patches = adjust_prediction(predicted_patches)
    predicted_patches = dict(zip(mask_block_coords, predicted_patches))
    predicted_mask = aggregate_block_patches(predicted_patches, old_size)
    save_file = image_file[:-4] + '_prediction.png'
    Image.fromarray(predicted_mask).save(save_file)
