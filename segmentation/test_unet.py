import os
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from unet import get_unet
from utils import *

PATCH_SIZE = 128
LOG_NUM = 1

# load unet model
ckpt_dir = 'ckpt_{}'.format(LOG_NUM)
latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

model = get_unet(PATCH_SIZE)
model.load_weights(latest_ckpt)

# set paths for prediction
test_data_dir = '../data/test_data'
prediction_dir = os.path.join(test_data_dir, 'predictions')
os.makedirs(prediction_dir, exist_ok=True)
test_image_files = glob(os.path.join(test_data_dir, 'images/*'))

# save predictions
for image_file in test_image_files:
    file_name = os.path.basename(image_file)
    image = np.array(Image.open(image_file)) / 255 # rescale the image
    old_size = image.shape[0:2]
    image_block_patches = make_block_patches(image, PATCH_SIZE)
    block_coords = list(image_block_patches.keys())
    image_patches = np.array(list(image_block_patches.values()))
    prediction_patches = model(image_patches, training=False)
    prediction_patches = adjust_prediction(prediction_patches)
    prediction_patches = dict(zip(block_coords, prediction_patches))
    prediction_mask = aggregate_block_patches(prediction_patches, old_size)
    save_file = os.path.join(prediction_dir, file_name[:-4] + '_prediction.tif')
    Image.fromarray(np.uint8(prediction_mask * 255), 'L').save(save_file)

# set paths for comparison plot
validation_data_dir = '../data/validation_data'
validation_image_files = glob(os.path.join(validation_data_dir, 'images/*'))
validation_mask_files = glob(os.path.join(validation_data_dir, 'masks/*'))

# save patches that contain vessels
for mask_file in validation_mask_files:
    mask = np.array(Image.open(mask_file))
    if np.max(mask) > 1:
        mask = mask / 255
    mask_block_patches = make_block_patches(mask, PATCH_SIZE)
    mask_patches = np.array(list(mask_block_patches.values()))
    vessel_idx = list(map(vessel_threshold,
                          mask_patches,
                          [0.2 for i in range(len(mask_patches))]))
    sample_idx = np.argsort(vessel_idx)[-30:]
    mask_patches = mask_patches[sample_idx]
    save_patches(mask_patches, 'mask', validation_data_dir)
for image_file in validation_image_files:
    image = np.array(Image.open(image_file)) / 255
    image_block_patches = make_block_patches(image, PATCH_SIZE)
    image_patches = np.array(list(image_block_patches.values()))
    image_patches = image_patches[sample_idx]
    save_patches(image_patches, 'image', validation_data_dir)
    prediction_patches = model(image_patches, training=False)
    prediction_patches = adjust_prediction(prediction_patches)
    save_patches(prediction_patches, 'prediction', validation_data_dir)

# generate comparison plot
comparison_plot(5, validation_data_dir)
