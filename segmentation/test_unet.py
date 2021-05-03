import os
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from unet import get_unet
from utils import make_block_patches, aggregate_block_patches, adjust_prediction

PATCH_SIZE = 128
LOG_NUM = 1

# load unet model
ckpt_dir = 'ckpt_{}'.format(LOG_NUM)
latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

model = get_unet(PATCH_SIZE)
model.load_weights(latest_ckpt)

# set paths for data
test_data_dir = '../data/test_data'
prediction_dir = os.path.join(test_data_dir, 'predictions')
os.makedirs(prediction_dir, exist_ok=True)
image_files = glob(os.path.join(test_data_dir, 'images/*'))

for image_file in image_files:
    file_name = os.path.basename(image_file)
    image = np.array(Image.open(image_file)) / 255 # rescale the image
    old_size = image.shape[0:2]
    image_block_patches = make_block_patches(image, PATCH_SIZE)
    block_coords = list(image_block_patches.keys())
    test_patches = np.array(list(image_block_patches.values()))
    predicted_patches = model(test_patches, training=False)
    predicted_patches = adjust_prediction(predicted_patches)
    predicted_patches = dict(zip(block_coords, predicted_patches))
    predicted_mask = aggregate_block_patches(predicted_patches, old_size)
    save_file = os.path.join(prediction_dir, file_name[:-4] + '_prediction.tif')
    Image.fromarray(np.uint8(predicted_mask * 255), 'L').save(save_file)
