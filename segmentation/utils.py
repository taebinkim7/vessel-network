import os
import numpy as np
import tensorflow as tf
from glob import glob
from numbers import Number
from PIL import Image
from math import ceil, floor
from skimage.util import pad, view_as_blocks, view_as_windows

def pad_image(image, new_size, pad_val=0):
    """
    Pads an image to a desired size.

    Parameters
    ----------
    image: ndarray, {(height, width, n_channels), (height, width)}
        Image to pad with
    new_size: {int, tuple}, (new_height, new_width)
        Image will be padded to the shape of (new_height, new_width, n_channels)
        or (new_heght, new_width)
    pad_val: {float, list-like}
        Values to pad with

    Returns
    -------
    image: nadarray, {(new_size[0], new_size[1], n_channels), new_size}
        Padded image
    """
    if isinstance(new_size, Number):
        new_size = (new_size, new_size)
    if image.shape[0] > new_size[0]:
        print('WARNING: image height larger than desired image size')
        return image
    if image.shape[1] > new_size[1]:
        print('WARNING: image width larger than desired image size')
        return image

    # width padding
    width_diff = new_size[1] - image.shape[1]

    # how much padding to add
    left = floor(width_diff / 2)
    right = ceil(width_diff / 2)
    height_diff = new_size[0] - image.shape[0]

    top = floor(height_diff / 2)
    bottom = ceil(height_diff / 2)
    pad_width = ((top, bottom), (left, right), (0, 0))

    # make work with 2-d arrays
    if len(image.shape) == 2:
        pad_width = pad_width[0:2]
    if isinstance(pad_val, Number):
        return pad(image, pad_width, mode='constant', constant_values=pad_val)
    else:
        n_channels = image.shape[2]
        return np.stack([pad(image[:, :, c], pad_width[0:2], mode='constant',
                             constant_values=pad_val[c])
                         for c in range(n_channels)], axis=2)


def make_train_patches(image, patch_size, step, pad_val=0, save_dir=None):
    """
    Call rolling window view of an image and save them in a list.

    Parameters
    ----------
    image: ndarray, {(height, width, n_channels), (height, width)}
        Image to make patches with
    patch_size: {int, tuple}, (patch_height, patch_width)
        Image will be padded according to the patch_size
    step: int
        Number of elements to skip when moving the window forward. We only use
        a factor of patch_size as step in this function
    save_dir: str, optional
        Directory to save the patches

    Returns
    -------
    patches_list: list
    """
    # pad the image
    if isinstance(patch_size, Number):
        patch_size = (patch_size, patch_size)
    new_size = (np.array(image.shape[0:2])//patch_size + 1)*patch_size
    image = pad_image(image, new_size, pad_val)

    # make patches
    temp_shape = image.shape[0:2] + (-1,) # for 2-d arrays
    image = image.reshape(temp_shape)
    n_channels = image.shape[2]
    patch_size = patch_size + (n_channels,)
    window_view = view_as_windows(image, patch_size, step)
    n_rows, n_cols = window_view.shape[0:2]
    patches = [window_view[i, j, 0] for i in range(n_rows) \
                                    for j in range(n_cols)]

    return patches


def make_block_patches(image, patch_size, pad_val=0, save_dir=None):
    """
    Call block view of an image and save them in a dictionary.

    Parameters
    ----------
    image: ndarray, {(height, width, n_channels), (height, width)}
        Image to make patches with
    patch_size: {int, tuple}, (patch_height, patch_width)
        Image will be padded according to the patch_size and then split into
        patches
    pad_val: {float, list-like}
        Values to pad with
    save_dir: str, optional
        Directory to save the patches

    Returns
    -------
    block_patches: dictionary
        Dictionary with block coordinates as keys and patches as values
    """
    # pad the image
    if isinstance(patch_size, Number):
        patch_size = (patch_size, patch_size)
    new_size = (np.array(image.shape[0:2])//patch_size + 1)*patch_size
    image = pad_image(image, new_size, pad_val)

    # make patches
    temp_shape = image.shape[0:2] + (-1,) # for 2-d arrays
    image = image.reshape(temp_shape)
    n_channels = image.shape[2]
    patch_size = patch_size + (n_channels,)
    block_view = view_as_blocks(image, patch_size)

    # make dictionary
    # Note: From Python 3.6 onwards, the standard dict type maintains insertion
    # order by default

    block_patches = {}

    n_rows, n_cols = block_view.shape[0:2]
    for i in range(n_rows):
        for j in range(n_cols):
            patch = block_view[i, j, 0]
            # if patch.shape[2] == 1:
            #     patch = patch.reshape(patch.shape[0:2]) # for 2-d arrays
            block_patches[i, j] = patch

    return block_patches


def aggregate_block_patches(block_patches, old_size=None, save_dir=None):
    """
    Aggregate block patches to construct an image.

    Parameters
    ----------
    block_patches: dictionary
        Dictionary with block coordinates as keys and patches as values
    old_size: {int, tuple}, (old_height, old_width)
        Image will be resized to the original shape

    Returns
    -------
    image: ndarray
    """
    block_coords = list(block_patches.keys())
    n_rows, n_cols = block_coords[-1]
    patches = [[block_patches[i, j] for j in range(n_cols)] \
                                        for i in range(n_rows)]
    image = np.hstack(np.hstack(patches))

    if old_size is not None:
        if isinstance(old_size, Number):
            old_size = (old_size, old_size)
        h, w = image.shape[0:2]
        assert (old_size[0] <= h) & (old_size[1] <= w)
        image = image[(h - old_size[0])//2:(h + old_size[0])//2,
                      (w - old_size[1])//2:(w + old_size[1])//2, :]

    if image.shape[2] == 1:
        image = image.reshape(image.shape[0:2]) # for 2-d arrays
    elif np.max(image) <= 1:
        image *= 255.0
        image = image.astype(np.uint8)

    return image

def vessel_threshold(image, alpha=0.01):
    return np.sum(image)/image.size > alpha

def get_train_dataset(data_dir, patch_size, step, batch_size, alpha,
                      threshold=True):
    image_files = glob(os.path.join(data_dir, 'images/*'))
    mask_files = glob(os.path.join(data_dir, 'masks/*'))

    # get patches
    image_patches, mask_patches = [], []
    for image_file, mask_file in zip(image_files, mask_files):
    	image = np.array(Image.open(image_file))/255 # rescale the image
    	mask = np.array(Image.open(mask_file)).astype(int)
    	image_patches += make_train_patches(image, patch_size, step)
    	mask_patches += make_train_patches(mask, patch_size, step)

    # drop patches with vessel area less than the threshold
    if threshold:
        vessel_idx = list(map(vessel_threshold,
                              mask_patches,
                              [alpha for i in range(len(mask_patches))]))
        image_patches = np.array(image_patches)[vessel_idx]
        mask_patches = np.array(mask_patches)[vessel_idx]

    # get dataset
    image_patches = tf.constant(image_patches)
    mask_patches = tf.constant(mask_patches)
    train_dataset = tf.data.Dataset.from_tensor_slices((image_patches, mask_patches))
    train_dataset = train_dataset.batch(batch_size)

    return train_dataset

def adjust_prediction(prediction):
    prediction = np.array(prediction)
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0

    return prediction

# def save_patches

# if save_dir is not None:
#     os.makedirs('save_dir', exist_ok=True)
#     image_name = os.path.basename(image_file)[:-4] # omit '.png'
#     patch_file = os.path.join(save_dir, image_name + \
#                               '_patch_r{}c{}.png'.format(i, j))
# else:
#     patch_file = image_file[:-4] + '_patch_r{}c{}.png'.format(i, j)
# Image.fromarray(patch).save(patch_file)
#
# def make_rand_patches(image, patch_size, n_patch):
#
