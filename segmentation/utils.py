import os
import numpy as np
import tensorflow as tf
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


def make_patches(image, patch_size, step, save_dir=None):
    """
    Call rolling window view of an image and save them in a dictionary.

    Parameters
    ----------
    image: ndarray, {(height, width, n_channels), (height, width)}
        Image to make patches with
    patch_size: {int, tuple}, (patch_height, patch_width)
        Image will be padded according to the patch_size and then split into
        patches
    step: int
        Number of elements to skip when moving the window forward
    save_dir: str, optional
        Directory to save the patches

    Returns
    -------
    patches_list: list
    """
    if isinstance(patch_size, Number):
        patch_size = (patch_size, patch_size)

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


def aggregate_block_patches(block_patches, save_dir=None):
    """
    Aggregate block patches to construct an image.

    Parameters
    ----------
    block_patches: dictionary
        Dictionary with block coordinates as keys and patches as values

    Returns
    -------
    image: ndarray
    """
    block_coords = list(block_patches.keys())
    n_rows, n_cols = block_coords[-1]
    patches = [[block_patches[i, j] for j in range(n_cols)] \
                                        for i in range(n_rows)]
    image = np.hstack(np.hstack(patches))
    if image.shape[2] == 1:
        image = image.reshape(image.shape[0:2]) # for 2-d arrays

    return image

def vessel_threshold(image, alpha=0.01):
    return np.sum(image)/image.size > alpha

def get_dataset(image_patches, label_patches, alpha=0.01):
    vessel_idx = list(map(vessel_threshold,
                          label_patches, [alpha for i in len(label_patches)]))
    image_patches = np.array(image_patches)[vessel_idx]
    label_patches = np.array(label_patches)[vessel_idx]
    train_images = tf.constant(image_patches)
    train_labels = tf.constant(label_patches)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images,
                                                        train_labels))

    return train_dataset
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
