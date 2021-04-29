import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import pad

def pad_image(image, new_size, pad_val):
    """
    Pads an image to a desired size.
    Parameters
    ----------
    image (ndarray): (height, width, n_channels), (height, width)
        Image to pad.
    new_size: int, tuple, (new_heght, new_width)
        Image will be padded to (new_height, new_width, n_channels) or
        (new_heght, new_width)
    pad_val: float, listlike value to pad with
    """
    if isinstance(new_size, numbers.Number):
        new_size = (new_size, new_size)
    if image.shape[0] > new_size[0]:
        print('WARNING: image height larger than desired cnn image size')
        return image
    if image.shape[1] > new_size[1]:
        print('WARNING: image width larger than desiered cnn image size')
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
    if isinstance(pad_val, numbers.Number):
        return pad(image, pad_width, mode='constant', constant_values=pad_val)
    else:
        n_channels = image.shape[2]
        return np.stack([pad(image[:, :, c], pad_width[0:2], mode='constant',
                             constant_values=pad_val[c])
                         for c in range(n_channels)], axis=2)

def make_patches(image, patch_size):
    """
    Make patches of a desired size after properly padding the image.
    Parameters
    ----------
    image (ndarray): (height, width, n_channels), (height, width)
        Image to pad.
    patch_size: int, tuple, (patch_heght, patch_width)
        Image will be padded according to the patch_size and then split into
        patches
    pad_val: float, listlike value to pad with
    """
    if isinstance(patch_size, numbers.Number):
        patch_size = (patch_size, patch_size)
    new_size = (np.array(image.shape)//patch_size + 1)*patch_size


def make_rand_patches(image, patch_size, n_patch):
