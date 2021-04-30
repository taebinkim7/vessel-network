import numpy as np
from numbers import Number
from PIL import Image
from math import ceil, floor
from skimage.util import pad, view_as_blocks

def pad_image(image, new_size, pad_val=0):
    """
    Pads an image to a desired size.

    Parameters
    ----------
    image (ndarray): (height, width, n_channels), (height, width)
        Image to pad
    new_size: int, tuple, (new_heght, new_width)
        Image will be padded to (new_height, new_width, n_channels) or
        (new_heght, new_width)
    pad_val: float, listlike value to pad with
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

def make_patches(image_file, patch_size, pad_val=0, save_dir=None):
    """
    Make and save patches of a desired size after properly padding the image.

    Parameters
    ----------
    image_file: str
        Path to the image file
    patch_size: int, tuple, (patch_heght, patch_width)
        Image will be padded according to the patch_size and then split into
        patches
    pad_val: float, listlike value to pad with
    save_dir: str
        Directory to save the patches
    """
    image = np.array(Image.open(image_file))

    if isinstance(patch_size, Number):
        patch_size = (patch_size, patch_size)
    new_size = (np.array(image.shape[0:2])//patch_size + 1)*patch_size
    image = pad_image(image, new_size, pad_val)

    # make patches
    temp_shape = image.shape[0:2] + (-1,) # for 2-d arrays
    image = image.reshape(temp_shape)
    n_channels = image.shape[2]
    patch_size = patch_size + (n_channels,)
    patches = view_as_blocks(image, patch_size)

    # save patches
    n_rows, n_cols = patches.shape[0:2]
    for i in range(n_rows):
        for j in range(n_cols):
            patch = patches[i, j, 0]
            if patch.shape[2] == 1:
                patch = patch.reshape(patch.shape[0:2]) # for 2-d arrays
            if save_dir is not None:
                image_name = os.path.basename(image_file)[:-4] # omit '.png'
                patch_file = image_name + '_patch_r{}c{}.png'.format(i, j)
            else:
                patch_file = image_file[:-4] + '_patch_r{}c{}.png'.format(i, j)
            Image.fromarray(patch).save(patch_file)

#
# def make_rand_patches(image, patch_size, n_patch):
#
# def aggregate_patches()
