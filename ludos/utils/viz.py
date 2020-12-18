import numpy as np


def apply_masks(image, mask, alpha=0.5):
    """
    Blend the mask within the images - #vectorize
    """
    image = image[:, :, :3]
    m = np.stack([mask[:, :, 0]] * 3).transpose(1, 2, 0) * np.array(
        [0, 0, 153]) / 255.
    mask_img = (m * alpha + image * (1 - alpha)).astype('uint8')
    img = mask_img * (m != 0) + image * (m == 0)
    return img
