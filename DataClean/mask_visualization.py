#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 3/10/2021 10:17 PM

import numpy as np
import colorsys
import random


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    new_image = np.zeros_like(image)
    for c in range(3):
        new_image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return new_image


def convert_to_RGB255(image, vmin=0, vmax=80, is_1k_shift=False):
    assert len(image.shape) == 2

    # convert to float32
    new_image = image.astype(np.float32) * 1.0
    if is_1k_shift:
        new_image -= 1000

    # adjust the window location
    new_image[new_image < vmin] = vmin
    new_image[new_image > vmax] = vmax
    new_image = (new_image - vmin) / (vmax - vmin) * 255
    new_image = new_image.astype(np.uint8)

    # convert to RGB
    new_image = new_image[:, :, np.newaxis]
    new_image = np.repeat(new_image, repeats=3, axis=-1)
    return new_image


if __name__ == '__main__':
    import nrrd
    import os
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    data_root = r'D:\share\data\Challenge\HeadAndNeckCancerCTAtlas\standardized\HNSCC-01-0001'
    CT_data, _ = nrrd.read(os.path.join(data_root, 'CT.nrrd'))
    mask_data, _ = nrrd.read(os.path.join(data_root, 'Brainstem.nrrd'))
    mandible_data, _ = nrrd.read(os.path.join(data_root, 'Lt_ParotidGland.nrrd'))

    slice_index = 30

    RGB_data = convert_to_RGB255(CT_data[slice_index], vmin=0, vmax=80, is_1k_shift=True)
    color = random_colors(1)[0]
    Colorized_data = apply_mask(RGB_data, mask_data[slice_index], color=[1, 1, 0], alpha=0.5)
    Colorized_data = apply_mask(Colorized_data, mandible_data[slice_index], color=[1, 0, 0], alpha=0.5)
    plt.imshow(Colorized_data)
    plt.show()
    print('Congrats! May the force be with you ...')
