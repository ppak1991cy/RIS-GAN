"""
    RIS-GAN
    func  : 数据加载工具
    Author: Chen Yu
    Date  : 2020.10.20
"""
import os
import random
import numpy as np
from PIL import Image


def match_samples_path(data_dir):
    sub_dirs = os.listdir(data_dir)
    shadow_images_dir = ''
    shadow_removal_images_dir = ''
    for sub_dir in sub_dirs:
        if sub_dir.endswith('_A'):
            shadow_images_dir = os.path.join(data_dir, sub_dir)
        elif sub_dir.endswith('_C'):
            shadow_removal_images_dir = os.path.join(data_dir, sub_dir)
    shadow_names = os.listdir(shadow_images_dir)
    shadow_removal_names = os.listdir(shadow_removal_images_dir)
    match_results = []
    for shadow_name in shadow_names:
        if shadow_name in shadow_removal_names:
            shadow_image_path = os.path.join(shadow_images_dir, shadow_name)
            shadow_removal_image_path = os.path.join(shadow_removal_images_dir, shadow_name)
            match_results.append((shadow_image_path, shadow_removal_image_path))
    return match_results


def sample_transform(shadow_image, shadow_removal_image, img_size=(256, 256)):
    flip_direction = random.randint(0, 3)
    if flip_direction == 1:
        shadow_image = shadow_image.transpose(Image.FLIP_LEFT_RIGHT)
        shadow_removal_image = shadow_removal_image.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_direction == 2:
        shadow_image = shadow_image.transpose(Image.FLIP_TOP_BOTTOM)
        shadow_removal_image = shadow_removal_image.transpose(Image.FLIP_TOP_BOTTOM)

    rotate_degree = random.randint(-45, 45)
    shadow_image = shadow_image.rotate(rotate_degree).resize(img_size)
    shadow_removal_image = shadow_removal_image.rotate(rotate_degree).resize(img_size)

    return shadow_image.resize(img_size), shadow_removal_image.resize(img_size)


def _normalize(data, mean=0.5, std=0.5):
    data = (data - mean) / std
    return data


def load_image_pair(match_result, img_size=(256, 256)):
    shadow_image = Image.open(match_result[0]).resize(img_size)
    shadow_removal_image = Image.open(match_result[1]).resize(img_size)
    shadow_image, shadow_removal_image = sample_transform(shadow_image, shadow_removal_image, img_size)
    shadow_data = np.asarray(shadow_image).astype(np.float) / 255
    shadow_removal_data = np.asarray(shadow_removal_image).astype(np.float) / 255
    # shadow_data[shadow_data == 0] = 1e-7
    # shadow_removal_data[shadow_removal_data == 0] = 1e-7
    sample = (_normalize(shadow_data), _normalize(shadow_removal_data))
    return sample


def get_negative_residual_image(shadow_data, shadow_removal_data):
    negative_residual_image = shadow_removal_data - shadow_data
    return negative_residual_image


def get_inverse_illuminate_map(shadow_data, shadow_removal_data):
    inverse_illuminate_map = shadow_removal_data / shadow_data
    return inverse_illuminate_map











