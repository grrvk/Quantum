import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple
import cv2
import numpy as np


def to_grayscale(image: np.ndarray, **kwargs) -> np.ndarray:
    """Convert RGB image to grayscale (1 channel)"""
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def add_channel_dim(image: np.ndarray, **kwargs) -> np.ndarray:
    """Add channel dimension if missing"""
    if len(image.shape) == 2:
        return image[:, :, np.newaxis]
    return image

def to_float(image: np.ndarray, **kwargs) -> np.ndarray:
    """Convert image to float32 in [0, 1] range"""
    return image.astype(np.float32)


def get_train_transforms():
    """Simple train transforms without resize for image only"""

    return A.Compose([
        A.Lambda(name='ToGrayscale', image=to_grayscale, p=1.0),
        A.Lambda(name='AddChannelDim', image=add_channel_dim, p=1.0),
        A.OneOf([
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)},
                rotate=(-30, 30),
                p=0.8
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                border_mode=cv2.BORDER_REFLECT,
                p=0.2
            ),
        ], p=1.0),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.Lambda(name='ToFloat', image=to_float, p=1.0),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Simple validation transforms without resize for image only"""

    return A.Compose([
        A.Lambda(name='ToGrayscale', image=to_grayscale, p=1.0),
        A.Lambda(name='AddChannelDim', image=add_channel_dim, p=1.0),
        A.Lambda(name='ToFloat', image=to_float, p=1.0),
        ToTensorV2(),
    ])

def get_train_transforms_with_resize(target_size: Tuple[int, int]):
    """Train transforms that include Resize for image and keypoints"""

    return A.Compose([
        A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_AREA),
        A.Lambda(name='ToGrayscale', image=to_grayscale, p=1.0),
        A.Lambda(name='AddChannelDim', image=add_channel_dim, p=1.0),
        A.OneOf([
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)},
                rotate=(-30, 30),
                p=0.8,
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                border_mode=cv2.BORDER_REFLECT,
                p=0.2
            ),
        ], p=0.8),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.Lambda(name='ToFloat', image=to_float, p=1.0),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def get_val_transforms_with_resize(target_size: Tuple[int, int]):
    """Validation transforms that include Resize for image and keypoints"""

    return A.Compose([
        A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_AREA),
        A.Lambda(name='ToGrayscale', image=to_grayscale, p=1.0),
        A.Lambda(name='AddChannelDim', image=add_channel_dim, p=1.0),
        A.Lambda(name='ToFloat', image=to_float, p=1.0),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
