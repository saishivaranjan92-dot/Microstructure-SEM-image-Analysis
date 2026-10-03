"""
Data Augmentation Transforms Module
Constructs Albumentations pipelines for microscopy training and validation.
"""

import albumentations as A
import cv2

def get_training_transforms() -> A.Compose:
    """
    Data augmentation pipeline for training patches.
    Includes spatial symmetries (flips, 90-deg rotations), affine shifts,
    and illumination variations (contrast, brightness, sensor noise).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
    ])

def get_validation_transforms() -> A.Compose:
    """
    Validation pipeline: identity transformation to evaluate raw performance.
    """
    return A.Compose([])
