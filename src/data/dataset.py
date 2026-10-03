"""
PyTorch Dataset Implementation for Micrograph Patches
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class MicrographPatchDataset(Dataset):
    """
    Dataset wrapping extracted (image_patch, mask_patch) arrays.
    Scales grayscale intensities linearly to [0, 1] float32 tensors.
    """

    def __init__(
        self,
        patches: List[Tuple[np.ndarray, np.ndarray]],
        transform: Optional[A.Compose] = None
    ):
        self.patches = patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, msk = self.patches[idx]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=msk)
            img = augmented["image"]
            msk = augmented["mask"]

        # Convert to float32 tensors: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        msk_tensor = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)

        return img_tensor, msk_tensor
