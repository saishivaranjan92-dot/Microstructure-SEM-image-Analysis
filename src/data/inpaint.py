"""
SEM Micrograph Data Banner Inpainting Module
Removes the microscope acquisition banner at the footer to avoid boundary artifacts.
"""

import numpy as np
import cv2

def inpaint_data_banner(image: np.ndarray, banner_height: int = 64) -> np.ndarray:
    """
    Inpaints the bottom instrument metadata strip using Telea's fast marching algorithm.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input micrograph (H, W) as uint8.
    banner_height : int
        Height of the bottom data strip in pixels (typically 64 px).

    Returns
    -------
    np.ndarray
        Cleaned micrograph with the data strip smoothly inpainted from surrounding matrix.
    """
    h, w = image.shape[:2]
    if banner_height <= 0 or banner_height >= h:
        return image.copy()

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h - banner_height:h, :] = 255

    # Use Telea fast marching inpainting with 3 px radius
    inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted
