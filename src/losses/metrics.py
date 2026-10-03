"""
Evaluation Metrics for Binary Semantic Segmentation
Computes Dice, IoU (Jaccard), Precision, Recall, and Pixel Accuracy.
"""

from typing import Dict
import numpy as np

def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Computes standard overlap and classification metrics between binary masks.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask (H, W) with {0, 1}.
    true_mask : np.ndarray
        Ground truth binary mask (H, W) with {0, 1}.
    eps : float
        Numerical epsilon to avoid division by zero.

    Returns
    -------
    Dict[str, float]
        Dictionary with dice, iou, precision, recall, and accuracy.
    """
    p = (pred_mask > 0).astype(np.uint8).ravel()
    y = (true_mask > 0).astype(np.uint8).ravel()

    tp = float(np.sum((p == 1) & (y == 1)))
    fp = float(np.sum((p == 1) & (y == 0)))
    fn = float(np.sum((p == 0) & (y == 1)))
    tn = float(np.sum((p == 0) & (y == 0)))

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / float(len(p))

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }
