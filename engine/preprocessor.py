"""
Engine Preprocessor: Edge-Preserving Denoising & Cel-Shading Flattening
Combines Bilateral filtering with Mean-Shift segmentation to eliminate compression
artifacts and flatten color gradients within cel regions while preserving sharp edges.
"""

import cv2
import numpy as np


def clean_frame_hybrid(
    img: np.ndarray,
    d: int = 5,
    sigma: float = 35.0,
    sp: int = 6,
    sr: int = 16,
    use_meanshift: bool = True,
    unsharp_strength: float = 0.0
) -> np.ndarray:
    """
    Applies a hybrid edge-preserving filtering pipeline with optional de-blurring:
    1. Unsharp Mask: sharpens soft/motion-blurred gradient transitions.
    2. Bilateral filter: attenuates high-frequency noise & macroblocks without destroying edges.
    3. Mean-shift filter: flattens slight color variations into uniform cel plateaus.
    
    Supports both BGR (3-channel) and RGBA (4-channel) images.
    """
    if img is None or img.size == 0:
        return img

    has_alpha = (img.ndim == 3 and img.shape[2] == 4)
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img

    # Stage 0: Optional Unsharp Masking (Counters motion blur and restores edge gradients)
    if unsharp_strength > 0:
        blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3.0)
        sharpened = cv2.addWeighted(bgr, 1.0 + unsharp_strength, blurred, -unsharp_strength, 0)
        filtered = np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        filtered = bgr

    # Stage 1: Bilateral Filter
    if d > 0:
        filtered = cv2.bilateralFilter(filtered, d=d, sigmaColor=sigma, sigmaSpace=sigma)

    # Stage 2: Mean-Shift Segmentation (Cel flattening)
    if use_meanshift and sp > 0 and sr > 0:
        # maxLevel=0 preserva a nitidez das bordas de personagens flat
        filtered = cv2.pyrMeanShiftFiltering(filtered, sp=sp, sr=sr, maxLevel=0)

    if has_alpha:
        return np.dstack([filtered, alpha])
    return filtered
