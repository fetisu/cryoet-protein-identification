# -*- coding: utf-8 -*-
import random

import numpy as np


def rotate_90_3d(voxel, mask, k_d=None, k_h=None, k_w=None):
    """
    Rotate a 3D voxel by 90 degrees.

    Args:
        voxel (numpy.ndarray): Voxel data to be rotated (C, D, H, W).
        mask (numpy.ndarray): Segmentation mask corresponding
        to the voxel (C, D, H, W).
        k_d (int): Number of 90-degree rotations along the Depth (D) axis.
        If None, a random value is chosen.
        k_h (int): Number of 90-degree rotations along the Height (H) axis.
        If None, a random value is chosen.
        k_w (int): Number of 90-degree rotations along the Width (W) axis.
        If None, a random value is chosen.

    Returns:
        tuple: (rotated voxel data, rotated mask data)
    """
    if k_d is None:
        k_d = random.randint(0, 3)
    if k_h is None:
        k_h = random.randint(0, 3)
    if k_w is None:
        k_w = random.randint(0, 3)

    # Rotation along the Depth axis (using H and W as the rotation axes)
    # voxel = np.rot90(voxel, k=k_d, axes=(-3, -2)).copy()
    # mask = np.rot90(mask, k=k_d, axes=(-3, -2)).copy()

    # Rotation along the Height axis (using D and W as the rotation axes)
    # voxel = np.rot90(voxel, k=k_h, axes=(-3, -1)).copy()
    # mask = np.rot90(mask, k=k_h, axes=(-3, -1)).copy()

    # Rotation along the Width axis (using D and H as the rotation axes)
    voxel = np.rot90(voxel, k=k_w, axes=(-2, -1)).copy()
    mask = np.rot90(mask, k=k_w, axes=(-2, -1)).copy()

    return voxel, mask


def random_flip_3d(
    voxel: np.ndarray,
    mask: np.ndarray,
    p_z: float = 0.5,
    p_y: float = 0.5,
    p_x: float = 0.5,
    seed: int = None,
) -> np.ndarray:
    """
    Randomly flip 3D voxel data (D, H, W) along the z-axis (D),
    y-axis (H), and x-axis (W).

    Parameters
    ----------
    voxel : np.ndarray
        Voxel data in the form of a 3D array (D, H, W).
    mask : np.ndarray
        Segmentation mask data in the form of a 3D array (D, H, W).
    p_z : float, optional
        Probability of flipping along the z-axis (D direction)
        (range 0.0 to 1.0),
        default is 0.5.
    p_y : float, optional
        Probability of flipping along the y-axis (H direction)
        (range 0.0 to 1.0),
        default is 0.5.
    p_x : float, optional
        Probability of flipping along the x-axis (W direction)
        (range 0.0 to 1.0),
        default is 0.5.
    seed : int, optional
        Random seed (used to ensure reproducibility), default is None.

    Returns
    -------
    np.ndarray
        Voxel data after flipping (shape remains (C, D, H, W)).

    Notes
    -----
    - The shape remains unchanged as (C, D, H, W).
    - The axis flipping is not performed in-place;
    a newly generated array is returned.
    """

    # Flip along the z-axis (Depth, D).
    if np.random.rand() < p_z:
        voxel = voxel[::-1, :, :].copy()
        mask = mask[:, ::-1, :, :].copy()

    # Flip along the y-axis (Height, H).
    if np.random.rand() < p_y:
        voxel = voxel[:, ::-1, :].copy()
        mask = mask[:, :, ::-1, :].copy()

    # Flip along the x-axis (Width, W).
    if np.random.rand() < p_x:
        voxel = voxel[:, :, ::-1].copy()
        mask = mask[:, :, :, ::-1].copy()

    return voxel, mask


def random_intensity_shift(
    voxel: np.ndarray, shift_range: tuple = (-0.1, 0.1), seed: int = None
) -> np.ndarray:
    """
    Apply a random intensity shift to 3D voxel data.

    Parameters
    ----------
    voxel : np.ndarray
        Voxel data to which the intensity shift is applied
        (any shape, e.g., (C, D, H, W) or (D, H, W)).
    shift_range : tuple, optional
        Range of shift values (min_shift, max_shift).
        The default is (-0.1, 0.1).
    seed : int, optional
        Random seed (used to ensure reproducibility), default is None.

    Returns
    -------
    np.ndarray
        Voxel data after applying the intensity shift
        (the shape remains the same as the input).

    Notes
    -----
    - Adjust the shift range according to the scale of the voxel values.
    - If necessary, the results can be clipped to restrict the range of values.
    """

    shift_value = np.random.uniform(shift_range[0], shift_range[1])
    voxel += shift_value

    return voxel
