from pathlib import Path
from typing import Tuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from cellpose import models as cp_models


def get_motion_correction_crop_xy_range_from_both_planes(
    oeid: int, paired_id: int, input_dir: Path
) -> Tuple[list, list]:
    """Get x-y ranges to crop motion-correction frame rolling from both planes

    TODO: when nonrigid registration parameter setting is done,
    include nonrigid shift max into the calculation.

    Parameters
    ----------
    oeid : int
        ophys experiment ID
    paired_id : int
        ophys experiment ID of the paired plane
    input_dir : Path
        path to input directory

    Returns
    -------
    list, list
        Lists of y range and x range, [start, end] pixel index
    """
    xrange_og, yrange_og = get_motion_correction_crop_xy_range(oeid, input_dir)
    xrange_paired, yrange_paired = get_motion_correction_crop_xy_range(
        paired_id, input_dir
    )

    xrange = [max(xrange_og[0], xrange_paired[0]), min(xrange_og[1], xrange_paired[1])]
    yrange = [max(yrange_og[0], yrange_paired[0]), min(yrange_og[1], yrange_paired[1])]

    return xrange, yrange


def get_motion_correction_crop_xy_range(
    oeid: int, input_dir: Path
) -> Tuple[list, list]:
    """Get x-y ranges to crop motion-correction frame rolling

    TODO: move to utils

    Parameters
    ----------
    oeid : int
        ophys experiment ID
    input_dir : Path
        path to input directory
    Returns
    -------
    list, list
        Lists of y range and x range, [start, end] pixel index
    """
    # TODO: validate in case where max < 0 or min > 0 (if there exists an example)
    suite2p_rigid_motion_transform_csv = (
        input_dir / oeid / "motion_correction" / f"{oeid}_motion_transform.csv"
    )
    motion_df = pd.read_csv(
        suite2p_rigid_motion_transform_csv
    )  # this is suite2p rigid motion transform csv file
    max_y = np.ceil(max(motion_df.y.max(), 1)).astype(int)
    min_y = np.floor(min(motion_df.y.min(), 0)).astype(int)
    max_x = np.ceil(max(motion_df.x.max(), 1)).astype(int)
    min_x = np.floor(min(motion_df.x.min(), 0)).astype(int)
    range_y = [-min_y, -max_y]
    range_x = [-min_x, -max_x]
    return range_y, range_x


def decrosstalk_roi_image_from_episodic_mean_fov(
    oeid: int,
    paired_reg_fn: Path,
    input_dir: Path,
    pixel_size: float = 0.78,
    grid_interval: float = 0.01,
    max_grid_val: float = 0.3,
    return_recon: float = False,
) -> Tuple[np.array, list, list, list]:
    """Get alpha and beta values for an experiment based on
    the mutual information of the ROI images from motion corrected episodic mean FOV images

    Parameters:
    -----------
    oeid : int
        oeid of the signal plane
    paired_reg_fn : Path
        path to paired registration file
        TODO: Once paired plane registration pipeline is finalized,
        this parameter can be removed or replaced with paired_oeid
    input_dir: Path
        path to the input directory
    pixel_size: float, optional
        pixel size in um of imaging plane, (400pixelsx400pixels 512umx512um)
    grid_interval : float, optional
        interval of the grid, by default 0.01
    max_grid_val : float, optional
        maximum value of alpha and beta, by default 0.3
    return_recon : bool, optional
        whether to return the reconstructed signal and paired images, by default True

    Returns:
    -----------
    alpha_list : list
        list of alpha values across epochs
    beta_list : list
        list of beta values across epochs
    mean_norm_mi_list : list
        list of mean normalized mutual information values across epochs
    """

    # Assign start frames for each epoch
    signal_fn = (
        Path("../results")
        / oeid
        / "decrosstalk"
        / f"{oeid}_registered_episodic_mean_fov.h5"
    )
    with h5py.File(signal_fn, "r") as f:
        data_length = f["data"].shape[0]
        signal_data = f["data"][()]
    start_frames = range(data_length)

    alpha_list = []
    beta_list = []
    mean_norm_mi_list = []
    for start_frame in start_frames:
        (
            alpha,
            beta,
            mean_norm_mi_values,
        ) = decrosstalk_roi_image_single_pair_from_episodic_mean_fov(
            oeid,
            paired_reg_fn,
            input_dir,
            start_frame,
            pixel_size,
            grid_interval=grid_interval,
            max_grid_val=max_grid_val,
        )
        alpha_list.append(alpha)
        beta_list.append(beta)
        mean_norm_mi_list.append(mean_norm_mi_values)

    alpha = np.mean(alpha_list)
    beta = np.mean(beta_list)

    if return_recon:
        with h5py.File(signal_fn, "r") as f:
            signal_data = f["data"][:]
        with h5py.File(paired_reg_fn, "r") as f:
            paired_data = f["data"][:]
        recon_signal_data = np.zeros_like(signal_data)
        for i in range(data_length):
            recon_signal_data[i, :, :] = apply_mixing_matrix(
                alpha, beta, signal_data[i, :, :], paired_data[i, :, :]
            )[0]
    else:
        recon_signal_data = None
    return recon_signal_data, alpha_list, beta_list, mean_norm_mi_list


def decrosstalk_roi_image_single_pair_from_episodic_mean_fov(
    oeid: int,
    paired_reg_emf_fn: str,
    input_dir: Path,
    start_frame: int,
    pix_size: float,
    motion_buffer: int = 5,
    grid_interval: float = 0.01,
    max_grid_val: float = 0.3,
) -> Tuple[float, float, list]:
    """Get alpha and beta values for a single pair of mean images
    based on the mean normalized mutual information of the ROI images

    Parameters:
    -----------
    oeid : int
        ophys experiment id
    paired_reg_emf_fn : str, Path
        path to paired registration file
        TODO: Once paired plane registration pipeline is finalized,
        this parameter can be removed or replaced with paired_oeid
    input_dir: Path
        path to the input directory
    start_frame: int
        start frame of the mean images
    pix_size = float
        pixel size in um of imaging plane
    motion_buffer : int, optional
        number of pixels to crop from the nonrigid motion corrected image, by default 5
        TODO: Get this from the suite2p parameters
    grid_interval : float, optional
        interval of the grid, by default 0.01
    max_grid_val : float, optional
        maximum value of alpha and beta, by default 0.3

    Returns:
    -----------
    alpha : float
        alpha value of the unmixing matrix
    beta : float
        beta value of the unmixing matrix
    mean_norm_mi_values : np.array
        mean normalized mutual information values
    """
    signal_fn = (
        Path("../results")
        / oeid
        / "decrosstalk"
        / f"{oeid}_registered_episodic_mean_fov.h5"
    )
    with h5py.File(signal_fn, "r") as f:
        signal_mean = f["data"][start_frame : start_frame + 1].mean(axis=0)
    with h5py.File(paired_reg_emf_fn, "r") as f:
        paired_mean = f["data"][start_frame : start_frame + 1].mean(axis=0)
    paired_id = paired_reg_emf_fn.parent.parent.name
    p1y, p1x = get_motion_correction_crop_xy_range_from_both_planes(
        oeid, paired_id, input_dir
    )
    signal_mean = signal_mean[
        p1y[0] + motion_buffer : p1y[1] - motion_buffer,
        p1x[0] + motion_buffer : p1x[1] - motion_buffer,
    ]
    paired_mean = paired_mean[
        p1y[0] + motion_buffer : p1y[1] - motion_buffer,
        p1x[0] + motion_buffer : p1x[1] - motion_buffer,
    ]

    # Get the top masks of the signal and paired planes
    signal_top_masks, paired_top_masks = get_signal_paired_top_masks(
        signal_mean, paired_mean, pix_size=pix_size
    )  # About 22 s
    # Create bounding boxes
    signal_bb_masks = get_bounding_box(signal_top_masks)
    paired_bb_masks = get_bounding_box(paired_top_masks)
    bb_masks = np.concatenate([signal_bb_masks, paired_bb_masks])
    # Calculate raw mutual information for normalization
    mi_raw = np.zeros(len(bb_masks))
    for bi, mask in enumerate(bb_masks):
        bb_yx = np.where(mask)
        mi_raw[bi] = skimage.metrics.normalized_mutual_information(
            signal_mean[bb_yx], paired_mean[bb_yx]
        )

    # Grid search for alpha and beta using mean normalized mutual information across ROIs
    data = np.vstack((signal_mean.ravel(), paired_mean.ravel()))

    alpha_list = np.arange(0, max_grid_val + grid_interval, grid_interval)
    beta_list = np.arange(0, max_grid_val + grid_interval, grid_interval)
    ab_pair = []
    mean_norm_mi_values = []
    for alpha in alpha_list:
        for beta in beta_list:
            temp_mixing = np.array([[1 - alpha, beta], [alpha, 1 - beta]])
            temp_unmixing = np.linalg.inv(temp_mixing)
            temp_unmixed_data = np.dot(temp_unmixing, data)
            temp_recon_signal = temp_unmixed_data[0, :].reshape(signal_mean.shape)
            temp_recon_paired = temp_unmixed_data[1, :].reshape(signal_mean.shape)
            temp_mi = np.zeros(len(bb_masks))
            for bi, mask in enumerate(bb_masks):
                bb_yx = np.where(mask)
                temp_mi[bi] = skimage.metrics.normalized_mutual_information(
                    temp_recon_signal[bb_yx], temp_recon_paired[bb_yx]
                )
            norm_mi = temp_mi / mi_raw
            mean_norm_mi_values.append(norm_mi.mean())
            ab_pair.append([alpha, beta])

    alpha, beta = ab_pair[np.argmin(mean_norm_mi_values)]
    return alpha, beta, np.array(mean_norm_mi_values).tolist()


def get_signal_paired_top_masks(
    signal_mean: np.array,
    paired_mean: np.array,
    dendrite_diameter_um: int = 10,
    pix_size: float = 0.78,
    nrshiftmax: int = 5,
    overlap_threshold: int = 0.7,
    num_top_rois: int = 15,
) -> Tuple[np.array, np.array]:
    """Get top masks of 2 paired mean images
    Apply CellPose to get the masks, then filter dendrites and border ROIs
    Then get the top n intensity masks from both planes

    There can be duplicates due to excessive crosstalk:
    - Identify duplicate ROIs based on the overlap between the masks of the two planes
    - Remove the one with lower rank in intensity from all ROIs in the corresponding plane

    Parameters:
    -----------
    signal_mean : np.array
        mean image of the signal plane
    paired_mean : np.array
        mean image of the paired plane
    dendrite_diameter_um : float, optional
        diameter of dendrite in um, by default 10
    pix_size : float, optional
        pixel size in um, by default 0.78
    nrshiftmax : int, optional
        number of pixels to crop from the nonrigid motion corrected image, by default 5
        #TODO: Get this from the suite2p parameters
    overlap_threshold : float, optional
        threshold of overlap between signal and paired masks, by default 0.7
    num_top_rois : int, optional
        number of top ROIs to keep, by default 15

    Returns:
    -----------
    signal_top_masks : np.array
        top masks of the signal plane
    paired_top_masks : np.array
        top masks of the paired plane
    """

    model = cp_models.Cellpose(gpu=False, model_type="cyto")
    signal_masks, _, _, _ = model.eval(signal_mean, diameter=None, channels=[0, 0])

    dendrite_diameter_px = dendrite_diameter_um / pix_size
    signal_masks_dendrite_filtered = filter_dendrite(
        signal_masks, dendrite_diameter_pix=dendrite_diameter_px
    )
    signal_masks_filtered = filter_border_roi(
        signal_masks_dendrite_filtered, buffer_pix=nrshiftmax
    )

    paired_masks, _, _, _ = model.eval(paired_mean, diameter=None, channels=[0, 0])
    paired_masks_dendrite_filtered = filter_dendrite(
        paired_masks, dendrite_diameter_pix=dendrite_diameter_px
    )
    paired_masks_filtered = filter_border_roi(
        paired_masks_dendrite_filtered, buffer_pix=nrshiftmax
    )

    signal_masks_filtered = reorder_mask(signal_masks_filtered)
    paired_masks_filtered = reorder_mask(paired_masks_filtered)

    num_signal_masks = np.max(signal_masks_filtered)
    num_paired_masks = np.max(paired_masks_filtered)
    overlap_matrix = np.zeros((num_signal_masks, num_paired_masks))
    for i in range(1, num_signal_masks + 1):
        for j in range(1, num_paired_masks + 1):
            overlap_matrix[i - 1, j - 1] = np.sum(
                (signal_masks_filtered == i) & (paired_masks_filtered == j)
            ) / np.sum((signal_masks_filtered == i) | (paired_masks_filtered == j))

    signal_ranks, _ = get_ranks_roi_inds(signal_mean, signal_masks_filtered)
    paired_ranks, _ = get_ranks_roi_inds(paired_mean, paired_masks_filtered)

    signal_masks_overlap_filtered = signal_masks_filtered.copy()
    paired_masks_overlap_filtered = paired_masks_filtered.copy()
    # remove lower rank overlaps
    for si, pi in zip(
        np.where(overlap_matrix > overlap_threshold)[0],
        np.where(overlap_matrix > overlap_threshold)[1],
    ):
        rank_signal = signal_ranks[si]
        rank_paired = paired_ranks[pi]
        if rank_signal < rank_paired:
            paired_masks_overlap_filtered[paired_masks_overlap_filtered == pi + 1] = 0
        else:
            signal_masks_overlap_filtered[signal_masks_overlap_filtered == si + 1] = 0

    signal_top_masks = get_top_intensity_mask(
        signal_mean, signal_masks_overlap_filtered, num_top_rois=num_top_rois
    )
    paired_top_masks = get_top_intensity_mask(
        paired_mean, paired_masks_overlap_filtered, num_top_rois=num_top_rois
    )
    return signal_top_masks, paired_top_masks


def filter_dendrite(
    masks: np.array, dendrite_diameter_pix: float = (10 / 0.78)
) -> np.array:
    """Filter dendrites from masks based on area threshold

    Input Parameters
    ----------------
    masks: 2d array, each ROI has a unique integer value
    dendrite_diameter_pix: float, diameter of dendrite in pix

    Returns
    -------
    filtered_mask: 2d array, after filtering
        Note - the filtered mask is not necessarily contiguous
    """
    dendrite_radius = dendrite_diameter_pix / 2
    area_threshold = np.pi * dendrite_radius**2
    num_roi = np.max(masks)
    filtered_mask = masks.copy()
    for roi_id in range(1, num_roi + 1):
        roi_mask = masks == roi_id
        roi_area = np.sum(roi_mask)
        if roi_area < area_threshold:
            filtered_mask[roi_mask] = 0
    return filtered_mask


def filter_border_roi(masks: np.array, buffer_pix: int = 5) -> np.array:
    """Filter ROIs that are too close to the border of the FOV

    Input Parameters
    ----------------
    masks: 2d array, each ROI has a unique integer value
    border_width_pix: int, width of border in pix

    Returns
    -------
    filtered_mask: 2d array, after filtering
    """
    border_mask = np.zeros(masks.shape, dtype=bool)
    border_mask[:buffer_pix, :] = 1
    border_mask[-buffer_pix:, :] = 1
    border_mask[:, :buffer_pix] = 1
    border_mask[:, -buffer_pix:] = 1

    filtered_mask = masks.copy()
    num_roi = np.max(masks)
    for roi_id in range(1, num_roi + 1):
        roi_mask = masks == roi_id
        if np.any(roi_mask & border_mask):
            filtered_mask[roi_mask] = 0

    return filtered_mask


def get_top_intensity_mask(
    img: np.array, mask: np.array, num_top_rois: int = 15
) -> np.array:
    """Get top intensity mask

    Parameters:
    -----------
    img : np.array
        the image to calculate mean intensity of the ROIs from
    mask : np.array
        ROI mask
    num_top_rois : int, optional
        number of top ROIs to keep, by default 15

    Returns:
    -----------
    top_mask : np.array
        top num_top_rois masks (in 2D) based on the intensity of img
    """
    roi_inds = np.setdiff1d(np.unique(mask), 0)
    num_roi = len(roi_inds)
    if num_roi < num_top_rois:
        return mask
    else:
        mean_intensities = [img[mask == i].mean() for i in roi_inds]
        sorted_inds = np.argsort(mean_intensities)[::-1]
        top_inds = sorted_inds[:num_top_rois]
        top_ids = roi_inds[top_inds]

        top_mask = mask.copy()
        for i in roi_inds:
            if i not in top_ids:
                top_mask[mask == i] = 0
        return top_mask


def get_ranks_roi_inds(img: np.array, mask: np.array) -> Tuple[np.array, np.array]:
    """Get ranks and ROI indices based on the intensity of img
    For each ROI ID in the mask

    Parameters:
    -----------
    img : np.array (2D)
        the image to calculate mean intensity of the ROIs from
    mask : np.array
        ROI mask (2D from CellPose)

    Returns:
    -----------
    ranks : np.array
        ranks of the ROIs based on the intensity of img
    roi_inds : np.array
        ROI indices
    """
    roi_inds = np.setdiff1d(np.unique(mask), 0)
    mean_intensities = np.array([img[mask == i].mean() for i in roi_inds])
    ranks = np.zeros_like(mean_intensities)
    ranks[np.argsort(mean_intensities)[::-1]] = np.arange(len(mean_intensities))
    return ranks, roi_inds


def reorder_mask(mask: np.array) -> np.array:
    """Reorder mask IDs to have 1 to N IDs
    N = number of ROIs
    Need to run this after filtering dendrites and border ROIs (for convenience)

    Parameters:
    -----------
    mask : np.array
        ROI mask (2D from CellPose)

    Returns:
    -----------
    mask_reordered : np.array
        reordered mask
    """
    mask_reordered = np.zeros_like(mask)
    roi_inds = np.setdiff1d(np.unique(mask), 0)
    for i, ind in enumerate(roi_inds):
        mask_reordered[mask == ind] = i + 1
    return mask_reordered


def get_bounding_box(masks: np.array, area_extension_factor: int = 2) -> np.array:
    """Get bounding box of ROI masks

    Parameters:
    -----------
    masks : np.array
        ROI masks (2D, from CellPose)
    area_extension_factor : float, optional
        factor to extend the bounding box, by default 2
        Roughly the area of the bounding box will be larger than that of the ROI by this factor
        Assuming circular ROI.

    Returns:
    -----------
    bb_masks : np.array
        bounding box masks (3D, allowing overlaps)
    """

    bb_extension = np.sqrt(area_extension_factor * np.pi / 4)
    mask_inds = np.setdiff1d(np.unique(masks), 0)

    bb_masks = np.zeros((len(mask_inds), *masks.shape), dtype=np.uint16)
    for i, mask_i in enumerate(mask_inds):
        y, x = np.where(masks == mask_i)
        bb_y_tight = [y.min(), y.max()]
        bb_x_tight = [x.min(), x.max()]
        bb_y_tight_len = bb_y_tight[1] - bb_y_tight[0]
        bb_x_tight_len = bb_x_tight[1] - bb_x_tight[0]
        bb_y = [
            max(0, int(np.round(bb_y_tight[0] - bb_extension * bb_y_tight_len / 2))),
            min(
                masks.shape[0],
                int(np.round(bb_y_tight[1] + bb_extension * bb_y_tight_len / 2)),
            ),
        ]
        bb_x = [
            max(0, int(np.round(bb_x_tight[0] - bb_extension * bb_x_tight_len / 2))),
            min(
                masks.shape[1],
                int(np.round(bb_x_tight[1] + bb_extension * bb_x_tight_len / 2)),
            ),
        ]
        bb_masks[i, bb_y[0] : bb_y[1], bb_x[0] : bb_x[1]] = mask_i
    return bb_masks


def apply_mixing_matrix(
    alpha: float, beta: float, signal_mean: np.array, paired_mean: np.array
) -> Tuple[np.array, np.array]:
    """Apply mixing matrix to the mean images to get reconstructed images

    Parameters:
    -----------
    alpha : float
        alpha value of the unmixing matrix
    beta : float
        beta value of the unmixing matrix
    signal_mean : np.array
        mean image of the signal plane
    paired_mean : np.array
        mean image of the paired plane

    Returns:
    -----------
    recon_signal : np.array
        reconstructed signal image
    recon_paired : np.array
        reconstructed paired image
    """
    mixing_mat = [[1 - alpha, beta], [alpha, 1 - beta]]
    unmixing_mat = np.linalg.inv(mixing_mat)
    raw_data = np.vstack([signal_mean.ravel(), paired_mean.ravel()])
    recon_data = np.dot(unmixing_mat, raw_data)
    recon_signal = recon_data[0, :].reshape(signal_mean.shape)
    recon_paired = recon_data[1, :].reshape(paired_mean.shape)
    return recon_signal, recon_paired


def draw_masks_on_image(
    img: np.array,
    masks: np.array,
    ax: matplotlib.axes.Axes = None,
    color: str = "r",
    linewidth: int = 1,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Draw masks on image

    Parameters:
    -----------
    img : np.array
        image
    masks : np.array
        masks (2D)
    ax : matplotlib.axes.Axes, optional
        axes to draw on, by default None
    color : str, optional
        color of the contour, by default 'r'
    linewidth : int, optional
        linewidth of the contour, by default 1

    Returns: Only if ax was not provided
    -----------
    fig : matplotlib.figure.Figure
        figure
    ax : matplotlib.axes.Axes
        axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray")

    num_roi = np.max(masks)
    for i in range(1, num_roi + 1):
        ax.contour(masks == i, colors=color, linewidths=linewidth)
    ax.axis("off")
    if "fig" in locals():
        return fig, ax
