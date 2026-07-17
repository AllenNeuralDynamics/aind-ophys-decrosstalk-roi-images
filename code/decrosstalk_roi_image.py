import json
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from scipy import ndimage
from skimage import filters, measure


def get_epoch_start_frames(
    data_length: int, max_num_epochs: int = 10, num_frames: int = 1000
) -> Tuple[list, int]:
    """Epoch start frames and per-epoch frame-window size, for a raw movie of
    `data_length` frames.

    Same epoch/frame-window convention as
    ``paired_plane_registration.episodic_mean_fov`` (duplicated intentionally rather than
    imported, to avoid a cross-module coupling -- keep this formula byte-for-byte identical
    to that function's inline logic).

    Parameters
    ----------
    data_length : int
        number of frames in the raw movie (equivalently, the row count of the
        ``{oeid}_motion_transform.csv`` motion-transform file, one row per raw frame)
    max_num_epochs : int, optional
        maximum number of epochs, by default 10
    num_frames : int, optional
        number of frames to average per epoch, by default 1000

    Returns
    -------
    start_frames : list
        start frame index (into the raw movie) of each epoch
    num_frames_actual : int
        number of frames actually averaged per epoch (may be less than `num_frames` if
        the epoch interval is smaller)
    """
    num_epochs = min(max_num_epochs, data_length // num_frames)
    epoch_interval = data_length // (num_epochs + 1)
    num_frames_actual = min(num_frames, epoch_interval)
    start_frames = [
        num_frames_actual // 2 + i * epoch_interval for i in range(num_epochs)
    ]
    return start_frames, num_frames_actual


def compute_mean_rigid_shift(
    motion_df: pd.DataFrame, start_frame: int, num_frames: int
) -> Tuple[float, float]:
    """Mean (y, x) rigid shift over motion_df.iloc[start_frame:start_frame+num_frames].

    Parameters
    ----------
    motion_df : pd.DataFrame
        motion-transform dataframe with columns "y" and "x" (per-frame rigid shift)
    start_frame : int
        start frame index of the window
    num_frames : int
        number of frames in the window

    Returns
    -------
    dy, dx : float
        mean rigid shift (y, x) over the frame window
    """
    window = motion_df.iloc[start_frame : start_frame + num_frames]
    return float(window["y"].mean()), float(window["x"].mean())


def shift_mask(mask: np.array, dy: float, dx: float) -> np.array:
    """Shift a labeled 2D mask by (dy, dx) pixels (rounded to nearest int).

    Implemented via ``np.roll``, then zeroing out the wrapped-around border strip that the
    roll introduces -- content must not wrap from one edge of the image to the opposite
    edge.

    Parameters
    ----------
    mask : np.array
        2D labeled mask
    dy, dx : float
        shift amount in pixels along (row, column); rounded to the nearest int before
        shifting

    Returns
    -------
    np.array
        shifted mask, same shape/dtype as `mask`
    """
    dy_int = int(np.round(dy))
    dx_int = int(np.round(dx))
    shifted = np.roll(mask, (dy_int, dx_int), axis=(0, 1))
    if dy_int > 0:
        shifted[:dy_int, :] = 0
    elif dy_int < 0:
        shifted[dy_int:, :] = 0
    if dx_int > 0:
        shifted[:, :dx_int] = 0
    elif dx_int < 0:
        shifted[:, dx_int:] = 0
    return shifted


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
    grid_interval_fine: float = 0.01,
    grid_interval_coarse: float = 0.04,
    coef_max: float = 0.36,
    dendrite_diameter_um: float = 4,
    max_diameter_um: float = 20,
    num_top_rois: int = 10,
    return_recon: float = False,
) -> Tuple[np.array, list, list, list, list, list, np.array, np.array]:
    """Get alpha and beta values for an experiment based on
    the mutual information of the ROI images from motion corrected episodic mean FOV images

    Segmentation is done ONCE per plane, from a high-SNR image averaged across all epochs
    -- not re-segmented from each epoch's own (noisier) mean image. A single global Otsu
    threshold on one noisy epoch's image can fail on images with heterogeneous cell
    brightness, yielding pathologically few ROI boxes (empirically verified: as few as 0-3
    boxes on some epochs under the old per-epoch-segmentation approach).

    Only the paired plane's ROI boxes are re-derived per epoch (by rigid-shifting the
    reference mask -- see `shift_mask`), because they must line up with that epoch's
    `registered_to_pair` data, whose registration differs epoch-to-epoch. The signal
    plane's boxes are reused unchanged across all epochs.

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
    grid_interval_fine : float, optional
        fine grid step of the coarse-to-fine search, by default 0.01
    grid_interval_coarse : float, optional
        coarse grid step of the coarse-to-fine search, by default 0.04
    coef_max : float, optional
        maximum value of alpha and beta, by default 0.36
    dendrite_diameter_um : float, optional
        lower-bound ROI diameter in um (see `get_signal_paired_top_masks`), by default 4
    max_diameter_um : float, optional
        upper-bound ROI diameter in um (see `get_signal_paired_top_masks`), by default 20
    num_top_rois : int, optional
        number of top-intensity ROIs to keep per plane, by default 10
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
    signal_bboxes_list : list
        per-epoch list of signal-plane ROI bounding boxes (see ``_bbox_coords``) -- the
        SAME boxes every epoch (segmented once, session-wide)
    paired_bboxes_list : list
        per-epoch list of paired-plane ROI bounding boxes (see ``_bbox_coords``) -- shifted
        per epoch to track that epoch's `registered_to_pair` registration
    example_signal_mean : np.array
        cropped signal-plane mean image from epoch 0 (used for the MI grid search)
    example_paired_mean : np.array
        cropped paired-plane mean image from epoch 0 (used for the MI grid search)
    """

    signal_fn = (
        Path("../results")
        / oeid
        / "decrosstalk"
        / f"{oeid}_registered_episodic_mean_fov.h5"
    )
    with h5py.File(signal_fn, "r") as f:
        data_length = f["data"].shape[0]
        signal_data = f["data"][()]

    paired_oeid = paired_reg_fn.parent.parent.name

    # Paired plane's own self-registered EMF (all epochs) -- segmented once, in the paired
    # plane's OWN frame; per-epoch boxes are then obtained by shifting this reference mask
    # (see below), not by re-segmenting the noisier, epoch-specific registered_to_pair data.
    paired_self_fn = (
        paired_reg_fn.parent / f"{paired_oeid}_registered_episodic_mean_fov.h5"
    )
    with h5py.File(paired_self_fn, "r") as f:
        paired_self_data = f["data"][()]

    motion_buffer = 5
    p1y, p1x = get_motion_correction_crop_xy_range_from_both_planes(
        oeid, paired_oeid, input_dir
    )
    signal_data_cropped = signal_data[
        :,
        p1y[0] + motion_buffer : p1y[1] - motion_buffer,
        p1x[0] + motion_buffer : p1x[1] - motion_buffer,
    ]
    paired_self_data_cropped = paired_self_data[
        :,
        p1y[0] + motion_buffer : p1y[1] - motion_buffer,
        p1x[0] + motion_buffer : p1x[1] - motion_buffer,
    ]
    signal_mean_avg = signal_data_cropped.mean(axis=0)
    paired_self_mean_avg = paired_self_data_cropped.mean(axis=0)

    # Segment ONCE per plane from the epoch-averaged (high-SNR) image. Each plane's
    # reference boxes come from its OWN independent call (that plane in the "signal"
    # position) rather than the "paired" side-output of the other plane's call: the
    # cross-plane overlap dedup in get_signal_paired_top_masks breaks rank ties in favor
    # of whichever image is passed as "paired", so the same plane's boxes could otherwise
    # subtly differ depending on which direction (this plane's own estimate, or its
    # partner's) referenced it. Calling it once per plane with that plane always in the
    # "signal" slot guarantees identical results either way.
    # `paired_top_masks_ref` is in the PAIRED plane's own self-registered frame -- it is
    # shifted into each epoch's registered_to_pair frame inside the loop below.
    # exclude_edge_touching_bbox=True here (only here -- the full-session-mean detection
    # step, per the "not from epochs" scoping) drops edge-touching ROIs BEFORE cross-plane
    # dedup and top-N selection happen inside get_signal_paired_top_masks, so a box lost to
    # the edge can still be backfilled by the next-best candidate instead of silently
    # under-filling num_top_rois.
    signal_top_masks_ref, _ = get_signal_paired_top_masks(
        signal_mean_avg,
        paired_self_mean_avg,
        dendrite_diameter_um=dendrite_diameter_um,
        max_diameter_um=max_diameter_um,
        pix_size=pixel_size,
        num_top_rois=num_top_rois,
        exclude_edge_touching_bbox=True,
    )
    paired_top_masks_ref, _ = get_signal_paired_top_masks(
        paired_self_mean_avg,
        signal_mean_avg,
        dendrite_diameter_um=dendrite_diameter_um,
        max_diameter_um=max_diameter_um,
        pix_size=pixel_size,
        num_top_rois=num_top_rois,
        exclude_edge_touching_bbox=True,
    )
    # Signal-plane boxes: reused for every epoch, never recomputed or shifted.
    signal_bb_masks_ref = get_bounding_box(signal_top_masks_ref, pix_size=pixel_size)

    # Motion-transform CSVs, loaded once, used to compute the per-epoch rigid-shift delta
    # that maps paired_top_masks_ref into that epoch's registered_to_pair frame.
    oeid_motion_df = pd.read_csv(
        input_dir / oeid / "motion_correction" / f"{oeid}_motion_transform.csv",
        usecols=["y", "x"],
    )
    paired_motion_df = pd.read_csv(
        input_dir
        / paired_oeid
        / "motion_correction"
        / f"{paired_oeid}_motion_transform.csv",
        usecols=["y", "x"],
    )
    assert len(oeid_motion_df) == len(paired_motion_df), (
        f"Motion-transform row counts differ between {oeid} ({len(oeid_motion_df)} rows) "
        f"and paired plane {paired_oeid} ({len(paired_motion_df)} rows) -- these two planes "
        "are assumed to be imaged simultaneously (same number of raw frames)."
    )

    start_frames, num_frames_actual = get_epoch_start_frames(len(oeid_motion_df))
    assert len(start_frames) == signal_data.shape[0], (
        f"Epoch count mismatch for {oeid}: get_epoch_start_frames computed "
        f"{len(start_frames)} epochs from {len(oeid_motion_df)} raw frames (motion "
        f"transform csv), but the cached episodic-mean-FOV file {signal_fn} has "
        f"{signal_data.shape[0]} epochs. This likely means the max_num_epochs/num_frames "
        "defaults used to generate that file differ from get_epoch_start_frames' defaults."
    )
    # paired_reg_fn (registered_to_pair) is the file actually indexed by epoch_idx inside
    # decrosstalk_roi_image_single_pair_from_episodic_mean_fov -- check its epoch count
    # too, not just signal_fn's, so a mismatch there (e.g. debug-mode truncating one file's
    # epochs but not the other's) is caught loudly instead of silently reading an
    # out-of-range (empty -> NaN) slice.
    with h5py.File(paired_reg_fn, "r") as f:
        paired_reg_fn_num_epochs = f["data"].shape[0]
    assert len(start_frames) == paired_reg_fn_num_epochs, (
        f"Epoch count mismatch for paired plane {paired_oeid}: get_epoch_start_frames "
        f"computed {len(start_frames)} epochs, but {paired_reg_fn} has "
        f"{paired_reg_fn_num_epochs} epochs."
    )

    alpha_list = []
    beta_list = []
    mean_norm_mi_list = []
    signal_bboxes_list = []
    paired_bboxes_list = []
    example_signal_mean = None
    example_paired_mean = None
    for epoch_idx, start_frame in enumerate(start_frames):
        dy_R, dx_R = compute_mean_rigid_shift(
            oeid_motion_df, start_frame, num_frames_actual
        )
        dy_P, dx_P = compute_mean_rigid_shift(
            paired_motion_df, start_frame, num_frames_actual
        )
        delta_y, delta_x = dy_P - dy_R, dx_P - dx_R
        paired_top_masks_epoch = shift_mask(paired_top_masks_ref, delta_y, delta_x)
        paired_bb_masks_epoch = get_bounding_box(paired_top_masks_epoch, pix_size=pixel_size)
        # A box near the crop edge in the reference frame can shift entirely into the
        # zeroed border strip introduced by shift_mask for some epochs, leaving it with
        # zero pixels. Left in, this poisons the MI grid search: normalized_mutual_
        # information on an empty selection is NaN, every (alpha, beta) grid point becomes
        # NaN, and argmin over an all-NaN array silently degenerates to index 0 (alpha=
        # beta=0.0) -- empirically observed and root-caused this way. Drop any box that
        # came out empty rather than propagate this.
        non_empty = [
            i for i in range(paired_bb_masks_epoch.shape[0])
            if paired_bb_masks_epoch[i].sum() > 0
        ]
        if len(non_empty) < paired_bb_masks_epoch.shape[0]:
            paired_bb_masks_epoch = paired_bb_masks_epoch[non_empty]

        (
            alpha,
            beta,
            mean_norm_mi_values,
            signal_mean,
            paired_mean,
            signal_bboxes,
            paired_bboxes,
        ) = decrosstalk_roi_image_single_pair_from_episodic_mean_fov(
            oeid,
            paired_reg_fn,
            input_dir,
            epoch_idx,
            pixel_size,
            grid_interval_fine=grid_interval_fine,
            grid_interval_coarse=grid_interval_coarse,
            coef_max=coef_max,
            signal_bb_masks=signal_bb_masks_ref,
            paired_bb_masks=paired_bb_masks_epoch,
        )
        alpha_list.append(alpha)
        beta_list.append(beta)
        mean_norm_mi_list.append(mean_norm_mi_values)
        # signal_bboxes / paired_bboxes returned above already equal
        # _bbox_coords(signal_bb_masks_ref) / _bbox_coords(paired_bb_masks_epoch), since
        # those exact bb_masks were passed in and no re-segmentation happened.
        signal_bboxes_list.append(signal_bboxes)
        paired_bboxes_list.append(paired_bboxes)
        if epoch_idx == 0:
            example_signal_mean = signal_mean
            example_paired_mean = paired_mean

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
    return (
        recon_signal_data,
        alpha_list,
        beta_list,
        mean_norm_mi_list,
        signal_bboxes_list,
        paired_bboxes_list,
        example_signal_mean,
        example_paired_mean,
    )


def _mean_norm_mi(alpha, beta, data, shape, bb_yx_list, mi_raw):
    """Mean (over ROI boxes) normalized MI between unmixed signal/paired, for one
    (alpha, beta). Uses precomputed box pixel indices `bb_yx_list` and per-box raw MI
    `mi_raw` for normalization."""
    temp_unmixing = np.linalg.inv([[1 - alpha, beta], [alpha, 1 - beta]])
    rec = np.dot(temp_unmixing, data)
    rs = rec[0, :].reshape(shape)
    rp = rec[1, :].reshape(shape)
    temp_mi = np.array(
        [skimage.metrics.normalized_mutual_information(rs[yx], rp[yx]) for yx in bb_yx_list]
    )
    return float((temp_mi / mi_raw).mean())


def coarse_to_fine_grid_search(
    signal_mean,
    paired_mean,
    bb_masks,
    grid_interval_coarse: float = 0.04,
    coef_max: float = 0.36,
    grid_interval_fine: float = 0.01,
    fine_window: float = 0.05,
):
    """Find (alpha, beta) minimizing mean normalized MI across ROI boxes.

    Two passes: a coarse grid (step `grid_interval_coarse`, 0..coef_max) locates the
    basin, then a fine grid (step `grid_interval_fine`, +/-`fine_window` around the
    coarse minimum, clipped to [0, coef_max]) refines it. ~5x fewer evaluations than the
    full grid. Verified to reproduce the full-resolution grid argmin within one grid step
    on good/over/under/flat cases (session02 verify gate); index-caching (precomputed
    `bb_yx_list`) is exact.

    Returns (alpha, beta, grid_vals). `grid_vals` is a flattened (n x n) grid over
    [0, coef_max] at `grid_interval_fine` (n = coef_max/grid_interval_fine + 1), filled at
    the evaluated coarse AND fine (alpha, beta) points and NaN elsewhere -- so it retains
    the fine (grid_interval_fine) resolution around the minimum plus the coarse wide
    landscape, with implicit regular-grid coordinates, without computing the full grid. NB
    shape differs from the old full grid (now 37x37 over 0-0.36, sparse/NaN); reshape and
    use nan-aware ops (e.g. np.nanargmin) downstream.
    """
    bb_yx_list = [np.where(mask) for mask in bb_masks]
    mi_raw = np.array(
        [
            skimage.metrics.normalized_mutual_information(signal_mean[yx], paired_mean[yx])
            for yx in bb_yx_list
        ]
    )
    data = np.vstack((signal_mean.ravel(), paired_mean.ravel()))
    shape = signal_mean.shape

    def _grid(lo_a, hi_a, lo_b, hi_b, step):
        av = np.arange(lo_a, hi_a + step, step)
        av = av[av <= coef_max + 1e-9]
        bv = np.arange(lo_b, hi_b + step, step)
        bv = bv[bv <= coef_max + 1e-9]
        vals, ab = [], []
        for a in av:
            for b in bv:
                vals.append(_mean_norm_mi(a, b, data, shape, bb_yx_list, mi_raw))
                ab.append([float(a), float(b)])
        return vals, ab

    coarse_vals, coarse_ab = _grid(0, coef_max, 0, coef_max, grid_interval_coarse)
    a0, b0 = coarse_ab[int(np.argmin(coarse_vals))]
    fine_vals, fine_ab = _grid(
        max(0, a0 - fine_window), min(coef_max, a0 + fine_window),
        max(0, b0 - fine_window), min(coef_max, b0 + fine_window),
        grid_interval_fine,
    )
    alpha, beta = fine_ab[int(np.argmin(fine_vals))]

    # Assemble a sparse full-resolution landscape: an (n x n) grid over [0, coef_max]
    # at `grid_interval_fine`, filled at the evaluated coarse AND fine (alpha, beta) points
    # and NaN elsewhere. Keeps implicit regular-grid coordinates while retaining fine
    # (0.01) resolution around the minimum plus the coarse wide landscape.
    n = int(round(coef_max / grid_interval_fine)) + 1
    grid_vals = np.full((n, n), np.nan)
    for (a, b), v in list(zip(coarse_ab, coarse_vals)) + list(zip(fine_ab, fine_vals)):
        grid_vals[int(round(a / grid_interval_fine)), int(round(b / grid_interval_fine))] = v
    return alpha, beta, grid_vals.ravel()


_LQ_KEYS = ("lam_min", "lam_max", "a_star", "b_star", "se_a", "se_b", "sigma", "depth", "snr")


def _fit_basin_quadratic(av, bv, zv, depth):
    """Fit z ~ c0 + c1 a + c2 b + c3 a^2 + c4 b^2 + c5 ab and return the landscape-quality
    metric dict. `depth` (global basin depth) is passed in for cross-region consistency.
    np.nan dict if degenerate / too few points."""
    if len(zv) < 6:
        return {k: np.nan for k in _LQ_KEYS}
    X = np.column_stack([np.ones_like(av), av, bv, av ** 2, bv ** 2, av * bv])
    coef, *_ = np.linalg.lstsq(X, zv, rcond=None)
    resid = zv - X @ coef
    sigma = float(np.sqrt((resid ** 2).sum() / max(len(zv) - 6, 1)))
    _, c1, c2, c3, c4, c5 = coef
    evals = np.linalg.eigvalsh(np.array([[2 * c3, c5], [c5, 2 * c4]]))
    lam_min, lam_max = float(evals[0]), float(evals[1])

    def _vertex(c):
        cc1, cc2, cc3, cc4, cc5 = c
        Hm = np.array([[2 * cc3, cc5], [cc5, 2 * cc4]])
        try:
            return -np.linalg.solve(Hm, np.array([cc1, cc2]))
        except np.linalg.LinAlgError:
            return np.array([np.nan, np.nan])

    v = _vertex(coef[1:])
    a_star, b_star = float(v[0]), float(v[1])
    try:
        cov5 = (sigma ** 2 * np.linalg.inv(X.T @ X))[1:, 1:]
        eps, J, base = 1e-6, np.zeros((2, 5)), coef[1:].copy()
        for k in range(5):
            cp = base.copy(); cp[k] += eps
            J[:, k] = (_vertex(cp) - _vertex(base)) / eps
        cov_v = J @ cov5 @ J.T
        se_a, se_b = float(np.sqrt(max(cov_v[0, 0], 0))), float(np.sqrt(max(cov_v[1, 1], 0)))
    except np.linalg.LinAlgError:
        se_a = se_b = np.nan
    snr = depth / sigma if sigma > 0 else np.nan
    return dict(lam_min=lam_min, lam_max=lam_max, a_star=a_star, b_star=b_star,
                se_a=se_a, se_b=se_b, sigma=sigma, depth=depth, snr=snr)


def landscape_quality(mean_norm_mi_values, region="coarse", grid_interval_fine=0.01,
                      grid_interval_coarse=0.04, fine_window=0.05):
    """Curvature / flatness / SNR of one epoch's MI objective basin, from a 2D quadratic
    fit to either the COARSE or the FINE grid points.

    `mean_norm_mi_values` is one epoch's flattened (n*n) objective grid (as stored in
    mean_norm_mi_list), reshaped to (n, n) over [0, (n-1)*grid_interval_fine]. Fits
        z ~ c0 + c1 a + c2 b + c3 a^2 + c4 b^2 + c5 a b
    on the selected region:
      region="coarse": coarse lattice (every grid_interval_coarse/grid_interval_fine-th
                       point) -> GLOBAL
      region="fine"  : points within +/-fine_window of the grid argmin (dense 0.01 block;
                       the stored fine block in the sparse format) -> LOCAL basin
    Returns metric VALUES only (lam_min/lam_max curvature, a_star/b_star vertex, se_a/se_b
    propagated vertex SE, sigma fit residual, depth = 1-nanmin, snr = depth/sigma).
    np.nan when degenerate. Kept identical to decrosstalk_qc.metrics.landscape_quality.
    """
    flat = np.asarray(mean_norm_mi_values, dtype=float).ravel()
    n = int(round(len(flat) ** 0.5))
    if n * n != len(flat):
        return {k: np.nan for k in _LQ_KEYS}
    G = flat.reshape(n, n)
    ax_full = np.arange(n) * grid_interval_fine
    depth = 1.0 - float(np.nanmin(G))
    if region == "coarse":
        step = max(int(round(grid_interval_coarse / grid_interval_fine)), 1)
        idx = np.arange(0, n, step)
        A, B = np.meshgrid(ax_full[idx], ax_full[idx], indexing="ij")
        Z = G[np.ix_(idx, idx)]
    elif region == "fine":
        i0, j0 = np.unravel_index(int(np.nanargmin(G)), G.shape)
        w = max(int(round(fine_window / grid_interval_fine)), 1)
        ii = np.arange(max(0, i0 - w), min(n, i0 + w + 1))
        jj = np.arange(max(0, j0 - w), min(n, j0 + w + 1))
        A, B = np.meshgrid(ax_full[ii], ax_full[jj], indexing="ij")
        Z = G[np.ix_(ii, jj)]
    else:
        raise ValueError(f"region must be 'coarse' or 'fine', got {region!r}")
    m = np.isfinite(Z)
    if int(m.sum()) < 6:
        return {k: np.nan for k in _LQ_KEYS}
    return _fit_basin_quadratic(A[m], B[m], Z[m], depth)


def mean_landscape_quality(mean_norm_mi_list, grid_interval_fine=0.01,
                           grid_interval_coarse=0.04, fine_window=0.05):
    """Epoch-mean of landscape_quality over all epochs, for BOTH the coarse and fine fits.
    Returns keys suffixed '_coarse' / '_fine' (nan-safe). Metric values only (no decision)."""
    out = {}
    for region in ("coarse", "fine"):
        per = [landscape_quality(g, region=region, grid_interval_fine=grid_interval_fine,
                                 grid_interval_coarse=grid_interval_coarse,
                                 fine_window=fine_window)
               for g in mean_norm_mi_list]
        for k in _LQ_KEYS:
            vals = np.array([p[k] for p in per], dtype=float)
            out[f"{k}_{region}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return out


COEF_MAX = 0.36     # fixed coefficient-value axis (grid max) -> figures comparable across sessions
RECIP_FLAG = 0.05   # reciprocity-gap flag line (above the observed max ~0.043; tunable)


def render_landscape_page(mean_norm_mi_list, alpha_list, beta_list, grid_interval_fine=0.01,
                          title="", applied=None, partner=None, coef_max=COEF_MAX,
                          recip_flag=RECIP_FLAG, save=None):
    """One-page landscape QC figure: per-epoch MI landscapes + coefficient stability + pair
    symmetry (reciprocity).

    Design (session02 bad_plane_analysis): landscape-shape metrics (curvature/flatness/SNR)
    do NOT discriminate good vs bad and were dropped from the figure. This lean view catches
    DRASTIC estimation failures only: a wandering/multi-modal minimum (landscapes), an
    unstable coefficient (α/β stability), and a one-plane segmentation/registration
    breakdown (pair symmetry: α_p≈β_q, β_p≈α_q).

    `mean_norm_mi_list` = per-epoch flattened (n*n) objective grids (as stored in the h5).
    `applied` = (alpha, beta) actually applied (reciprocity-averaged). `partner` =
    (alpha_q_list, beta_q_list) of the partner plane; when given, adds the pair-symmetry row.
    Coefficient-value axes are square and fixed to [0, coef_max]. Kept identical to
    decrosstalk_qc.plots.render_landscape_page (ASCII labels for headless-font safety).
    """
    import math
    if save is not None:
        plt.switch_backend("Agg")

    grids = []
    for g in mean_norm_mi_list:
        flat = np.asarray(g, dtype=float).ravel()
        nn = int(round(len(flat) ** 0.5))
        grids.append(flat.reshape(nn, nn))
    grid = np.stack(grids)
    alpha, beta = np.asarray(alpha_list, float), np.asarray(beta_list, float)
    n, N = grid.shape[0], grid.shape[1]
    gm = (N - 1) * grid_interval_fine
    ep = np.arange(n)
    finite = grid[np.isfinite(grid)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    ncols = math.ceil(n / 2)
    has_partner = partner is not None
    if has_partner:
        aq, bq = np.asarray(partner[0], float), np.asarray(partner[1], float)
        E = min(n, len(aq))

    bot_rows = 2 if has_partner else 1
    fig = plt.figure(figsize=(1.6 * ncols + 2.4, 4.0 + 2.5 * bot_rows), layout="constrained")
    fig.suptitle(title, fontsize=11)
    sf_land, sf_bot = fig.subfigures(2, 1, height_ratios=[1.5, 1.3 * bot_rows])

    # --- per-epoch landscapes: 2 rows x ncols, square panels ---
    axl = sf_land.subplots(2, ncols, squeeze=False)
    sf_land.suptitle("per-epoch MI landscapes (min = r+)", fontsize=9)
    im = None
    for e in range(2 * ncols):
        ax = axl[e // ncols][e % ncols]
        if e < n:
            im = ax.imshow(grid[e].T, origin="lower", extent=[0, gm, 0, gm], vmin=vmin,
                           vmax=vmax, cmap="viridis", aspect="auto")
            if np.isfinite(grid[e]).any():
                ai, bi = np.unravel_index(int(np.nanargmin(grid[e])), grid[e].shape)
                ax.plot(ai * grid_interval_fine, bi * grid_interval_fine, "r+", ms=7)
            ax.set_box_aspect(1)
            ax.set_title(f"ep{e}", fontsize=7); ax.tick_params(labelsize=5)
            if e % ncols == 0: ax.set_ylabel("beta", fontsize=7)
            if e // ncols == 1 or n <= ncols: ax.set_xlabel("alpha", fontsize=7)
        else:
            ax.axis("off")
    if im is not None:
        sf_land.colorbar(im, ax=axl, shrink=0.7, pad=0.01, label="norm. MI (basin=low)")

    axb = sf_bot.subplots(bot_rows, 2, squeeze=False)

    # (0,0) argmin stability -- SQUARE, fixed axes, legend outside right
    ax = axb[0][0]
    ax.scatter(alpha, beta, c=ep, cmap="plasma", s=38, edgecolor="k", linewidth=0.3, zorder=3)
    ax.scatter(alpha.mean(), beta.mean(), marker="*", s=180, c="lime", edgecolor="k",
               zorder=4, label="epoch mean")
    if applied is not None:
        ax.scatter([applied[0]], [applied[1]], marker="X", s=110, c="red", edgecolor="k",
                   zorder=5, label="applied")
    ax.set_xlim(0, coef_max); ax.set_ylim(0, coef_max); ax.set_box_aspect(1)
    ax.set_xlabel("alpha*", fontsize=8); ax.set_ylabel("beta*", fontsize=8)
    ax.set_title(f"argmin stability (sd_a={alpha.std():.3f}, sd_b={beta.std():.3f})", fontsize=8)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # (0,1) alpha/beta vs epoch -- y fixed to [0, coef_max], legend outside right
    ax = axb[0][1]
    ax.plot(ep, alpha, "o-", label="alpha*"); ax.plot(ep, beta, "s-", label="beta*")
    if applied is not None:
        ax.axhline(applied[0], color="C0", ls="--", lw=1, alpha=0.6)
        ax.axhline(applied[1], color="C1", ls="--", lw=1, alpha=0.6)
    ax.set_ylim(0, coef_max); ax.set_xlabel("epoch", fontsize=8); ax.set_ylabel("coefficient", fontsize=8)
    ax.set_title("alpha / beta vs epoch", fontsize=8); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    if has_partner:
        # (1,0) pair symmetry scatter -- SQUARE: this plane (a_p,b_p) vs partner flipped (b_q,a_q)
        ax = axb[1][0]
        ax.scatter(alpha[:E], beta[:E], s=42, facecolor="none", edgecolor="C0", linewidth=1.4,
                   label="this plane (a_p,b_p)")
        ax.scatter(bq[:E], aq[:E], marker="x", s=42, c="C3", label="partner flipped (b_q,a_q)")
        ax.plot([0, coef_max], [0, coef_max], "k:", linewidth=0.6)
        ax.set_xlim(0, coef_max); ax.set_ylim(0, coef_max); ax.set_box_aspect(1)
        ax.set_xlabel("a_p  (leak p->q)", fontsize=8); ax.set_ylabel("b_p  (leak q->p)", fontsize=8)
        gap = 0.5 * (np.mean(np.abs(alpha[:E] - bq[:E])) + np.mean(np.abs(beta[:E] - aq[:E])))
        ax.set_title(f"pair symmetry (recip gap={gap:.3f}, {'OK' if gap < recip_flag else 'FLAG'})", fontsize=8)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

        # (1,1) reciprocity gap vs epoch, legend outside right
        ax = axb[1][1]
        ax.plot(ep[:E], np.abs(alpha[:E] - bq[:E]), "o-", label="|a_p - b_q|")
        ax.plot(ep[:E], np.abs(beta[:E] - aq[:E]), "s-", label="|b_p - a_q|")
        ax.axhline(recip_flag, color="r", ls="--", linewidth=1, label=f"{recip_flag:g} flag")
        ax.set_ylim(0, max(recip_flag * 1.6, float(np.abs(beta[:E] - aq[:E]).max()) * 1.15))
        ax.set_xlabel("epoch", fontsize=8); ax.set_ylabel("reciprocity gap", fontsize=8)
        ax.set_title("reciprocity gap vs epoch", fontsize=8); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    if save is not None:
        fig.savefig(save, dpi=100)
        plt.close(fig)
        return save
    return fig


def render_roi_bbox_page(signal_mean, paired_mean, signal_bboxes, paired_bboxes,
                         title="", save=None):
    """One-page QC figure: the two planes' cropped mean images (used for the MI grid
    search), each overlaid with BOTH planes' ROI bounding boxes (signal boxes in one
    color, paired-plane boxes in another) -- both images share the same pixel coordinate
    frame (paired_mean is already registered into signal_mean's frame), so either box set
    is valid to draw on either panel. Shows exactly which regions the estimator used.
    Coordinates are in the CROPPED image's own pixel frame (see save_qc_values).
    """
    import matplotlib
    if save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7))
    for ax, img, panel_title in [(axes[0], signal_mean, "signal"), (axes[1], paired_mean, "paired")]:
        lo, hi = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, origin="upper", aspect="equal")
        for boxes, color, label in [(signal_bboxes, "red", "signal ROI boxes"),
                                    (paired_bboxes, "cyan", "paired ROI boxes")]:
            for bi, b in enumerate(boxes):
                x, y = b["top_left"]
                rect = Rectangle((x, y), b["width"], b["height"], linewidth=1,
                                 edgecolor=color, facecolor="none",
                                 label=label if bi == 0 else None)
                ax.add_patch(rect)
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if save is not None:
        fig.savefig(save, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return save
    return fig


def _json_sanitize(x):
    """Recursively convert a (possibly nested) array-like to plain Python types that
    ``json.dump`` accepts: NumPy scalars/arrays -> native float/list, NaN -> null (JSON
    has no NaN token; None round-trips to NaN via e.g. ``float(v) if v is not None else
    float('nan')``)."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_json_sanitize(v) for v in np.asarray(x, dtype=float).tolist()]
    x = float(x)
    return None if np.isnan(x) else x


def save_qc_values(mean_norm_mi_list, alpha_list, beta_list, applied, oeid=None,
                   paired_oeid=None, partner=None, grid_interval_fine=0.01,
                   grid_interval_coarse=0.04, coef_max=COEF_MAX, recip_flag=RECIP_FLAG,
                   save=None, signal_bboxes_list=None, paired_bboxes_list=None):
    """Save the exact values needed to reproduce this plane's QC figure
    (:func:`render_landscape_page`) to a JSON file -- the plot's data, without the plot.

    Parameters mirror ``render_landscape_page``; nothing here is derived/recomputed, it is
    the same data passed to that function. NaN cells in ``mean_norm_mi_list`` (the sparse
    coarse-to-fine grid's unevaluated points) are written as ``null``. ``grid_interval_fine``
    is the resolution of the stored grid itself (needed to reshape/plot it);
    ``grid_interval_coarse`` is recorded too even though the figure doesn't need it, purely
    so the sparse coarse+fine structure (why most cells are null) is self-documenting from
    the JSON alone, rather than only inferable from the NaN pattern.

    signal_bboxes_list/paired_bboxes_list (if given): per-epoch list of ROI box dicts
    {top_left:[x,y], width, height} in the cropped-image pixel frame (same crop for every
    epoch of this oeid) -- lets the boxes be redrawn without re-running segmentation.

    Returns the dict written (or returned, if ``save`` is None).
    """
    qc = {
        "oeid": oeid,
        "paired_oeid": paired_oeid,
        "grid_interval_fine": float(grid_interval_fine),
        "grid_interval_coarse": float(grid_interval_coarse),
        "coef_max": float(coef_max),
        "recip_flag": float(recip_flag),
        "applied_alpha": float(applied[0]),
        "applied_beta": float(applied[1]),
        "alpha_list": _json_sanitize(alpha_list),
        "beta_list": _json_sanitize(beta_list),
        "mean_norm_mi_list": _json_sanitize(mean_norm_mi_list),
    }
    if partner is not None:
        qc["partner_alpha_list"] = _json_sanitize(partner[0])
        qc["partner_beta_list"] = _json_sanitize(partner[1])
    if signal_bboxes_list is not None:
        qc["signal_bboxes_list"] = signal_bboxes_list
    if paired_bboxes_list is not None:
        qc["paired_bboxes_list"] = paired_bboxes_list
    if save is not None:
        with open(save, "w") as f:
            json.dump(qc, f, indent=2)
        return save
    return qc


def decrosstalk_roi_image_single_pair_from_episodic_mean_fov(
    oeid: int,
    paired_reg_emf_fn: str,
    input_dir: Path,
    start_frame: int,
    pix_size: float,
    motion_buffer: int = 5,
    grid_interval_fine: float = 0.01,
    grid_interval_coarse: float = 0.04,
    coef_max: float = 0.36,
    signal_bb_masks: np.array = None,
    paired_bb_masks: np.array = None,
) -> Tuple[float, float, list, np.array, np.array, list, list]:
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
    grid_interval_fine : float, optional
        fine grid step of the coarse-to-fine search, by default 0.01
    grid_interval_coarse : float, optional
        coarse grid step of the coarse-to-fine search, by default 0.04
    coef_max : float, optional
        maximum value of alpha and beta, by default 0.36
    signal_bb_masks : np.array, optional
        pre-computed signal-plane ROI bounding-box masks (see `get_bounding_box`). If
        given (together with `paired_bb_masks`), segmentation is skipped entirely for this
        epoch and these masks are used directly -- the caller (
        `decrosstalk_roi_image_from_episodic_mean_fov`) segments once per plane and passes
        the (per-epoch-shifted, for the paired plane) result in. By default None, which
        preserves the original behavior: segment fresh from this epoch's own
        `signal_mean`/`paired_mean` images (used directly by external callers/scripts that
        exercise this function on its own).
    paired_bb_masks : np.array, optional
        pre-computed paired-plane ROI bounding-box masks; see `signal_bb_masks`.

    Returns:
    -----------
    alpha : float
        alpha value of the unmixing matrix
    beta : float
        beta value of the unmixing matrix
    mean_norm_mi_values : np.array
        mean normalized mutual information values
    signal_mean : np.array
        cropped signal-plane mean image used for the MI grid search
    paired_mean : np.array
        cropped paired-plane mean image used for the MI grid search
    signal_bboxes : list
        signal-plane ROI bounding boxes (see ``_bbox_coords``)
    paired_bboxes : list
        paired-plane ROI bounding boxes (see ``_bbox_coords``)
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

    assert (signal_bb_masks is None) == (paired_bb_masks is None), (
        "signal_bb_masks and paired_bb_masks must be given together or not at all -- "
        f"got signal_bb_masks={'None' if signal_bb_masks is None else 'given'}, "
        f"paired_bb_masks={'None' if paired_bb_masks is None else 'given'}."
    )
    if signal_bb_masks is None:
        # Backward-compat path: segment fresh from this epoch's own mean images (also
        # exercised directly by external test scripts) -- see get_signal_paired_top_masks.
        signal_top_masks, paired_top_masks = get_signal_paired_top_masks(
            signal_mean, paired_mean, pix_size=pix_size
        )
        signal_bb_masks = get_bounding_box(signal_top_masks, pix_size=pix_size)
        paired_bb_masks = get_bounding_box(paired_top_masks, pix_size=pix_size)

    signal_bboxes = _bbox_coords(signal_bb_masks)
    paired_bboxes = _bbox_coords(paired_bb_masks)
    bb_masks = np.concatenate([signal_bb_masks, paired_bb_masks])
    # Coarse-to-fine grid search for (alpha, beta) minimizing mean normalized MI across
    # ROI boxes (see coarse_to_fine_grid_search). Reproduces the full-resolution grid
    # argmin within one grid step (session02 verify gate) at ~5x fewer evaluations.
    # NB: mean_norm_mi_values is now a sparse 37x37 grid (0..coef_max at grid_interval_fine)
    # -- fine resolution near the minimum, coarse elsewhere, NaN at unevaluated points --
    # not the old dense 31x31 (0-0.30) grid; use nan-aware ops downstream.
    alpha, beta, mean_norm_mi_values = coarse_to_fine_grid_search(
        signal_mean, paired_mean, bb_masks,
        coef_max=coef_max, grid_interval_fine=grid_interval_fine,
        grid_interval_coarse=grid_interval_coarse,
    )
    return (
        alpha,
        beta,
        np.array(mean_norm_mi_values).tolist(),
        signal_mean,
        paired_mean,
        signal_bboxes,
        paired_bboxes,
    )


def basic_segmentation(
    mean_img: np.array,
    sigma_segmentation: int = 30,
) -> np.array:
    """Fast classical soma segmentation, replacing CellPose.

    Adapted from aind-ophys-movie-qc `get_and_plot_basic_segmentation`:
    Gaussian high-pass (remove neuropil/background) -> Otsu threshold ->
    connected components. Returns an integer-labeled mask (0=background,
    1..N=ROIs), matching the CellPose `model.eval` output consumed downstream.
    Size-based pruning is left entirely to `filter_dendrite`'s physical
    (diameter-based) criterion rather than an arbitrary pixel-count cutoff here.

    Validated to reproduce the CellPose-pipeline alpha/beta (esp. beta, the
    crosstalk-removal knob) within ~0.01-0.02; see session02 consistency check.
    Much faster (~0.1 s vs ~22 s per epoch) and drops the torch/cellpose dependency.
    """
    neuropil = ndimage.gaussian_filter(mean_img, sigma=sigma_segmentation)
    high_pass = mean_img - neuropil
    binary = high_pass > filters.threshold_otsu(high_pass)
    return measure.label(binary)


def segment_and_filter_with_relaxation(
    mean_img: np.array,
    dendrite_diameter_pix: float,
    max_diameter_pix: float,
    target_count: int,
    sigma_segmentation: int = 30,
    relaxation_factors: tuple = (1.0, 0.7, 0.5),
    exclude_edge_touching_bbox: bool = False,
    pix_size: float = 0.78,
) -> np.array:
    """Segment `mean_img` and progressively relax the Otsu threshold until at least
    `target_count` ROIs survive size (and, if requested, edge) filtering.

    A single global Otsu threshold on one image can fail to separate cells of
    heterogeneous brightness, yielding too few ROIs. Rather than re-picking a threshold
    (unstable), this computes the image's own high-pass image and base Otsu threshold
    ONCE, then tries `base_thresh * factor` for each `factor` in `relaxation_factors` (in
    order, e.g. 1.0 -> 0.7 -> 0.5) so dimmer ROIs are progressively included, stopping as
    soon as enough ROIs are found. Does not call `basic_segmentation` (kept standalone and
    unmodified for other callers); the high-pass computation is intentionally duplicated
    here.

    `exclude_edge_touching_bbox`, if set, drops edge-touching ROIs (see
    `filter_edge_touching_roi`) as part of EACH attempt's own filtering -- i.e. before
    `target_count` is checked and before any downstream cross-plane dedup / top-N
    selection happens on the result. Filtering edge-touching boxes only *after* top-N
    selection would silently under-fill the requested count (a selected box dropped late
    can't be backfilled by the next-best candidate); doing it here, as part of the
    candidate pool itself, also lets relaxation compensate for edge losses the same way it
    already compensates for a strict threshold.

    Parameters
    ----------
    mean_img : np.array
        mean image to segment
    dendrite_diameter_pix : float
        lower-bound diameter (pix); ROIs smaller than this (by area) are discarded as
        dendrite fragments (see `filter_dendrite`)
    max_diameter_pix : float
        upper-bound diameter (pix); ROIs larger than this (by area) are discarded as
        oversized blobs/artifacts (see `filter_oversized_roi`)
    target_count : int
        stop relaxing once the labeled mask has at least this many ROIs
    sigma_segmentation : int, optional
        Gaussian high-pass sigma, by default 30 (same as `basic_segmentation`)
    relaxation_factors : tuple, optional
        multipliers applied to the base Otsu threshold, tried in order, by default
        (1.0, 0.7, 0.5)
    exclude_edge_touching_bbox : bool, optional
        if True, also drop ROIs whose bounding box touches the image edge (see
        `filter_edge_touching_roi`), as part of each attempt, by default False
    pix_size : float, optional
        pixel size in um, only used if `exclude_edge_touching_bbox` is True, by default
        0.78

    Returns
    -------
    np.array
        reordered (1..N), filtered labeled mask from the first relaxation factor that
        reaches `target_count` ROIs, or the most-relaxed attempt's result if none do
    """
    neuropil = ndimage.gaussian_filter(mean_img, sigma=sigma_segmentation)
    high_pass = mean_img - neuropil
    base_thresh = filters.threshold_otsu(high_pass)

    result = None
    for factor in relaxation_factors:
        binary = high_pass > (base_thresh * factor)
        labeled = measure.label(binary)
        filtered = filter_dendrite(labeled, dendrite_diameter_pix=dendrite_diameter_pix)
        filtered = filter_oversized_roi(filtered, max_diameter_pix=max_diameter_pix)
        if exclude_edge_touching_bbox:
            filtered = filter_edge_touching_roi(filtered, pix_size=pix_size)
        result = reorder_mask(filtered)
        if int(result.max()) >= target_count:
            break
    return result


def get_signal_paired_top_masks(
    signal_mean: np.array,
    paired_mean: np.array,
    dendrite_diameter_um: float = 4,
    max_diameter_um: float = 20,
    pix_size: float = 0.78,
    overlap_threshold: float = 0.7,
    num_top_rois: int = 10,
    relaxation_factors: tuple = (1.0, 0.7, 0.5),
    exclude_edge_touching_bbox: bool = False,
) -> Tuple[np.array, np.array]:
    """Get top masks of 2 paired mean images.

    Each image is segmented with a dynamically-relaxed Otsu threshold (see
    `segment_and_filter_with_relaxation`): starting from the image's own Otsu threshold,
    progressively relax it (per `relaxation_factors`) until at least `num_top_rois` ROIs
    survive size filtering. This makes segmentation robust to heterogeneous cell
    brightness within a single (possibly noisy) image, instead of failing outright (too
    few ROIs) when one global threshold is too strict.

    ROIs are filtered on BOTH size bounds: too-small (`dendrite_diameter_um`, dendrite
    fragments, via `filter_dendrite`) and too-large (`max_diameter_um`, merged
    blobs/artifacts, via `filter_oversized_roi`). Then the top n intensity masks are kept
    from both planes.

    There can be duplicates due to excessive crosstalk:
    - Identify duplicate ROIs based on the overlap between the masks of the two planes
    - Remove the one with lower rank in intensity from all ROIs in the corresponding plane

    No separate border filter is applied here: `signal_mean`/`paired_mean` are already
    cropped to the motion-correction-safe region (plus a buffer) by the caller, so a
    ROI surviving into this function is already clear of the registration-rolling border.

    Parameters:
    -----------
    signal_mean : np.array
        mean image of the signal plane
    paired_mean : np.array
        mean image of the paired plane
    dendrite_diameter_um : float, optional
        lower-bound diameter of a valid ROI in um (below this, discarded as a dendrite
        fragment), by default 4
    max_diameter_um : float, optional
        upper-bound diameter of a valid ROI in um (above this, discarded as an oversized
        blob/artifact), by default 20
    pix_size : float, optional
        pixel size in um, by default 0.78
    overlap_threshold : float, optional
        threshold of overlap between signal and paired masks, by default 0.7
    num_top_rois : int, optional
        number of top ROIs to keep, and the target ROI count for the Otsu-relaxation
        search, by default 10
    relaxation_factors : tuple, optional
        multipliers applied (in order) to each image's own Otsu threshold until
        `num_top_rois` ROIs are found, by default (1.0, 0.7, 0.5)
    exclude_edge_touching_bbox : bool, optional
        if True, drop ROIs whose bounding box touches the image edge as part of each
        plane's own candidate-pool filtering, BEFORE cross-plane dedup and top-N
        selection (see `segment_and_filter_with_relaxation`) -- filtering this only after
        top-N selection would silently under-fill num_top_rois, since a selected box
        dropped late can't be backfilled by the next-best candidate. By default False.

    Returns:
    -----------
    signal_top_masks : np.array
        top masks of the signal plane
    paired_top_masks : np.array
        top masks of the paired plane
    """

    dendrite_diameter_px = dendrite_diameter_um / pix_size
    max_diameter_px = max_diameter_um / pix_size

    signal_masks_filtered = segment_and_filter_with_relaxation(
        signal_mean,
        dendrite_diameter_pix=dendrite_diameter_px,
        max_diameter_pix=max_diameter_px,
        target_count=num_top_rois,
        relaxation_factors=relaxation_factors,
        exclude_edge_touching_bbox=exclude_edge_touching_bbox,
        pix_size=pix_size,
    )
    paired_masks_filtered = segment_and_filter_with_relaxation(
        paired_mean,
        dendrite_diameter_pix=dendrite_diameter_px,
        max_diameter_pix=max_diameter_px,
        target_count=num_top_rois,
        relaxation_factors=relaxation_factors,
        exclude_edge_touching_bbox=exclude_edge_touching_bbox,
        pix_size=pix_size,
    )

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


def filter_oversized_roi(
    masks: np.array, max_diameter_pix: float = (20 / 0.78)
) -> np.array:
    """Filter oversized ROIs from masks based on area threshold

    Input Parameters
    ----------------
    masks: 2d array, each ROI has a unique integer value
    max_diameter_pix: float, maximum diameter of a valid ROI in pix

    Returns
    -------
    filtered_mask: 2d array, after filtering
        Note - the filtered mask is not necessarily contiguous
    """
    max_radius = max_diameter_pix / 2
    area_threshold = np.pi * max_radius**2
    num_roi = np.max(masks)
    filtered_mask = masks.copy()
    for roi_id in range(1, num_roi + 1):
        roi_mask = masks == roi_id
        roi_area = np.sum(roi_mask)
        if roi_area > area_threshold:
            filtered_mask[roi_mask] = 0
    return filtered_mask


def filter_edge_touching_roi(
    masks: np.array,
    area_extension_factor: float = 2,
    min_buffer_um: float = 5,
    pix_size: float = 0.78,
) -> np.array:
    """Remove ROIs whose BOUNDING BOX (see get_bounding_box -- the tight ROI extended by
    its buffer) touches any edge of the image (row 0, last row, col 0, last col).

    Checked on the box get_bounding_box would actually produce, not just the raw
    segmented ROI shape: a box can reach the edge purely because its buffer gets clamped
    there, even when the underlying detected cell is fully interior -- such a box is just
    as edge-limited/asymmetric in practice (less context on the clamped side) as one whose
    raw detection touches the edge outright, so both are excluded here. Intended for use
    on the full-session-averaged reference image only (see caller); per-epoch boxes are
    derived by shifting an already-vetted reference box, not re-filtered here.

    Input Parameters
    ----------------
    masks: 2d array, each ROI has a unique integer value

    Returns
    -------
    filtered_mask: 2d array, after filtering
        Note - the filtered mask is not necessarily contiguous
    """
    bb_masks = get_bounding_box(
        masks,
        area_extension_factor=area_extension_factor,
        min_buffer_um=min_buffer_um,
        pix_size=pix_size,
    )
    mask_inds = np.setdiff1d(np.unique(masks), 0)
    filtered_mask = masks.copy()
    for i, roi_id in enumerate(mask_inds):
        box = bb_masks[i]
        if (
            np.any(box[0, :])
            or np.any(box[-1, :])
            or np.any(box[:, 0])
            or np.any(box[:, -1])
        ):
            filtered_mask[masks == roi_id] = 0
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


def get_bounding_box(
    masks: np.array,
    area_extension_factor: float = 2,
    min_buffer_um: float = 5,
    pix_size: float = 0.78,
) -> np.array:
    """Get bounding box of ROI masks: the tight bounding box of each ROI, expanded on
    each axis by whichever is LARGER of (a) a size-proportional extension (assuming a
    circular ROI, area_extension_factor controls how much bigger the box's area is vs
    the ROI's own area) or (b) a fixed min_buffer_um (converted to pixels via pix_size).

    The size-proportional extension alone gives very small ROIs almost no buffer at all
    (e.g. a 1px-tall ROI got ~0px of padding under a pure ratio, occasionally rounding
    down to an empty box entirely) -- flooring at min_buffer_um keeps every box a
    physically meaningful, non-degenerate size regardless of how small the underlying
    segmented region is, while still growing proportionally for larger ROIs as before.

    Parameters:
    -----------
    masks : np.array
        ROI masks (2D, from CellPose)
    area_extension_factor : float, optional
        factor to extend the bounding box, by default 2
        Roughly the area of the bounding box will be larger than that of the ROI by this factor
        Assuming circular ROI.
    min_buffer_um : float, optional
        minimum buffer (in um) enforced on all sides of each ROI's tight bounding box,
        even where the size-proportional extension would be smaller, by default 5
    pix_size : float, optional
        pixel size in um, by default 0.78

    Returns:
    -----------
    bb_masks : np.array
        bounding box masks (3D, allowing overlaps)
    """

    bb_extension = np.sqrt(area_extension_factor * np.pi / 4)
    min_buffer_px = min_buffer_um / pix_size
    mask_inds = np.setdiff1d(np.unique(masks), 0)

    bb_masks = np.zeros((len(mask_inds), *masks.shape), dtype=np.uint16)
    for i, mask_i in enumerate(mask_inds):
        y, x = np.where(masks == mask_i)
        bb_y_tight = [y.min(), y.max()]
        bb_x_tight = [x.min(), x.max()]
        bb_y_tight_len = bb_y_tight[1] - bb_y_tight[0]
        bb_x_tight_len = bb_x_tight[1] - bb_x_tight[0]
        y_extension = max(bb_extension * bb_y_tight_len / 2, min_buffer_px)
        x_extension = max(bb_extension * bb_x_tight_len / 2, min_buffer_px)
        bb_y = [
            max(0, int(np.round(bb_y_tight[0] - y_extension))),
            min(masks.shape[0], int(np.round(bb_y_tight[1] + y_extension))),
        ]
        bb_x = [
            max(0, int(np.round(bb_x_tight[0] - x_extension))),
            min(masks.shape[1], int(np.round(bb_x_tight[1] + x_extension))),
        ]
        bb_masks[i, bb_y[0] : bb_y[1], bb_x[0] : bb_x[1]] = mask_i
    return bb_masks


def _bbox_coords(bb_masks):
    """Extract (top_left, width, height) for each box in a get_bounding_box() output.
    Returns a list of dicts: {"top_left": [x, y], "width": w, "height": h} (x=column,
    y=row, 0-indexed from the image's top-left corner; matches the box's exact nonzero
    rectangular extent -- get_bounding_box fills each box as a solid rectangle)."""
    boxes = []
    for i in range(bb_masks.shape[0]):
        y, x = np.where(bb_masks[i] > 0)
        if len(y) == 0:
            continue
        y0, y1, x0, x1 = int(y.min()), int(y.max()), int(x.min()), int(x.max())
        boxes.append({"top_left": [x0, y0], "width": x1 - x0 + 1, "height": y1 - y0 + 1})
    return boxes


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
