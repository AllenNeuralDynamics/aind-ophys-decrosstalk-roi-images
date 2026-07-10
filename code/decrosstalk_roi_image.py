import warnings
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from scipy import ndimage
from skimage import filters, measure

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
    max_grid_val: float = 0.36,
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
    coarse_step: float = 0.04,
    max_grid_val: float = 0.36,
    grid_interval: float = 0.01,
    fine_window: float = 0.05,
):
    """Find (alpha, beta) minimizing mean normalized MI across ROI boxes.

    Two passes: a coarse grid (step `coarse_step`, 0..max_grid_val) locates the basin,
    then a fine grid (step `grid_interval`, +/-`fine_window` around the coarse minimum,
    clipped to [0, max_grid_val]) refines it. ~5x fewer evaluations than the full grid.
    Verified to reproduce the full-resolution grid argmin within one grid step on
    good/over/under/flat cases (session02 verify gate); index-caching (precomputed
    `bb_yx_list`) is exact.

    Returns (alpha, beta, grid_vals). `grid_vals` is a flattened (n x n) grid over
    [0, max_grid_val] at `grid_interval` (n = max_grid_val/grid_interval + 1), filled at
    the evaluated coarse AND fine (alpha, beta) points and NaN elsewhere -- so it retains
    the fine (grid_interval) resolution around the minimum plus the coarse wide landscape,
    with implicit regular-grid coordinates, without computing the full grid. NB shape
    differs from the old full grid (now 37x37 over 0-0.36, sparse/NaN); reshape and use
    nan-aware ops (e.g. np.nanargmin) downstream.
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
        av = av[av <= max_grid_val + 1e-9]
        bv = np.arange(lo_b, hi_b + step, step)
        bv = bv[bv <= max_grid_val + 1e-9]
        vals, ab = [], []
        for a in av:
            for b in bv:
                vals.append(_mean_norm_mi(a, b, data, shape, bb_yx_list, mi_raw))
                ab.append([float(a), float(b)])
        return vals, ab

    coarse_vals, coarse_ab = _grid(0, max_grid_val, 0, max_grid_val, coarse_step)
    a0, b0 = coarse_ab[int(np.argmin(coarse_vals))]
    fine_vals, fine_ab = _grid(
        max(0, a0 - fine_window), min(max_grid_val, a0 + fine_window),
        max(0, b0 - fine_window), min(max_grid_val, b0 + fine_window),
        grid_interval,
    )
    alpha, beta = fine_ab[int(np.argmin(fine_vals))]

    # Assemble a sparse full-resolution landscape: an (n x n) grid over [0, max_grid_val]
    # at `grid_interval`, filled at the evaluated coarse AND fine (alpha, beta) points and
    # NaN elsewhere. Keeps implicit regular-grid coordinates while retaining fine (0.01)
    # resolution around the minimum plus the coarse wide landscape.
    n = int(round(max_grid_val / grid_interval)) + 1
    grid_vals = np.full((n, n), np.nan)
    for (a, b), v in list(zip(coarse_ab, coarse_vals)) + list(zip(fine_ab, fine_vals)):
        grid_vals[int(round(a / grid_interval)), int(round(b / grid_interval))] = v
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


def landscape_quality(mean_norm_mi_values, region="coarse", grid_interval=0.01,
                      coarse_step=0.04, fine_window=0.05):
    """Curvature / flatness / SNR of one epoch's MI objective basin, from a 2D quadratic
    fit to either the COARSE or the FINE grid points.

    `mean_norm_mi_values` is one epoch's flattened (n*n) objective grid (as stored in
    mean_norm_mi_list), reshaped to (n, n) over [0, (n-1)*grid_interval]. Fits
        z ~ c0 + c1 a + c2 b + c3 a^2 + c4 b^2 + c5 a b
    on the selected region:
      region="coarse": coarse lattice (every coarse_step/grid_interval-th point) -> GLOBAL
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
    ax_full = np.arange(n) * grid_interval
    depth = 1.0 - float(np.nanmin(G))
    if region == "coarse":
        step = max(int(round(coarse_step / grid_interval)), 1)
        idx = np.arange(0, n, step)
        A, B = np.meshgrid(ax_full[idx], ax_full[idx], indexing="ij")
        Z = G[np.ix_(idx, idx)]
    elif region == "fine":
        i0, j0 = np.unravel_index(int(np.nanargmin(G)), G.shape)
        w = max(int(round(fine_window / grid_interval)), 1)
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


def mean_landscape_quality(mean_norm_mi_list, grid_interval=0.01, coarse_step=0.04,
                           fine_window=0.05):
    """Epoch-mean of landscape_quality over all epochs, for BOTH the coarse and fine fits.
    Returns keys suffixed '_coarse' / '_fine' (nan-safe). Metric values only (no decision)."""
    out = {}
    for region in ("coarse", "fine"):
        per = [landscape_quality(g, region=region, grid_interval=grid_interval,
                                 coarse_step=coarse_step, fine_window=fine_window)
               for g in mean_norm_mi_list]
        for k in _LQ_KEYS:
            vals = np.array([p[k] for p in per], dtype=float)
            out[f"{k}_{region}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return out


def _cell_mask(img, dilate=2):
    """Boolean mask of cell footprints from basic_segmentation, with holes filled (so the
    dim nucleus inside a detected cell rim is included, not left as a low-value pixel) and
    a small dilation for a margin. Used to EXCLUDE cells before estimating background."""
    m = basic_segmentation(img) > 0
    m = ndimage.binary_fill_holes(m)
    if dilate:
        m = ndimage.binary_dilation(m, iterations=dilate)
    return m


def background_correlation(sig_mean, pai_mean, block=16, min_valid=0.3,
                           gauss_sigma=30, dilate=2):
    """Low-frequency background correlation between a plane and its paired plane (both
    full-session mean FOVs, same registration frame), with CELLS REMOVED.

    The MI model assumes the vasculature-shadow / illumination background is shared between
    the two planes; for far-apart (deep) pairs this can break. Background is estimated after
    masking out segmented cells (basic_segmentation, holes filled so dim nuclei are excluded
    too, dilated) in EITHER plane, then the two planes are correlated:
      - ``bg_corr``       : per-block MEDIAN of the non-cell pixels (neuropil/vasculature
                            background). Primary metric.
      - ``bg_corr_gauss`` : naive heavy-Gaussian low-pass, no cell removal (reference).
    Invalid (<=0, warped-border) pixels also masked; blocks below min_valid valid fraction
    dropped (no motion crop needed). bg_corr ~1 = shared background (assumption holds),
    lower = patterns differ. Falls monotonically with pair separation (0-1 ~0.77 ->
    6-7 ~0.43). Identical to decrosstalk_qc.metrics.background_correlation.

    Returns dict(bg_corr, bg_corr_gauss, n_blocks).
    """
    s = np.asarray(sig_mean, dtype=float)
    p = np.asarray(pai_mean, dtype=float)
    H, W = s.shape
    h, w = (H // block) * block, (W // block) * block
    s, p = s[:h, :w], p[:h, :w]
    infov = np.isfinite(s) & np.isfinite(p) & (s > 0) & (p > 0)
    cells = _cell_mask(s, dilate) | _cell_mask(p, dilate)  # cell in either plane
    valid = infov & ~cells
    nb = (h // block, w // block)
    frac = valid.reshape(nb[0], block, nb[1], block).mean(axis=(1, 3))
    goodblk = frac >= min_valid

    def _coarse_median(img):
        a = np.where(valid, img, np.nan).reshape(nb[0], block, nb[1], block)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmedian(a, axis=(1, 3))

    def _coarse_gauss(img):
        g = ndimage.gaussian_filter(np.where(infov, img, 0.0), gauss_sigma)
        return g.reshape(nb[0], block, nb[1], block).mean(axis=(1, 3))

    def _corr(x, y):
        m = goodblk & np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[m], y[m])[0, 1]) if int(m.sum()) >= 10 else float("nan")

    return {
        "bg_corr": _corr(_coarse_median(s), _coarse_median(p)),
        "bg_corr_gauss": _corr(_coarse_gauss(s), _coarse_gauss(p)),
        "n_blocks": int(goodblk.sum()),
    }


# Fixed axis caps for the landscape-quality panels so pages are comparable across sessions
# at a glance (from the good+bad distribution over ~156 planes, ~p99 with headroom). Values
# above a cap are drawn as a "^" marker on the top edge (off-scale). Kept identical to
# decrosstalk_qc.plots.
CURV_YMAX = 20.0   # Hessian eigenvalues; p99~17, max~24
SNR_YMAX = 100.0   # basin SNR; p99~98, max~158
SE_YMAX = 0.02     # vertex SE (flatness); p99 se_b~0.016, rare spikes up to ~4


def _capped_plot(ax, x, y, ymax, fmt, color, label=None):
    """Line plot with values capped at ymax; off-scale points marked '^' on the top edge."""
    y = np.asarray(y, dtype=float)
    ax.plot(x, np.minimum(y, ymax), fmt, color=color, label=label)
    over = np.isfinite(y) & (y > ymax)
    if over.any():
        ax.plot(np.asarray(x)[over], np.full(int(over.sum()), ymax), "^", color=color,
                ms=10, mec="k", mew=0.5, clip_on=False, zorder=6)


def render_landscape_page(mean_norm_mi_list, alpha_list, beta_list, grid_interval=0.01,
                          title="", applied=None, save=None):
    """One-page landscape QC figure: per-epoch MI landscapes + stability across epochs.

    `mean_norm_mi_list` is the list of per-epoch flattened (n*n) objective grids (as stored
    in the decrosstalk h5). Top block: a grid of per-epoch heatmaps (shared color scale,
    argmin marked). Bottom: (1) per-epoch argmins on the alpha-beta plane, (2) alpha & beta
    vs epoch. `applied` = the (alpha, beta) actually applied (reciprocity-averaged), marked
    distinctly. Kept identical to decrosstalk_qc.plots.render_landscape_page. Saves to
    `save` (Agg) if given, else returns the figure.
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
    gmax = (N - 1) * grid_interval
    extent = [0, gmax, 0, gmax]
    ax_vals = np.arange(N) * grid_interval
    finite = grid[np.isfinite(grid)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

    # per-epoch landscape-quality (fine = local basin, coarse = global bowl)
    ep = np.arange(n)
    pf = [landscape_quality(grid[e].ravel(), region="fine", grid_interval=grid_interval)
          for e in range(n)]
    pc = [landscape_quality(grid[e].ravel(), region="coarse", grid_interval=grid_interval)
          for e in range(n)]

    def qa(per, k):
        return np.array([per[e][k] for e in range(n)], dtype=float)

    ncols = min(n, 5)
    nrows_land = math.ceil(n / ncols)
    land_h, bot_h, title_h = nrows_land * 2.5, 5.2, 0.7
    H = land_h + bot_h + title_h
    fig = plt.figure(figsize=(max(ncols, 3) * 2.7, H))
    f_land_top, f_land_bot = 1 - title_h / H, (bot_h + 0.4) / H
    f_bot_top, f_bot_bot = (bot_h - 0.2) / H, 0.5 / H
    gs_top = fig.add_gridspec(nrows_land, ncols, top=f_land_top, bottom=f_land_bot,
                              left=0.06, right=0.89, hspace=0.5, wspace=0.32)
    gs_bot = fig.add_gridspec(2, 3, top=f_bot_top, bottom=f_bot_bot,
                              left=0.07, right=0.95, hspace=0.6, wspace=0.45)

    im = None
    for e in range(n):
        ax = fig.add_subplot(gs_top[e // ncols, e % ncols])
        im = ax.imshow(grid[e].T, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                       aspect="auto", cmap="viridis")
        if np.isfinite(grid[e]).any():
            ai, bi = np.unravel_index(int(np.nanargmin(grid[e])), grid[e].shape)
            ax.plot(ax_vals[ai], ax_vals[bi], "r+", ms=9, mew=1.6)
        ax.set_title(f"ep{e}  a={alpha[e]:.2f} b={beta[e]:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
        if e % ncols == 0:
            ax.set_ylabel("beta", fontsize=8)
        if e // ncols == nrows_land - 1:
            ax.set_xlabel("alpha", fontsize=8)
    if im is not None:
        cax = fig.add_axes([0.905, f_land_bot + 0.02, 0.012, (f_land_top - f_land_bot) * 0.9])
        fig.colorbar(im, cax=cax).set_label("norm. MI (basin = low)", fontsize=8)

    # (0,0) per-epoch argmins on the alpha-beta plane
    axs = fig.add_subplot(gs_bot[0, 0])
    sc = axs.scatter(alpha, beta, c=ep, cmap="plasma", s=45,
                     edgecolor="k", linewidth=0.4, zorder=3)
    axs.scatter([alpha.mean()], [beta.mean()], marker="*", s=220, c="lime",
                edgecolor="k", zorder=4, label="epoch mean")
    if applied is not None:
        axs.scatter([applied[0]], [applied[1]], marker="X", s=130, c="red",
                    edgecolor="k", zorder=5, label="applied")
    axs.set_xlabel("alpha*", fontsize=8); axs.set_ylabel("beta*", fontsize=8)
    axs.set_xlim(0, gmax); axs.set_ylim(0, gmax)  # fixed range -> tight vs spread visible
    axs.set_title(f"argmin stability (n={n})  sd_a={alpha.std():.3f} sd_b={beta.std():.3f}",
                  fontsize=8)
    axs.legend(fontsize=7, loc="best"); axs.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=axs, fraction=0.046, pad=0.02); cb.set_label("epoch", fontsize=7)

    # (0,1) alpha & beta vs epoch
    axl = fig.add_subplot(gs_bot[0, 1])
    axl.plot(ep, alpha, "o-", color="C0", label="alpha*")
    axl.plot(ep, beta, "s-", color="C1", label="beta*")
    if applied is not None:
        axl.axhline(applied[0], color="C0", ls="--", lw=1, alpha=0.7)
        axl.axhline(applied[1], color="C1", ls="--", lw=1, alpha=0.7)
    axl.set_xlabel("epoch", fontsize=8); axl.set_ylabel("coefficient", fontsize=8)
    axl.set_ylim(0, gmax)
    axl.set_title("coefficient vs epoch", fontsize=8)
    axl.legend(fontsize=7); axl.grid(alpha=0.3)

    # (0,2) basin curvature vs epoch (fine-fit Hessian eigenvalues)
    axc = fig.add_subplot(gs_bot[0, 2])
    _capped_plot(axc, ep, qa(pf, "lam_min"), CURV_YMAX, "o-", "C0", label="lam_min")
    _capped_plot(axc, ep, qa(pf, "lam_max"), CURV_YMAX, "s-", "C3", label="lam_max")
    axc.set_xlabel("epoch", fontsize=8); axc.set_ylabel(f"curvature (fine, <={CURV_YMAX:g})", fontsize=8)
    axc.set_ylim(0, CURV_YMAX)
    axc.set_title("basin curvature vs epoch", fontsize=8)
    axc.legend(fontsize=7); axc.grid(alpha=0.3)

    # (1,0) basin SNR vs epoch (fine)
    axsn = fig.add_subplot(gs_bot[1, 0])
    _capped_plot(axsn, ep, qa(pf, "snr"), SNR_YMAX, "o-", "C2")
    axsn.set_xlabel("epoch", fontsize=8); axsn.set_ylabel(f"SNR (fine, <={SNR_YMAX:g})", fontsize=8)
    axsn.set_ylim(0, SNR_YMAX)
    axsn.set_title("basin SNR vs epoch", fontsize=8); axsn.grid(alpha=0.3)

    # (1,1) flatness (vertex SE) vs epoch: alpha and beta (fine)
    axf = fig.add_subplot(gs_bot[1, 1])
    _capped_plot(axf, ep, qa(pf, "se_a"), SE_YMAX, "o-", "C0", label="alpha flatness (se_a)")
    _capped_plot(axf, ep, qa(pf, "se_b"), SE_YMAX, "s-", "C1", label="beta flatness (se_b)")
    axf.set_xlabel("epoch", fontsize=8); axf.set_ylabel(f"vertex SE (fine, <={SE_YMAX:g})", fontsize=8)
    axf.set_ylim(0, SE_YMAX)
    axf.set_title("flatness (alpha, beta) vs epoch  (^=off-scale)", fontsize=8)
    axf.legend(fontsize=7); axf.grid(alpha=0.3)

    # (1,2) epoch-mean quality summary (coarse / fine)
    axt = fig.add_subplot(gs_bot[1, 2]); axt.axis("off")
    lines = "epoch-mean quality\n%-9s %8s %8s\n" % ("", "coarse", "fine")
    for k in ("lam_min", "lam_max", "se_a", "se_b", "snr"):
        lines += "%-9s %8.4g %8.4g\n" % (k, np.nanmean(qa(pc, k)), np.nanmean(qa(pf, k)))
    axt.text(0.0, 1.0, lines, family="monospace", fontsize=9, va="top",
             transform=axt.transAxes)
    axt.set_title("quality summary", fontsize=8)

    fig.suptitle(title, fontsize=11, y=1 - 0.25 * title_h / H)
    if save is not None:
        fig.savefig(save, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return save
    return fig


def decrosstalk_roi_image_single_pair_from_episodic_mean_fov(
    oeid: int,
    paired_reg_emf_fn: str,
    input_dir: Path,
    start_frame: int,
    pix_size: float,
    motion_buffer: int = 5,
    grid_interval: float = 0.01,
    max_grid_val: float = 0.36,
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
    # Coarse-to-fine grid search for (alpha, beta) minimizing mean normalized MI across
    # ROI boxes (see coarse_to_fine_grid_search). Reproduces the full-resolution grid
    # argmin within one grid step (session02 verify gate) at ~5x fewer evaluations.
    # NB: mean_norm_mi_values is now a sparse 37x37 grid (0..max_grid_val at grid_interval)
    # -- fine resolution near the minimum, coarse elsewhere, NaN at unevaluated points --
    # not the old dense 31x31 (0-0.30) grid; use nan-aware ops downstream.
    alpha, beta, mean_norm_mi_values = coarse_to_fine_grid_search(
        signal_mean, paired_mean, bb_masks,
        max_grid_val=max_grid_val, grid_interval=grid_interval,
    )
    return alpha, beta, np.array(mean_norm_mi_values).tolist()


def basic_segmentation(
    mean_img: np.array,
    min_object_size: int = 100,
    max_object_size: int = 300,
    sigma_segmentation: int = 30,
) -> np.array:
    """Fast classical soma segmentation, replacing CellPose.

    Adapted from aind-ophys-movie-qc `get_and_plot_basic_segmentation`:
    Gaussian high-pass (remove neuropil/background) -> Otsu threshold ->
    connected components -> keep objects with min < area < max pixels.
    Returns an integer-labeled mask (0=background, 1..N=ROIs), matching the
    CellPose `model.eval` output consumed downstream.

    Validated to reproduce the CellPose-pipeline alpha/beta (esp. beta, the
    crosstalk-removal knob) within ~0.01-0.02; see session02 consistency check.
    Much faster (~0.1 s vs ~22 s per epoch) and drops the torch/cellpose dependency.
    """
    neuropil = ndimage.gaussian_filter(mean_img, sigma=sigma_segmentation)
    high_pass = mean_img - neuropil
    binary = high_pass > filters.threshold_otsu(high_pass)
    label_image = measure.label(binary)
    masks = np.zeros_like(label_image)
    n = 0
    for region in measure.regionprops(label_image):
        if min_object_size < region.area < max_object_size:
            n += 1
            masks[label_image == region.label] = n
    return masks


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

    signal_masks = basic_segmentation(signal_mean)

    dendrite_diameter_px = dendrite_diameter_um / pix_size
    signal_masks_dendrite_filtered = filter_dendrite(
        signal_masks, dendrite_diameter_pix=dendrite_diameter_px
    )
    signal_masks_filtered = filter_border_roi(
        signal_masks_dendrite_filtered, buffer_pix=nrshiftmax
    )

    paired_masks = basic_segmentation(paired_mean)
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
