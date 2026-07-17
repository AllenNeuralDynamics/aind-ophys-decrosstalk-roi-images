"""top level run script"""

import argparse
import json
import logging
import os
from datetime import datetime as dt
from pathlib import Path
from typing import Union

import decrosstalk_roi_image as dri
import h5py as h5
import numpy as np
import paired_plane_registration as ppr
from aind_data_schema.core.processing import DataProcess, ProcessName
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_log_utils.log import setup_logging
from aind_qcportal_schema.metric_value import DropdownMetric


def write_data_process(
    metadata: dict,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    unique_id: str,
    start_time: dt,
    end_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    input_fp: str
        path to raw movies
    output_fp: str
        path to motion corrected movies
    start_time: dt
        start time of processing
    end_time: dt
        end time of processing
    """
    data_proc = DataProcess(
        name=ProcessName.VIDEO_PLANE_DECROSSTALK,
        software_version=os.getenv("VERSION", ""),
        start_date_time=start_time.isoformat(),
        end_date_time=end_time.isoformat(),
        input_location=str(input_fp),
        output_location=str(output_fp),
        code_url=(os.getenv("REPO_URL", "")),
        parameters=metadata,
    )
    output_dir = Path(output_fp).parent
    with open(output_dir / f"{unique_id}_decrosstalk_data_process.json", "w") as f:
        json.dump(json.loads(data_proc.model_dump_json()), f, indent=4)


def write_qc_metrics(output_dir: Path, unique_id: str) -> None:
    """Write QC metrics to output directory

    Parameters
    ----------
    output_dir: Path
        path to output data
    unique_id: str
        unique identifier for experiment
    """

    metric = QCMetric(
        name=f"{unique_id} Decrosstalk",
        description="Episodic mean FOV of decrosstalk movie",
        reference=f"{unique_id}/decrosstalk/{unique_id}_decrosstalk_episodic_mean_fov.webm",
        status_history=[
            QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)
        ],
        value=DropdownMetric(
            value="Reasonable",
            options=[
                "Reasonable",
                "Unreasonable",
            ],
            status=[
                Status.PASS,
                Status.FAIL,
            ],
        ),
    )

    with open(
        output_dir / f"{unique_id}_decrosstalk_episodic_mean_fov_metric.json", "w"
    ) as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)


def average_paired_coeffs(a1_list, b1_list, a2_list, b2_list):
    """Reciprocity-average the two paired planes' epoch-mean (alpha, beta).

    A physical crosstalk leak is estimated twice. With mixing matrix
    [[1-alpha, beta], [alpha, 1-beta]] (beta = leak INTO the signal plane, alpha = leak
    OUT of it), for planes 1 and 2:
        L(1->2) = alpha_1 = beta_2
        L(2->1) = beta_1  = alpha_2
    Averaging the two estimates of each leak yields one value per physical leak (enforcing
    reciprocity), then assigns them, flipped, to each plane. Validated in session02
    (residual_cv): free vs per-plane on held-out data (~ -0.0001 residual MI) with tighter
    physical consistency.

    Returns ((alpha1, beta1), (alpha2, beta2)) to apply to plane 1 and plane 2.
    """
    a1, b1 = float(np.mean(a1_list)), float(np.mean(b1_list))
    a2, b2 = float(np.mean(a2_list)), float(np.mean(b2_list))
    leak_1to2 = (a1 + b2) / 2.0
    leak_2to1 = (b1 + a2) / 2.0
    return (leak_1to2, leak_2to1), (leak_2to1, leak_1to2)


def estimate_alpha_beta(
    oeid: str,
    paired_oeid: str,
    input_dir: Path,
    output_dir: Path,
    grid_interval_fine: float = 0.01,
    grid_interval_coarse: float = 0.04,
    coef_max: float = 0.36,
    dendrite_diameter_um: float = 4,
    max_diameter_um: float = 20,
    num_top_rois: int = 10,
    sigma_segmentation: int = 30,
):
    """Estimate per-epoch (alpha, beta) for one plane from the episodic-mean-FOV images.

    No movie is reconstructed here; estimation is separated from application so paired
    coefficients can be reciprocity-averaged before the (expensive) full-movie apply.

    `sigma_segmentation` is the Gaussian high-pass sigma used for the once-per-plane
    reference segmentation (see `decrosstalk_roi_image.get_signal_paired_top_masks` /
    `segment_and_filter_with_relaxation`), passed straight through.

    Returns
    -------
    (alpha_list, beta_list, mean_norm_mi_list, paired_reg_emf_fn, signal_bboxes_list,
     paired_bboxes_list, example_signal_mean, example_paired_mean, signal_mean_avg,
     paired_mean_avg, paired_bboxes_ref, signal_mean_list, paired_mean_list)
    """
    logging.info(f"Estimating alpha/beta for {oeid} (paired {paired_oeid})")
    paired_reg_emf_fn = next(
        output_dir.parent.parent.rglob(
            f"{paired_oeid}_registered_to_pair_episodic_mean_fov.h5"
        )
    )
    (
        _,
        alpha_list,
        beta_list,
        mean_norm_mi_list,
        signal_bboxes_list,
        paired_bboxes_list,
        example_signal_mean,
        example_paired_mean,
        signal_mean_avg,
        paired_mean_avg,
        paired_bboxes_ref,
        signal_mean_list,
        paired_mean_list,
    ) = dri.decrosstalk_roi_image_from_episodic_mean_fov(
        oeid,
        paired_reg_emf_fn,
        input_dir.parent,
        grid_interval_fine=grid_interval_fine,
        grid_interval_coarse=grid_interval_coarse,
        coef_max=coef_max,
        dendrite_diameter_um=dendrite_diameter_um,
        max_diameter_um=max_diameter_um,
        num_top_rois=num_top_rois,
        sigma_segmentation=sigma_segmentation,
    )
    return (
        alpha_list,
        beta_list,
        mean_norm_mi_list,
        paired_reg_emf_fn,
        signal_bboxes_list,
        paired_bboxes_list,
        example_signal_mean,
        example_paired_mean,
        signal_mean_avg,
        paired_mean_avg,
        paired_bboxes_ref,
        signal_mean_list,
        paired_mean_list,
    )


def apply_decrosstalk_movie(
    oeid: str,
    paired_oeid: str,
    input_dir: Path,
    output_dir: Path,
    alpha: float,
    beta: float,
    alpha_list: list,
    beta_list: list,
    mean_norm_mi_list: list,
    paired_reg_emf_fn: Path,
    start_time: dt,
    partner_alpha_list: list = None,
    partner_beta_list: list = None,
    grid_interval_fine: float = 0.01,
    grid_interval_coarse: float = 0.04,
    coef_max: float = 0.36,
    recip_flag: float = 0.05,
    signal_bboxes_list: list = None,
    paired_bboxes_list: list = None,
    example_signal_mean: np.ndarray = None,
    example_paired_mean: np.ndarray = None,
    signal_mean_avg: np.ndarray = None,
    paired_mean_avg: np.ndarray = None,
    paired_bboxes_ref: list = None,
    signal_mean_list: list = None,
    paired_mean_list: list = None,
) -> Path:
    """Apply the given (alpha, beta) mixing correction to the full registered movie in
    chunks and write {oeid}_decrosstalk.h5.

    `alpha`/`beta` are the coefficients actually applied (may be reciprocity-averaged);
    they are recorded in the metadata (alpha_mean/beta_mean). The stored alpha_list /
    beta_list / mean_norm_mi_list remain this plane's raw per-epoch estimates for QC.
    `partner_alpha_list`/`partner_beta_list`, if given, add the pair-symmetry panel to the
    landscape QC figure (reciprocity check). `signal_bboxes_list`/`paired_bboxes_list` are
    stored alongside the other QC values (qc-values.json). `example_signal_mean`/
    `example_paired_mean` are accepted for backward compatibility but are no longer used
    here (superseded by the two ROI-bounding-box QC pages below); kept as parameters since
    callers still forward them from `estimate_alpha_beta`. `signal_mean_avg`/
    `paired_mean_avg`/`paired_bboxes_ref`, if given, render a full-session-mean-FOV
    ROI-bounding-box QC page. `signal_mean_list`/`paired_mean_list`, if given (together
    with `signal_bboxes_list`/`paired_bboxes_list`), render a per-epoch grid ROI-bounding-
    box QC page.
    """
    logging.info(
        f"Applying decrosstalk to {oeid}: alpha={alpha:.3f}, beta={beta:.3f}"
    )
    paired_oeid_reg_to_oeid_full_fn = next(
        Path("../scratch").rglob(f"{paired_oeid}_registered_to_pair.h5")
    )
    # Landscape-quality diagnostics (curvature / flatness / SNR) from the per-epoch MI
    # grid, coarse-grid fit. METRIC VALUES ONLY -- no pass/warn decision (thresholds TBD
    # from accumulated data). Recorded for QC aggregation across sessions.
    lq = dri.mean_landscape_quality(
        mean_norm_mi_list, grid_interval_fine=grid_interval_fine,
        grid_interval_coarse=grid_interval_coarse,
    )
    metadata = {
        "alpha_mean": round(float(alpha), 2),
        "beta_mean": round(float(beta), 2),
        "paired_emf": str(paired_reg_emf_fn),
        "landscape_quality": {k: round(v, 5) for k, v in lq.items()},
    }

    # To reduce RAM usage, get/save the decrosstalk_data in chunks:
    chunk_size = 5000  # num of frames in each chunk

    with h5.File(input_dir / "motion_correction" / f"{oeid}_registered.h5", "r") as f:
        data_shape = f["data"].shape
    data_length = data_shape[0]
    start_frames = np.arange(0, data_length, chunk_size)
    end_frames = np.append(start_frames[1:], data_length)
    assert end_frames[-1] == data_length
    decrosstalk_fn = output_dir / f"{oeid}_decrosstalk.h5"

    # generate the decrosstalk movie with the applied alpha and beta values
    # using the full paired registered movie
    chunk_no = 0
    for start_frame, end_frame in zip(start_frames, end_frames):
        with h5.File(paired_oeid_reg_to_oeid_full_fn, "r") as f:
            paired_data = f["data"][start_frame:end_frame]
        with h5.File(
            input_dir / "motion_correction" / f"{oeid}_registered.h5", "r"
        ) as f:
            signal_data = f["data"][start_frame:end_frame]
        recon_signal_data = np.zeros_like(signal_data, dtype=np.int16)
        for temp_frame_index in range(signal_data.shape[0]):
            recon_signal_data[temp_frame_index, :, :] = dri.apply_mixing_matrix(
                alpha,
                beta,
                signal_data[temp_frame_index, :, :],
                paired_data[temp_frame_index, :, :],
            )[0]
        if chunk_no == 0:
            with h5.File(decrosstalk_fn, "w") as f:
                f.create_dataset(
                    "data",
                    data=recon_signal_data,
                    maxshape=(None, data_shape[1], data_shape[2]),
                )
                f.create_dataset("alpha_list", data=alpha_list)
                f.create_dataset("beta_list", data=beta_list)
                f.create_dataset("mean_norm_mi_list", data=mean_norm_mi_list)
                # Coefficients actually applied to this movie (may be reciprocity-
                # averaged, so NOT necessarily mean(alpha_list)/mean(beta_list)). Stored
                # unrounded so the corrected movie is self-describing; the rounded copy
                # also lives in {oeid}_decrosstalk_data_process.json.
                f.attrs["applied_alpha"] = float(alpha)
                f.attrs["applied_beta"] = float(beta)
                # landscape-quality diagnostics (metric values only, no decision)
                for _k, _v in lq.items():
                    f.attrs[f"landscape_{_k}"] = float(_v)
        else:
            with h5.File(decrosstalk_fn, "a") as f:
                f["data"].resize(
                    (f["data"].shape[0] + recon_signal_data.shape[0]), axis=0
                )
                f["data"][start_frame:end_frame] = recon_signal_data
        chunk_no += 1
    write_data_process(
        metadata,
        input_dir / "motion_correction" / f"{oeid}_registered.h5",
        decrosstalk_fn,
        oeid,
        start_time,
        dt.now(),
    )
    # One-page landscape QC figure: per-epoch landscapes + coefficient stability + pair
    # symmetry (if the partner's per-epoch coeffs are provided).
    # Non-critical (guarded) -- a plotting failure must not fail the decrosstalk run.
    partner = None
    if partner_alpha_list is not None and partner_beta_list is not None:
        partner = (partner_alpha_list, partner_beta_list)
    try:
        dri.render_landscape_page(
            mean_norm_mi_list, alpha_list, beta_list,
            title=f"{oeid} decrosstalk landscapes  (applied a={alpha:.3f}, b={beta:.3f})",
            applied=(float(alpha), float(beta)), partner=partner,
            grid_interval_fine=grid_interval_fine, grid_interval_coarse=grid_interval_coarse,
            coef_max=coef_max,
            recip_flag=recip_flag,
            save=str(output_dir / f"{oeid}_decrosstalk_landscape.png"),
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"landscape QC page failed for {oeid}: {exc}")
    # ROI-bounding-box QC figures: shows exactly which regions the estimator used, on both
    # planes' mean images. Non-critical (guarded) -- a plotting failure must not fail the
    # decrosstalk run.
    # (1) Full-session mean FOV: signal plane's (session-wide) boxes and the paired
    # plane's own PRE-shift reference boxes, both drawn on the full-session-averaged
    # reference images.
    try:
        if signal_mean_avg is not None and paired_mean_avg is not None:
            dri.render_roi_bbox_page(
                signal_mean_avg, paired_mean_avg,
                signal_bboxes_list[0] if signal_bboxes_list else [],
                paired_bboxes_ref if paired_bboxes_ref else [],
                title=f"{oeid} ROI bounding boxes (full-session mean FOV)",
                save=str(output_dir / f"{oeid}_decrosstalk_roi_boxes_mean_fov.png"),
            )
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"ROI bbox mean-FOV QC page failed for {oeid}: {exc}")
    # (2) Per-epoch grid: current-plane (fixed) boxes alongside the paired-plane
    # (per-epoch-shifted) boxes, epoch by epoch.
    try:
        if signal_mean_list and paired_mean_list:
            dri.render_roi_bbox_per_epoch_page(
                signal_mean_list, paired_mean_list,
                signal_bboxes_list, paired_bboxes_list,
                title=f"{oeid} ROI bounding boxes (per epoch)",
                save=str(output_dir / f"{oeid}_decrosstalk_roi_boxes_per_epoch.png"),
            )
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"ROI bbox per-epoch QC page failed for {oeid}: {exc}")
    # Same data as the figure above, as JSON (no plot) -- lets the QC figure be
    # regenerated or re-styled downstream without re-reading the (large) decrosstalk h5.
    try:
        dri.save_qc_values(
            mean_norm_mi_list, alpha_list, beta_list, applied=(float(alpha), float(beta)),
            oeid=oeid, paired_oeid=paired_oeid, partner=partner,
            grid_interval_fine=grid_interval_fine, grid_interval_coarse=grid_interval_coarse,
            coef_max=coef_max, recip_flag=recip_flag,
            signal_bboxes_list=signal_bboxes_list, paired_bboxes_list=paired_bboxes_list,
            save=str(output_dir / "qc-values.json"),
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"qc-values.json failed for {oeid}: {exc}")
    return decrosstalk_fn


def debug_movie(
    h5_file: Path, input_dir: Path, temp_path: Path = Path("../scratch")
) -> Path:
    """debug movie for development

    Parameters
    ----------
    h5_file: Path
        path to h5 file
    input_dir: Path
        root input directory
    temp_path: Path, optional
        path to temp directory, default is "../scratch"

    Returns
    -------
    h5_file: Path
        path to h5 file
    """
    logging.info("Running in debug %s", h5_file)
    session_fp = next(input_dir.rglob("session.json"), "")
    if not session_fp:
        raise FileNotFoundError(f"Could not find {session_fp}")
    frame_rate_hz = get_frame_rate(session_fp)
    with h5.File(h5_file, "r") as f:
        frames_6min = int(360 * float(frame_rate_hz))
        data = f["data"][:frames_6min]
    h5_file = temp_path / h5_file.name
    with h5.File(h5_file, "w") as f:
        f.create_dataset("data", data=data)
    return h5_file


def prepare_cached_paired_plane_movies(
    oeid1: str,
    oeid2: str,
    input_dir: Path,
    non_rigid: bool = True,
    block_size: list = [128, 128],
    debug: bool = False,
) -> Path:
    """
    Prepare cached paired plane movies

    Parameters
    ----------
    oeid1: str
        ophys experiment id
    oeid2: str
        ophys experiment id of paired experiment
    input_dir: Path
        path to input data
    non_rigid: bool
        True if non-rigid registration was run, False otherwise
    block_size: list
        block size of image, default is [128, 128]
    debug: bool, optional
        True if debugging, False otherwise
    Returns
    -------
    h5_file: Path
        path to cached paired plane movie
    """
    h5_file = next(input_dir.rglob(f"{oeid1}.h5"), "")
    if not h5_file:
        raise FileNotFoundError(f"Could not find {oeid1}.h5")
    if debug:
        h5_file = debug_movie(h5_file, input_dir)
    oeid_mt = next(input_dir.rglob(f"{oeid2}_motion_transform.csv"), "")
    if not oeid_mt:
        raise FileNotFoundError(f"Could not find {oeid2}_motion_transform.csv")
    transform_df = ppr.get_s2p_motion_transform(oeid_mt)
    if debug:
        with h5.File(h5_file, "r") as f:
            transform_df = transform_df.iloc[: f["data"].shape[0]].reset_index(drop=True)
    return ppr.paired_plane_cached_movie(
        h5_file, transform_df, non_rigid=non_rigid, block_size=block_size
    )


def read_json(json_fp: Path) -> dict:
    """
    Get processing json from input directory

    Parameters
    ----------
    json_fp: Path
        path to json

    Returns
    -------
    data: dict
        processing json
    """
    with open(json_fp, "r") as f:
        return json.load(f)


def get_block_size(input_dir: Path) -> list:
    """get image dimensions from processing json

    Parameters
    ----------
    input_dir: Path
        path to input data

    Returns
    -------
    block_size: list
        block size of image
    """
    data_process_fp = next(input_dir.rglob("*data_process.json"), "")
    if not data_process_fp:
        raise FileNotFoundError(f"Could not find data_process.json in {input_dir}")
    data_process_json = read_json(data_process_fp)
    try:
        block_size = data_process_json["parameters"]["suite2p_args"]["block_size"]
    except KeyError:
        block_size = data_process_json["parameters"]["suite2p_args"]["block_size"]
    return block_size


def check_non_rigid_registration(input_dir: Path) -> bool:
    """check processing json to see if non-rigid registration was run

    Parameters
    ----------
    input_dir: Path
        path to input data

    Returns
    -------
    bool
        True if non-rigid registration was run, False otherwise
    """
    data_process_fp = next(input_dir.rglob("*data_process.json"), "")
    if not data_process_fp:
        raise FileNotFoundError(f"Could not find data_process.json in {input_dir}")
    data_process_json = read_json(data_process_fp)
    try:
        nonrigid = data_process_json["parameters"]["suite2p_args"]["nonrigid"]
    except KeyError:
        nonrigid = data_process_json["parameters"]["suite2p_args"]["nonrigid"]
    return nonrigid


def make_output_dirs(oeid: str, output_dir: Path) -> Path:
    """
    Make output directories for decrosstalk processing

    Parameters
    ----------
    oeid: str
        ophys experiment id
    output_dir: Path
        path to output data

    Returns
    -------
    results_dir: Path
        path to decrosstalk output directory
    """
    results_dir = output_dir / oeid
    results_dir.mkdir(exist_ok=True)
    results_dir = output_dir / oeid / "decrosstalk"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def get_frame_rate(session_fp: Path) -> float:
    """Return frame rate from session.json

    Parameters
    ----------
    session_fp: Path
        Path to session file

    Returns
    -------
    frame_rate_hz: float
        Frame rate of time series
    """
    session_data = read_json(session_fp)
    frame_rate_hz = None
    for i in session_data.get("data_streams", ""):
        frame_rate_hz = [j["frame_rate"] for j in i["ophys_fovs"]]
        frame_rate_hz = frame_rate_hz[0]
        if frame_rate_hz:
            break
    if isinstance(frame_rate_hz, str):
        frame_rate_hz = float(frame_rate_hz)
    return frame_rate_hz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: x.lower() == "true",
        help="Enable debug mode (use --debug true or --debug false)",
    )
    parser.add_argument("--grid-interval-fine", type=float, default=0.01,
                        help="fine grid step for the coarse-to-fine (alpha,beta) search")
    parser.add_argument("--grid-interval-coarse", type=float, default=0.04,
                        help="coarse grid step for the coarse-to-fine (alpha,beta) search")
    parser.add_argument("--coef-max", type=float, default=0.36,
                        help="maximum alpha/beta value for the grid search and QC plot axes")
    parser.add_argument("--recip-flag", type=float, default=0.05,
                        help="reciprocity-gap threshold flagged in the QC figure")
    parser.add_argument("--dendrite-diameter-um", type=float, default=4,
                        help="lower-bound ROI diameter (um); smaller ROIs are discarded as dendrite fragments")
    parser.add_argument("--max-diameter-um", type=float, default=20,
                        help="upper-bound ROI diameter (um); larger ROIs are discarded as oversized blobs/artifacts")
    parser.add_argument("--num-top-rois", type=int, default=10,
                        help="number of top-intensity ROIs to keep per plane (also the Otsu-relaxation target count)")
    parser.add_argument("--sigma-segmentation", type=int, default=30,
                        help="Gaussian high-pass sigma for basic_segmentation/segment_and_filter_with_relaxation")

    args = parser.parse_args()
    input_dir = Path("../data/").resolve()
    output_dir = Path("../results/").resolve()
    debug = args.debug
    if debug:
        logging.info("Running in debug mode")
    num_frames = 1000
    if debug:
        num_frames = 300
    subject_fp = next(input_dir.rglob("subject.json"), "")
    if not subject_fp:
        raise FileNotFoundError(f"Could not find {subject_fp}")
    subject_data = read_json(subject_fp)
    data_description_fp = next(input_dir.rglob("data_description.json"), "")
    if not data_description_fp:
        raise FileNotFoundError(f"Could not find {data_description_fp}")
    data_description = read_json(data_description_fp)
    subject_id = subject_data.get("subject_id", "")
    name = data_description.get("name", "")
    setup_logging(
        "aind-ophys-ophys-decrosstalk-roi-images",
        mouse_id=subject_id,
        session_name=name,
    )
    experiment_dirs = input_dir.glob("pair*/*")
    oeid1_input_dir = next(experiment_dirs)
    oeid2_input_dir = next(experiment_dirs)
    oeid1 = oeid1_input_dir.name
    oeid2 = oeid2_input_dir.name
    oeid1_output_dir = make_output_dirs(oeid1, output_dir)
    oeid2_output_dir = make_output_dirs(oeid2, output_dir)
    non_rigid = check_non_rigid_registration(oeid1_input_dir)
    block_size = get_block_size(oeid1_input_dir)

    oeid1_reg_to_oeid2_motion_filepath = prepare_cached_paired_plane_movies(
        oeid1, oeid2, input_dir, non_rigid=non_rigid, block_size=block_size, debug=debug
    )
    oeid2_reg_to_oeid1_motion_filepath = prepare_cached_paired_plane_movies(
        oeid2, oeid1, input_dir, non_rigid=non_rigid, block_size=block_size, debug=debug
    )
    ppr.episodic_mean_fov(
        oeid1_reg_to_oeid2_motion_filepath, oeid1_output_dir, num_frames=num_frames
    )
    ppr.episodic_mean_fov(
        oeid2_reg_to_oeid1_motion_filepath, oeid2_output_dir, num_frames=num_frames
    )
    # Self-registered episodic-mean-FOV images for both planes (input to estimation).
    # Hoisted here (rather than inline with estimation) because reciprocity averaging
    # needs both planes estimated before either is applied.
    ppr.episodic_mean_fov(
        oeid1_input_dir / "motion_correction" / f"{oeid1}_registered.h5",
        oeid1_output_dir,
        num_frames=num_frames,
    )
    ppr.episodic_mean_fov(
        oeid2_input_dir / "motion_correction" / f"{oeid2}_registered.h5",
        oeid2_output_dir,
        num_frames=num_frames,
    )
    # Estimate per-epoch (alpha, beta) for BOTH planes first, then reciprocity-average the
    # paired coefficients (one physical leak -> one value), then apply to each full movie.
    start_time_oeid1 = dt.now()
    (
        a1_list, b1_list, mi1_list, paired_emf1,
        signal_bboxes1, paired_bboxes1, example_signal_mean1, example_paired_mean1,
        signal_mean_avg1, paired_mean_avg1, paired_bboxes_ref1,
        signal_mean_list1, paired_mean_list1,
    ) = estimate_alpha_beta(
        oeid1, oeid2, oeid1_input_dir, oeid1_output_dir,
        grid_interval_fine=args.grid_interval_fine,
        grid_interval_coarse=args.grid_interval_coarse,
        coef_max=args.coef_max,
        dendrite_diameter_um=args.dendrite_diameter_um,
        max_diameter_um=args.max_diameter_um,
        num_top_rois=args.num_top_rois,
        sigma_segmentation=args.sigma_segmentation,
    )
    start_time_oeid2 = dt.now()
    (
        a2_list, b2_list, mi2_list, paired_emf2,
        signal_bboxes2, paired_bboxes2, example_signal_mean2, example_paired_mean2,
        signal_mean_avg2, paired_mean_avg2, paired_bboxes_ref2,
        signal_mean_list2, paired_mean_list2,
    ) = estimate_alpha_beta(
        oeid2, oeid1, oeid2_input_dir, oeid2_output_dir,
        grid_interval_fine=args.grid_interval_fine,
        grid_interval_coarse=args.grid_interval_coarse,
        coef_max=args.coef_max,
        dendrite_diameter_um=args.dendrite_diameter_um,
        max_diameter_um=args.max_diameter_um,
        num_top_rois=args.num_top_rois,
        sigma_segmentation=args.sigma_segmentation,
    )
    (alpha1, beta1), (alpha2, beta2) = average_paired_coeffs(
        a1_list, b1_list, a2_list, b2_list
    )
    logging.info(
        f"Reciprocity-averaged coeffs: {oeid1} (alpha={alpha1:.3f}, beta={beta1:.3f}), "
        f"{oeid2} (alpha={alpha2:.3f}, beta={beta2:.3f})"
    )
    decrosstalk_fn1 = apply_decrosstalk_movie(
        oeid1, oeid2, oeid1_input_dir, oeid1_output_dir, alpha1, beta1,
        a1_list, b1_list, mi1_list, paired_emf1, start_time_oeid1,
        partner_alpha_list=a2_list, partner_beta_list=b2_list,
        grid_interval_fine=args.grid_interval_fine,
        grid_interval_coarse=args.grid_interval_coarse,
        coef_max=args.coef_max, recip_flag=args.recip_flag,
        signal_bboxes_list=signal_bboxes1, paired_bboxes_list=paired_bboxes1,
        example_signal_mean=example_signal_mean1, example_paired_mean=example_paired_mean1,
        signal_mean_avg=signal_mean_avg1, paired_mean_avg=paired_mean_avg1,
        paired_bboxes_ref=paired_bboxes_ref1,
        signal_mean_list=signal_mean_list1, paired_mean_list=paired_mean_list1,
    )
    decrosstalk_fn2 = apply_decrosstalk_movie(
        oeid2, oeid1, oeid2_input_dir, oeid2_output_dir, alpha2, beta2,
        a2_list, b2_list, mi2_list, paired_emf2, start_time_oeid2,
        partner_alpha_list=a1_list, partner_beta_list=b1_list,
        grid_interval_fine=args.grid_interval_fine,
        grid_interval_coarse=args.grid_interval_coarse,
        coef_max=args.coef_max, recip_flag=args.recip_flag,
        signal_bboxes_list=signal_bboxes2, paired_bboxes_list=paired_bboxes2,
        example_signal_mean=example_signal_mean2, example_paired_mean=example_paired_mean2,
        signal_mean_avg=signal_mean_avg2, paired_mean_avg=paired_mean_avg2,
        paired_bboxes_ref=paired_bboxes_ref2,
        signal_mean_list=signal_mean_list2, paired_mean_list=paired_mean_list2,
    )
    # Episodic-mean-FOV of the corrected movies (QC / downstream)
    ppr.episodic_mean_fov(
        decrosstalk_fn1, oeid1_output_dir, num_frames=num_frames, save_webm=True
    )
    ppr.episodic_mean_fov(
        decrosstalk_fn2, oeid2_output_dir, num_frames=num_frames, save_webm=True
    )
    (Path("../scratch/") / f"{oeid1}_registered_to_pair.h5").unlink()
    print("unlinking paired registered flies")
    (Path("../scratch/") / f"{oeid2}_registered_to_pair.h5").unlink()

    write_qc_metrics(oeid1_output_dir, oeid1)
    write_qc_metrics(oeid2_output_dir, oeid2)
