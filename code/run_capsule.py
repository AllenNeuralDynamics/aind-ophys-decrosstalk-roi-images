"""top level run script"""

import argparse
import json
import logging
import os
import shutil
import sys
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


def decrosstalk_roi_movie(
    oeid: str, paired_oeid: str, input_dir: Path, output_dir: Path, start_time: dt
) -> Path:
    """
    Run decrosstalk on roi movie

    Parameters
    ----------
    oeid: str
        ophys experiment id
    paired_oeid: str
        ophys experiment id of paired experiment
    input_dir: Path
        path to input data
    output_dir: Path
        path to output data
    start_time: dt
        start time of decrosstalk processing

    Returns
    -------
    decrosstalk_fn: Path
        path to decrosstalk roi movie
    """
    logging.info(f"Input directory, {input_dir}")
    logging.info(f"Output directory, {output_dir}")
    logging.info(f"Ophys experiment ID pairs, {oeid}, {paired_oeid}")
    paired_oeid_reg_to_oeid_full_fn = next(
        Path("../scratch").rglob(f"{paired_oeid}_registered_to_pair.h5")
    )
    paired_reg_emf_fn = next(
        (
            output_dir.parent.parent.rglob(
                f"{paired_oeid}_registered_to_pair_episodic_mean_fov.h5"
            )
        )
    )

    # Just to get alpha and beta for the experiment using the episodic mean fov paired movie
    (
        _,
        alpha_list,
        beta_list,
        mean_norm_mi_list,
    ) = dri.decrosstalk_roi_image_from_episodic_mean_fov(
        oeid, paired_reg_emf_fn, input_dir.parent
    )
    alpha = np.mean(alpha_list)
    beta = np.mean(beta_list)
    metadata = {
        "alpha_mean": round(alpha, 2),
        "beta_mean": round(beta, 2),
        "paired_emf": str(paired_reg_emf_fn),
    }

    # To reduce RAM usage, you can get/save the decrosstalk_data in chunks:
    chunk_size = 5000  # num of frames in each chunk

    with h5.File(input_dir / "motion_correction" / f"{oeid}_registered.h5", "r") as f:
        data_shape = f["data"].shape
    data_length = data_shape[0]
    start_frames = np.arange(0, data_length, chunk_size)
    end_frames = np.append(start_frames[1:], data_length)
    assert end_frames[-1] == data_length
    decrosstalk_fn = output_dir / f"{oeid}_decrosstalk.h5"

    # generate the decrosstalk movie with alpha and beta values calculated above
    # using the full paired registered movie
    chunk_no = 0
    for start_frame, end_frame in zip(start_frames, end_frames):
        with h5.File(paired_oeid_reg_to_oeid_full_fn, "r") as f:
            paired_data = f["data"][start_frame:end_frame]
        with h5.File(input_dir / "motion_correction" / f"{oeid}_registered.h5", "r") as f:
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


def run_decrosstalk(
    input_dir: Path,
    output_dir: Path,
    oeid: str,
    paired_oeid: str,
    start_time: dt,
    num_frames: int = 1000,
) -> None:
    """Runs paired plane registration and decrosstalk for a given pair of experiments

    Parameters
    ----------
    input_dir: Path
        path to input data
    output_dir: Path
        path to output data
    oeid: str
        ophys experiment id
    paired_oeid: str
        ophys experiment id of paired experiment
    start_time: dt
        start time of decrosstalk processing
    num_frames: int, optional
        number of frames to process, default is 1000
    """
    logging.info("Running paired plane registration...")
    # create cached registered to pair movie for each pair

    # create the EMF of the registered to pair movie from cache

    # create EMF of the self registered movies
    ppr.episodic_mean_fov(
        input_dir / "motion_correction" / f"{oeid}_registered.h5", output_dir
    )
    logging.info("Creating movie...")
    # run decrosstalk
    decrosstalk = decrosstalk_roi_movie(
        oeid, paired_oeid, input_dir, output_dir, start_time
    )
    ppr.episodic_mean_fov(decrosstalk, output_dir, num_frames=num_frames, save_webm=True)


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


def run(args: argparse.Namespace) -> None:
    """Run decrosstalk processing" """
    input_dir = Path("../data/").resolve()
    output_dir = Path("../results/").resolve()
    debug = args.debug
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
    start_time_oeid1 = dt.now()
    run_decrosstalk(oeid1_input_dir, oeid1_output_dir, oeid1, oeid2, start_time_oeid1)
    start_time_oeid2 = dt.now()
    run_decrosstalk(oeid2_input_dir, oeid2_output_dir, oeid2, oeid1, start_time_oeid2)
    (Path("../scratch/") / f"{oeid1}_registered_to_pair.h5").unlink()
    print("unlinking paired registered flies")
    (Path("../scratch/") / f"{oeid2}_registered_to_pair.h5").unlink()

    write_qc_metrics(oeid1_output_dir, oeid1)
    write_qc_metrics(oeid2_output_dir, oeid2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    input_dir = Path("../data/").resolve()
    nf_output = next(input_dir.glob("output"), "")
    single_output = next(input_dir.glob("single.txt"), "")
    if nf_output:
        sys.exit()
    elif single_output:
        write_data_process(
            {},
            "no data",
            "no data",
            "no data",
            dt.now(),
            dt.now(),
        )
        sys.exit()
    else:
        run(args)
