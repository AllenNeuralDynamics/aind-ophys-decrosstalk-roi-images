""" top level run script """
from pathlib import Path
import h5py as h5
import logging
import numpy as np
import paired_plane_registration as ppr
import decrosstalk_roi_image as dri
import shutil
import json
from aind_data_schema.core.processing import Processing
from aind_data_schema.core.processing import DataProcess, ProcessName, PipelineProcess
from typing import Union
from datetime import datetime as dt
import argparse
import os

def write_output_metadata(
    metadata: dict,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    url: str,
    start_date_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    input_fp: str
        path to data input
    output_fp: str
        path to data output
    url: str
        url to code repository
    """
    original_proc_file = input_fp.parent
    with open(original_proc_file / "processing.json", "r") as f:
        proc_data = json.load(f)
    prev_processing = Processing(**proc_data)
    processing = Processing(
        processing_pipeline=PipelineProcess(
            processor_full_name="Multplane Ophys Processing Pipeline",
            pipeline_url=os.getenv("PIPELINE_URL", ""),
            pipeline_version=os.getenv("PIPELINE_VERSION", ""),
            data_processes=[
                DataProcess(
                    name=ProcessName.VIDEO_PLANE_DECROSSTALK,
                    software_version=os.getenv("VERSION", ""),
                    start_date_time=start_date_time,  # TODO: Add actual dt
                    end_date_time=dt.now(),  # TODO: Add actual dt
                    input_location=str(input_fp),
                    output_location=str(output_fp),
                    code_url=(url),
                    parameters=metadata,
                )
            ],
        )
    )
    prev_processing.processing_pipeline.data_processes.append(processing.processing_pipeline.data_processes[0])
    prev_processing.write_standard_file(output_directory=Path(output_fp).parent)


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
    oeid_mt = input_dir / "motion_correction" / f"{oeid}_motion_transform.csv"
    paired_oeid_reg_to_oeid_full_fn = next(
        Path("../scratch").glob(f"{paired_oeid}_registered_to_pair.h5")
    )
    paired_reg_emf_fn = next(
        (output_dir.parent.parent / paired_oeid / "decrosstalk").glob(
            f"{paired_oeid}_registered_to_pair_episodic_mean_fov.h5"
        )
    )

    ## Just to get alpha and beta for the experiment using the episodic mean fov paired movie
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

    ## To reduce RAM usage, you can get/save the decrosstalk_data in chunks:
    chunk_size = 5000  # num of frames in each chunk

    with h5.File(input_dir / "motion_correction" / f"{oeid}_registered.h5", "r") as f:
        data_shape = f["data"].shape
    data_length = data_shape[0]
    start_frames = np.arange(0, data_length, chunk_size)
    end_frames = np.append(start_frames[1:], data_length)
    assert end_frames[-1] == data_length
    decrosstalk_fn = output_dir / f"{oeid}_decrosstalk.h5"

    # generate the decrosstalk movie with alpha and beta values calculated above using the full paired registered movie
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
    write_output_metadata(
        metadata,
        input_dir / "motion_correction" / f"{oeid}_registered.h5",
        decrosstalk_fn,
        "https://github.com/AllenNeuralDynamics/aind-ophys-decrosstalk-roi-images/tree/development",
        start_time,
    )
    return decrosstalk_fn


def prepare_cached_paired_plane_movies(
    oeid1: str, oeid2: str, input_dir: Path, non_rigid: bool = True, block_size=[128, 128]
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

    Returns
    -------
    h5_file: Path
        path to cached paired plane movie
    """
    h5_file = input_dir / "motion_correction" / f"{oeid1}.h5"
    oeid_mt = (
        input_dir.parent / oeid2 / "motion_correction" / f"{oeid2}_motion_transform.csv"
    )
    if not h5_file.is_file():
        h5_file = input_dir.parent.parent / f"{oeid1}_registered.h5"
        print(f"~~~~~~~~~~~~~~~~~~~~~~~{h5_file}")
    transform_df = ppr.get_s2p_motion_transform(oeid_mt)
    return ppr.paired_plane_cached_movie(
        h5_file, transform_df, non_rigid=non_rigid, block_size=block_size
    )


def get_processing_json(input_dir: Path) -> dict:
    """
    Get processing json from input directory

    Parameters
    ----------
    input_dir: Path
        path to input data

    Returns
    -------
    pj: dict
        processing json
    """
    processing_json = next(input_dir.glob("*/processing.json"))
    with open(processing_json, "r") as f:
        pj = json.load(f)
    return pj


def get_block_size(input_dir: Path) -> list:
    """get image dimensions from processing json

    Parameters
    ----------
    processing: dict
        processing json

    Returns
    -------
    block_size: list
        block size of image
    """
    processing_json = get_processing_json(input_dir)
    try:
        block_size = processing_json["processing_pipeline"]["data_processes"][0][
            "parameters"
        ]["suite2p_args"]["block_size"]
    except KeyError:
        block_size = processing_json["data_processes"][0]["parameters"]["suite2p_args"][
            "block_size"
        ]
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
    processing_json = get_processing_json(input_dir)
    try:
        nonrigid = processing_json["processing_pipeline"]["data_processes"][0][
            "parameters"
        ]["suite2p_args"]["nonrigid"]
    except KeyError:
        nonrigid = processing_json["data_processes"][0]["parameters"]["suite2p_args"][
            "nonrigid"
        ]
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
    """
    logging.info(f"Running paired plane registration...")
    # create cached registered to pair movie for each pair

    # create the EMF of the registered to pair movie from cache

    # create EMF of the self registered movies
    ppr.episodic_mean_fov(
        input_dir / "motion_correction" / f"{oeid}_registered.h5", output_dir
    )
    logging.info(f"Creating movie...")
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
    oeid: Path
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        num_frames = 300
    else:
        num_frames = 1000
    input_dir = Path("../data/").resolve()
    output_dir = Path("../results/").resolve()
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
        oeid1, oeid2, oeid1_input_dir, non_rigid=non_rigid, block_size=block_size
    )
    oeid2_reg_to_oeid1_motion_filepath = prepare_cached_paired_plane_movies(
        oeid2, oeid1, oeid2_input_dir, non_rigid=non_rigid, block_size=block_size
    )
    processing_json = get_processing_json(oeid1_input_dir)
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
