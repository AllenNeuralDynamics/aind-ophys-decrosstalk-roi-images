""" top level run script """
from pathlib import Path
import h5py as h5
import logging
import shutil
import numpy as np
import paired_plane_registration as ppr
import decrosstalk_roi_image as dri
import shutil
import json
from aind_data_schema import Processing
from aind_data_schema.processing import DataProcess, ProcessName, PipelineProcess
from typing import Union
from datetime import datetime as dt
from datetime import timezone as tz


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
    processing = Processing(
        processing_pipeline=PipelineProcess(
            processor_full_name="Multplane Ophys Processing Pipeline",
            pipeline_url="https://codeocean.allenneuraldynamics.org/capsule/5472403/tree",
            pipeline_version="0.1.0",
            data_processes=[
                DataProcess(
                    name=ProcessName.VIDEO_PLANE_DECROSSTALK,
                    software_version="0.1.0",
                    start_date_time=start_date_time,  # TODO: Add actual dt
                    end_date_time=dt.now(tz.utc),  # TODO: Add actual dt
                    input_location=str(input_fp),
                    output_location=str(output_fp),
                    code_url=(url),
                    parameters=metadata,
                )
            ],
        )
    )
    print(f"Output filepath: {output_fp}")
    with open(output_fp.parent.parent / "processing.json", "r") as f:
        proc_data = json.load(f)
    processing.write_standard_file(output_directory=Path(output_fp.parent.parent))
    with open(output_fp.parent.parent / "processing.json", "r") as f:
        dct_data = json.load(f)
    proc_data["processing_pipeline"]["data_processes"].append(
        dct_data["processing_pipeline"]["data_processes"][0]
    )
    with open(output_fp.parent.parent / "processing.json", "w") as f:
        json.dump(proc_data, f, indent=4)


def decrosstalk_roi_movie(oeid, paired_oeid, input_dir, output_dir, start_time):
    logging.info(f"Input directory, {input_dir}")
    logging.info(f"Output directory, {output_dir}")
    logging.info(f"Ophys experiment ID pairs, {oeid}, {paired_oeid}")
    oeid_mt = input_dir / f"{oeid}_motion_transform.csv"
    paired_reg_full_fn = next(Path("../scratch").glob(f"{paired_oeid}_registered_to_pair.h5"))
    print(output_dir)
    shutil.copy(oeid_mt, output_dir)
    shutil.copy(next(input_dir.glob(f"processing.json")), output_dir.parent / "processing.json")
    print(output_dir.parent.parent / paired_oeid / "decrosstalk")
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
    ) = dri.decrosstalk_roi_image_from_episodic_mean_fov(oeid, paired_reg_emf_fn, input_dir.parent)
    alpha = np.mean(alpha_list)
    beta = np.mean(beta_list)
    print(alpha)
    metadata = {
        "alpha_list": alpha_list,
        "beta_list": beta_list,
        "mean_norm_mi_list": mean_norm_mi_list,
        "alpha_mean": alpha,
        "beta_mean": beta,
        "paired_emf": str(paired_reg_emf_fn),
    }

    ## To reduce RAM usage, you can get/save the decrosstalk_data in chunks:
    chunk_size = 5000  # num of frames in each chunk

    with h5.File(input_dir / f"{oeid}_registered.h5", "r") as f:
        data_shape = f["data"].shape
    data_length = data_shape[0]
    start_frames = np.arange(0, data_length, chunk_size)
    end_frames = np.append(start_frames[1:], data_length)
    assert end_frames[-1] == data_length
    decrosstalk_fn = output_dir / f"{oeid}_decrosstalk.h5"
    # write_output_metadata(prefix= f'{oeid}_decrosstalk', metadata=metadata, input_fp=paired_reg_full_fn, output_fp=decrosstalk_fn, url='}')

    # generate the decrosstalk movie with alpha and beta values calculated above using the full paired registered movie
    chunk_no = 0
    for start_frame, end_frame in zip(start_frames, end_frames):
        with h5.File(paired_reg_full_fn, "r") as f:
            paired_data = f["data"][start_frame:end_frame]
        with h5.File(input_dir / f"{oeid}_registered.h5", "r") as f:
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
                    data=recon_signal_data.astype(dtype=np.int16),
                    maxshape=(None, data_shape[1], data_shape[2]),
                )
                f.create_dataset("alpha_list", data=alpha_list)
                f.create_dataset("beta_list", data=beta_list)
                f.create_dataset("mean_norm_mi_list", data=mean_norm_mi_list)
        else:
            with h5.File(decrosstalk_fn, "a") as f:
                f["data"].resize((f["data"].shape[0] + recon_signal_data.shape[0]), axis=0)
                f["data"][start_frame:end_frame] = recon_signal_data.astype(dtype=np.int16)
        chunk_no += 1
    write_output_metadata(
        metadata,
        input_dir / oeid / f"{oeid}_registered.h5",
        decrosstalk_fn,
        "https://github.com/AllenNeuralDynamics/aind-ophys-decrosstalk-roi-images/tree/development",
        start_time,
    )
    return decrosstalk_fn


def prepare_cached_paired_plane_movies(oeid1, oeid2, input_dir, non_rigid=True):
    h5_file = input_dir / f"{oeid1}.h5"
    oeid_mt = Path(input_dir.parent) / oeid2 / f"{oeid2}_motion_transform.csv"
    transform_df = ppr.get_s2p_motion_transform(oeid_mt)
    return ppr.paired_plane_cached_movie(h5_file, transform_df, non_rigid=non_rigid)


def check_non_rigid_registration(input_dir, oeid):
    """check processing json to see if non-rigid registration was run"""
    processing_json = next(input_dir.glob("processing.json"))
    with open(processing_json, "r") as f:
        pj = json.load(f)
    # if pj["data_processes"][0]["parameters"]["suite2p_args"].get(
    if pj["processing_pipeline"]["data_processes"][0]["parameters"]["suite2p_args"].get(
        "nonrigid", False
    ):
        return True
    else:
        return False


def run_decrosstalk(input_dir: Path, output_dir: Path, oeid: str, paired_oeid: str):
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
    """
    logging.info(f"Running paired plane registration...")
    # create cached registered to pair movie for each pair

    # create the EMF of the registered to pair movie from cache

    # create EMF of the self registered movies
    ppr.episodic_mean_fov(input_dir / f"{oeid}_registered.h5", output_dir)
    logging.info(f"Creating movie...")
    # run decrosstalk
    decrosstalk = decrosstalk_roi_movie(oeid, paired_oeid, input_dir, output_dir)
    ppr.episodic_mean_fov(decrosstalk, output_dir)


def make_output_dirs(oeid, output_dir):
    results_dir = output_dir / oeid
    results_dir.mkdir(exist_ok=True)
    results_dir = output_dir / oeid / "decrosstalk"
    results_dir.mkdir(exist_ok=True)
    return results_dir


if __name__ == "__main__":
    input_dir = Path("../data/").resolve()
    output_dir = Path("../results/").resolve()
    experiment_dirs = input_dir.glob("*/motion_correction/*")
    oeid1_input_dir = next(experiment_dirs)
    oeid2_input_dir = next(experiment_dirs)
    oeid1 = oeid1_input_dir.name
    oeid2 = oeid2_input_dir.name
    oeid1_output_dir = make_output_dirs(oeid1, output_dir)
    oeid2_output_dir = make_output_dirs(oeid2, output_dir)
    non_rigid = check_non_rigid_registration(oeid1_input_dir, oeid1)
    start_time_oeid1 = dt.now(tz.utc)
    paired_reg_oeid1 = prepare_cached_paired_plane_movies(
        oeid1, oeid2, oeid1_input_dir, non_rigid=non_rigid
    )
    start_time_oeid2 = dt.now(tz.utc)
    paired_reg_oeid2 = prepare_cached_paired_plane_movies(
        oeid2, oeid1, oeid2_input_dir, non_rigid=non_rigid
    )
    ppr.episodic_mean_fov(paired_reg_oeid1, oeid1_output_dir)
    ppr.episodic_mean_fov(paired_reg_oeid2, oeid2_output_dir)
    run_decrosstalk(oeid1_input_dir, oeid1_output_dir, oeid1, oeid2, start_time_oeid1)
    start_time_oeid2 = start_time_oeid2 - start_time_oeid1
    run_decrosstalk(oeid2_input_dir, oeid2_output_dir, oeid2, oeid1, start_time_oeid2)
    print("unlinking paired registered flies")
    (Path("../scratch/") / f"{oeid1}_registered_to_pair.h5").unlink()
    print("unlinking paired registered flies")
    (Path("../scratch/") / f"{oeid2}_registered_to_pair.h5").unlink()
