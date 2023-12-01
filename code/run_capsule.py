""" top level run script """
from pathlib import Path
import h5py as h5
import logging
import shutil
import numpy as np
import paired_plane_registration as ppr
import decrosstalk_roi_image as dri
import pandas as pd
import shutil
import json


def decrosstalk_roim(oeid, paired_oeid, input_dir, output_dir):
    logging.info(f"Input directory, {input_dir}")
    logging.info(f"Output directory, {output_dir}")
    logging.info(f"Ophys experiment ID pairs, {oeid}, {paired_oeid}")
    output_dir = output_dir / oeid
    oeid_mt = input_dir / f"{oeid}_motion_transform.csv"
    paired_reg_full_fn = next(Path("../scratch").glob(f"{paired_oeid}_registered_to_pair.h5"))
    shutil.copy(oeid_mt, output_dir)
    shutil.copy(next(input_dir.glob(f"processing.json")), output_dir / "processing.json")
    print(Path( output_dir.parent / paired_oeid ))
    paired_reg_emf_fn = next(
       Path( output_dir.parent / paired_oeid ).glob(f"{paired_oeid}_registered_to_pair_episodic_mean_fov.h5")
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

    ## To reduce RAM usage, you can get/save the decrosstalk_data in chunks:
    chunk_size = 5000  # num of frames in each chunk

    with h5.File(input_dir / f"{oeid}_registered.h5", "r") as f:
        data_shape = f["data"].shape
    data_length = data_shape[0]
    start_frames = np.arange(0, data_length, chunk_size)
    end_frames = np.append(start_frames[1:], data_length)
    assert end_frames[-1] == data_length
    decrosstalk_fn = output_dir / f"{oeid}_decrosstalk.h5"

    # generate the decrosstalk movie with alpha and beta values calculated above using the full paired registered movie
    chunk_no = 0
    for start_frame, end_frame in zip(start_frames, end_frames):
        with h5.File(paired_reg_full_fn, "r") as f:
            paired_data = f["data"][start_frame:end_frame]
            assert paired_data.shape[0] == end_frame[-1]
        with h5.File(input_dir / f"{oeid}_registered.h5", "r") as f:
            signal_data = f["data"][start_frame:end_frame]
        recon_signal_data = np.zeros_like(signal_data)
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
                f["data"].resize((f["data"].shape[0] + recon_signal_data.shape[0]), axis=0)
                f["data"][start_frame:end_frame] = recon_signal_data
        chunk_no += 1

    # remove the paired cache when finished


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
    if pj["data_processes"][0]["parameters"].get('nonrigid', False):
        return True
    else:
        return False



def run():
    """basic run function"""
    input_dir = Path("../data/").resolve()
    output_dir = Path("../results/").resolve()
    experiment_dirs = input_dir.glob("*/*")
    oeid1_input_dir = next(experiment_dirs)
    oeid2_input_dir = next(experiment_dirs)
    oeid1 = oeid1_input_dir.name 
    oeid2 = oeid2_input_dir.name
    logging.info(f"Processing pairs, Pair_1, {oeid1}, Pair_2, {oeid2}")
    logging.info(f"Running paired plane registration...")
    non_rigid = check_non_rigid_registration(oeid1_input_dir, oeid1)
    # create cached registered to pair movie for each pair
    oeid1_paired_reg = prepare_cached_paired_plane_movies(oeid1, oeid2, oeid1_input_dir, non_rigid=non_rigid)
    oeid2_paired_reg = prepare_cached_paired_plane_movies(oeid2, oeid1, oeid2_input_dir, non_rigid=non_rigid)
    results_dir_oeid1 = output_dir / oeid1
    results_dir_oeid2 = output_dir / oeid2
    results_dir_oeid1.mkdir(exist_ok=True)
    results_dir_oeid2.mkdir(exist_ok=True)
    # create the EMF of the registered to pair movie from cache
    ppr.episodic_mean_fov(oeid1_paired_reg, output_dir / oeid1)
    ppr.episodic_mean_fov(oeid2_paired_reg, output_dir / oeid2)
    # create EMF of the self registered movies
    ppr.episodic_mean_fov(oeid1_input_dir / f"{oeid1}_registered.h5"
    ppr.episodic_mean_fov(oeid2_input_dir / f"{oeid2}_registered.h5"
    logging.info(f"Creating movie...")
    # run decrosstalk
    decrosstalk_roim(oeid1, oeid2, oeid1_input_dir, output_dir)
    decrosstalk_roim(oeid2, oeid1, oeid2_input_dir, output_dir)
    shutil.rmtree(Path("../scratch/") / f"{oeid1}_registered_to_pair.h5")
    shutil.rmtree(Path("../scratch/") / f"{oeid2}_registered_to_pair.h5")


if __name__ == "__main__":
    run()
