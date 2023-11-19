""" top level run script """
from pathlib import Path
import h5py as h5
import logging
import shutil
import numpy as np
import paired_plane_registration as ppr
import decrosstalk_roi_image as dri
import pandas as pd

def decrosstalk_roim(oeid, paired_oeid, input_dir, output_dir):
    logging.info(f"Input directory, {input_dir}")
    logging.info(f"Output directory, {output_dir}")
    logging.info(f"Ophys experiment ID pairs, {oeid}, {paired_oeid}")
    output_dir = output_dir / oeid
    output_dir.mkdir(exist_ok=True)
    oeid_pj = list(input_dir.glob(f"{oeid}_processing.json"))[0]
    oeid_mt = input_dir / f"{oeid}_motion_transform.csv"
    shutil.copy(oeid_mt, output_dir)
    shutil.copy(oeid_pj, output_dir / "processing.json")
    paired_reg_fn = list(input_dir.glob(f"{paired_oeid}_paired_reg_mean_episodic_fov.h5"))[0]

    ## Just to get alpha and beta for the experiment:
    _, alpha_list, beta_list, mean_norm_mi_list = dri.decrosstalk_roi_image_from_episodic_mean_fov(
        oeid, paired_reg_fn, input_dir
    )
    alpha = np.mean(alpha_list)
    beta = np.mean(beta_list)

    ## To reduce RAM usage, you can get/save the decrosstalk_data in chunks:
    chunk_size = 5000  # num of frames in each chunk

    with h5.File(input_dir / f"{oeid}_registered.h5", "r") as f:
        data_shape = f["data"].shape
    data_length = data_shape[0]
    start_frames = np.arange(0, data_length, chunk_size)
    end_frames = np.append(start_frames[1:], data_length)

    decrosstalk_fn = output_dir / f"{oeid}_decrosstalk.h5"

    i = 0
    with h5.File(paired_reg_fn, "r") as f:
        paired_data = f["data"][()]
    for start_frame, end_frame in zip(start_frames, end_frames):
        with h5.File(input_dir / f"{oeid}_registered.h5", "r") as f:
            signal_data = f["data"][start_frame:end_frame]
        recon_signal_data = np.zeros_like(signal_data)
        for j in range(signal_data.shape[0]):
            recon_signal_data[j, :, :] = dri.apply_mixing_matrix(
                alpha, beta, signal_data[j, :, :], paired_data[i, :, :]
            )[0]
        if i == 0:
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
        i += 1

def prepare_cached_paired_plane_movies(oeid1, oeid2, input_dir):
    h5_file = input_dir / f"{oeid1}.h5"
    oeid_mt = input_dir / f"{oeid2}_motion_transform.csv"
    transform_df = pd.read_csv(oeid_mt)
    return ppr.paired_plane_cached_movie(h5_file, transform_df)

def run():
    """basic run function"""
    input_dir = Path("../data/").resolve()
    output_dir = Path("../results/").resolve()
    paired_directories = list(input_dir.glob("*"))
    for i in paired_directories:
        oeid1, oeid2 = str(i.name).split("_")[0], str(i.name).split("_")[-1]
        logging.info(f"Processing pairs, Pair_1, {oeid1}, Pair_2, {oeid2}")
        logging.info(f"Running paired plane registration...")
        ppr.generate_mean_episodic_fov_pairings_registered_frames(
            i,
            (oeid1, oeid2),
            save_dir=output_dir,
        )
        oeid1_paired_reg = ppr.paired_plane_cached_movie(oeid1, oeid2, i)
        oeid2_paired_reg = ppr.paired_plane_cached_movie(oeid2, oeid1, i)
        logging.info(f"Creating movie...")
        decrosstalk_roim(oeid1, oeid2, oeid2_paired_reg, i, output_dir)
        decrosstalk_roim(oeid2, oeid1, oeid1_paired_reg, i, output_dir)


if __name__ == "__main__":
    run()
