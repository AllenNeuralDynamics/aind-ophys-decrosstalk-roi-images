""" top level run script """
import decrosstalk_roi_image as dri
from pathlib import Path
import h5py as h5
import json
import logging
import shutil
import numpy as np


def decrosstalk_fov(oeid, paired_oeid, input_dir, output_dir):
    logging.info(f"Input directory, {input_dir}")
    logging.info(f"Output directory, {output_dir}")
    logging.info(f"Ophys experiment ID pairs, {oeid}, {paired_oeid}")
    output_dir = output_dir / oeid
    output_dir.mkdir(exist_ok=True)
    oeid_pj = list(input_dir.glob(f"{oeid}_processing.json"))[0]
    oeid_mt = input_dir / f"{oeid}_motion_transform.csv"
    shutil.copy(oeid_mt, output_dir)
    shutil.copy(oeid_pj, output_dir / "processing.json")
    platform_json_fp = list(input_dir.glob("*platform.json"))[0]
    with open(platform_json_fp, "r") as f:
        platform_json = json.load(f)
    try:
        pixel_size_um = platform_json["imaging_plane_groups"][0]["imaging_planes"][0]["registration"][
        "pixel_size_um"
        ]
    except KeyError:
        print(f"Could not pull pixel size from platform json, {platform_json_fp}. Using default value of 0.78um/pixel")
    paired_reg_fn = list(input_dir.glob(f"{paired_oeid}_paired_registered.h5"))[0]
    decrosstalk_data, alpha_list, beta_list, mean_norm_mi_list = dri.decrosstalk_movie_roi_image(
        oeid, paired_reg_fn, input_dir, pixel_size_um
    )
    dc_fov_dir = output_dir / 'decrosstalk_roi_images'
    if not dc_fov_dir.exists():
        dc_fov_dir.mkdir()
    decrosstalk_fn = dc_fov_dir / f'{oeid}_decrosstalk.h5'
    with h5.File(decrosstalk_fn, 'w') as f:
        f.create_dataset('data', data=decrosstalk_data)
        f.create_dataset('alpha_list', data=alpha_list)
        f.create_dataset('beta_list', data=beta_list)
        f.create_dataset('mean_norm_mi_list', data=mean_norm_mi_list)

    ## Just to get alpha and beta for the experiment:
    _, alpha_list, beta_list, mean_norm_mi_list = dri.decrosstalk_movie_roi_image(oeid, paired_reg_fn,input_dir return_recon=False)
    alpha = np.mean(alpha_list)
    beta = np.mean(beta_list)

    ## To reduce RAM usage, you can get/save the decrosstalk_data in chunks:
    chunk_size = 5000 # num of frames in each chunk

    with h5.File(input_dir / f"{oeid}_registered.h5", 'r') as f:
        data_shape = f['data'].shape
    data_length = data_shape[0]
    num_chunks = int(np.ceil(data_length / chunk_size))
    start_frames = np.arange(0, data_length, chunk_size)
    end_frames = np.append(start_frames[1:], data_length)

    dc_fov_dir = output_dir / 'decrosstalk_roi_images'
    if not dc_fov_dir.exists():
        dc_fov_dir.mkdir()
    decrosstalk_fn = dc_fov_dir / f'{oeid}_decrosstalk.h5'

    i = 0
    for start_frame, end_frame in zip(start_frames, end_frames):
        with h5.File(input_dir / f"{oeid}_registered.h5", 'r') as f:
            signal_data = f['data'][start_frame:end_frame]
        with h5.File(paired_reg_fn, 'r') as f:
            paired_data = f['data'][start_frame:end_frame]
        recon_signal_data = np.zeros_like(signal_data)
        for j in range(signal_data.shape[0]):
            recon_signal_data[j, :, :] = dri.apply_mixing_matrix(alpha, beta, signal_data[j, :, :], paired_data[j, :, :])[0]

        if i == 0:
            with h5.File(decrosstalk_fn, 'w') as f:
                f.create_dataset('data', data=recon_signal_data, maxshape=(None, data_shape[1], data_shape[2]))
                f.create_dataset('alpha_list', data=alpha_list)
                f.create_dataset('beta_list', data=beta_list)
                f.create_dataset('mean_norm_mi_list', data=mean_norm_mi_list)
        else:
            with h5.File(decrosstalk_fn, 'a') as f:
                f['data'].resize((f['data'].shape[0] + recon_signal_data.shape[0]), axis=0)
                f['data'][start_frame:end_frame] = recon_signal_data
        i += 1

def run():
    """basic run function"""
    input_dir = Path("../data/").resolve()
    output_dir = Path("../results/").resolve()
    paired_directories = list(input_dir.glob("*"))
    for i in paired_directories:
        oeid1, oeid2 = str(i.name).split("_")[0], str(i.name).split("_")[-1]
        logging.info(f"Processing pairs, Pair_1, {oeid1}, Pair_2, {oeid2}")
        decrosstalk_fov(oeid1, oeid2, i, output_dir)
        decrosstalk_fov(oeid2, oeid1, i, output_dir)


if __name__ == "__main__":
    run()
