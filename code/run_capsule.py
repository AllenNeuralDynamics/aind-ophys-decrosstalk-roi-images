""" top level run script """
import decrosstalk_roi_image as dri
from pathlib import Path
import re
import h5py as h5
import json
import logging
import shutil


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
    paired_reg_fn = list(input_dir.glob(f"{paired_oeid}_paired_registered.h5")[0]
    decrosstalk_data, alpha_list, beta_list = dri.decrosstalk_movie_roi_image(
        oeid, paired_oeid, input_dir, pixel_size_um
    )
    decrosstalk_fn = output_dir / f"{oeid}_decrosstalk.h5"
    logging.info(f"Writing decrosstalk file, ExperimentID, {oeid}, File, {decrosstalk_fn}")
    with h5.File(decrosstalk_fn, "w") as f:
        f.create_dataset("data", data=decrosstalk_data)
        f.create_dataset("alpha_list", data=alpha_list)
        f.create_dataset("beta_list", data=beta_list)


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
