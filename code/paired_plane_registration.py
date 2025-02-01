import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aind_ophys_utils.array_utils import normalize_array
from aind_ophys_utils.video_utils import encode_video
from suite2p.registration import nonrigid

# NOTE: currently this module works in the Session level, someone may want to calculat per
# experiment
# TODO: implement per experiment level


def get_s2p_motion_transform(csv_path: Path, non_rigid: bool = True) -> pd.DataFrame:
    """Get suite2p motion transform for experiment
    Also correct for data type in nonrigid columns (from str to np.array)

    Parameters
    ----------
    csv_path : Path
        path to suite2p rigid or non-rigid motion transform
    non_rigid : bool, optional
        whether the motion transform is non-rigid, by default True

    Returns
    -------
    pandas.DataFrame
        # TODO LOW: add more context about DF

    """

    reg_df = pd.read_csv(csv_path)
    if non_rigid:
        assert "nonrigid_x" in reg_df.columns
        assert "nonrigid_y" in reg_df.columns
        if isinstance(reg_df.nonrigid_x[0], str):
            reg_df.nonrigid_x = reg_df.nonrigid_x.apply(
                lambda x: np.array(
                    [np.float32(xx) for xx in x.split("[")[1].split("]")[0].split(",")]
                )
            )
            reg_df.nonrigid_y = reg_df.nonrigid_y.apply(
                lambda y: np.array(
                    [np.float32(yy) for yy in y.split("[")[1].split("]")[0].split(",")]
                )
            )

    return reg_df


def generate_mean_episodic_fov_pairings_registered_frames(
    input_dir,
    oeids: tuple,
    save_dir: Path = Path("../results/"),
    max_num_epochs=10,
    num_frames=1000,
    non_rigid=True,
):
    """Generate mean episodic FOVs registered to the paired experiment for both experiments in the
    pair. Create a full paired registered movie in a temp directory to get torn down at end of
    capsule processing

    Parameters
    ----------
    input_dir: str
        session directory
    oeids : tuple of strings
        experiment ids for pair 1 and pair 2 from mesoscope file splitting queue input json
    save_dir : Path, optional
        path to save registered frames, by default None
    max_num_epochs : int
        Maximum number of epochs to calculate the mean FOV image for
    num_frames : int
        Number of frames to average to calculate the mean FOV image
    non_rigid: bool
        whether the motion transform is non-rigid, by default True

    Returns
    -------
    Path to temporary paired registered movie
    """
    assert len(oeids) == 2
    if not save_dir.is_dir():
        raise (ValueError("save_dir must be a directory"))
    paired_plane_data = {}
    for oeid in oeids:
        paired_plane_data[oeid] = {}
        # if (input_dir / oeid / f"{oeid}.h5").is_file():
        paired_plane_data[oeid]["raw_movie_fp"] = input_dir / oeid / f"{oeid}.h5"
        # else:
        #     paired_plane_data[oeid]["raw_movie_fp"] = input_dir / f"{oeid}.h5"
    # to tie the paired suite2p rigid motion transform to the correct oeid
    paired_plane_data[oeids[0]]["paired_motion_df"] = get_s2p_motion_transform(
        input_dir / oeids[-1] / f"{oeids[-1]}_motion_transform.csv", non_rigid=non_rigid
    )
    paired_plane_data[oeids[-1]]["paired_motion_df"] = get_s2p_motion_transform(
        input_dir / oeids[0] / f"{oeids[0]}_motion_transform.csv", non_rigid=non_rigid
    )
    shutil.copy(
        input_dir / oeids[0] / f"{oeids[0]}_motion_transform.csv",
        save_dir,
    )
    shutil.copy(
        input_dir / oeids[-1] / f"{oeids[-1]}_motion_transform.csv",
        save_dir,
    )

    # Not all the experiments have general_info_for_ophys_experiment_id (e.g., pilot data)
    # But since this is about paired plane registration, they all must have
    # motion_corrected_movie_filepath

    for k in paired_plane_data.keys():
        with h5py.File(paired_plane_data[k]["raw_movie_fp"], "r") as f:
            data_length = f["data"].shape[0]
            assert data_length == paired_plane_data[k]["paired_motion_df"].shape[0]

            num_epochs = min(max_num_epochs, data_length // num_frames)
            epoch_interval = data_length // (
                num_epochs + 1
            )  # +1 to avoid the very first frame (about half of each epoch)
            num_frames = min(num_frames, epoch_interval)
            start_frames = [
                num_frames // 2 + i * epoch_interval for i in range(num_epochs)
            ]
            assert start_frames[-1] + num_frames < data_length

            # Calculate the mean FOV image for each epoch, after paired plane registration
            mean_fov = np.zeros((num_epochs, f["data"].shape[1], f["data"].shape[2]))
            for i in range(num_epochs):
                start_frame = start_frames[i]
                epoch_data = f["data"][start_frame : start_frame + num_frames]
                y = paired_plane_data[k]["paired_motion_df"]["y"].values[
                    start_frame : start_frame + num_frames
                ]
                x = paired_plane_data[k]["paired_motion_df"]["x"].values[
                    start_frame : start_frame + num_frames
                ]
                if "nonrigid_x" in paired_plane_data[k]["paired_motion_df"]:
                    nonrigid_y = paired_plane_data[k]["paired_motion_df"][
                        "nonrigid_y"
                    ].values[start_frame : start_frame + num_frames]
                    nonrigid_x = paired_plane_data[k]["paired_motion_df"][
                        "nonrigid_x"
                    ].values[start_frame : start_frame + num_frames]
                    # from default parameters:
                    # TODO: read from a file
                    Ly1 = 512
                    Lx1 = 512
                    block_size = (128, 128)
                    blocks = nonrigid.make_blocks(Ly=Ly1, Lx=Lx1, block_size=block_size)
                epoch_registered = epoch_data.copy()
                for frame, dy, dx in zip(epoch_registered, y, x):
                    frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)
                if "nonrigid_x" in paired_plane_data[k]["paired_motion_df"]:
                    epoch_registered = nonrigid.transform_data(
                        epoch_registered,
                        yblock=blocks[0],
                        xblock=blocks[1],
                        nblocks=blocks[2],
                        ymax1=np.stack(nonrigid_y, axis=0),
                        xmax1=np.stack(nonrigid_x, axis=0),
                        bilinear=True,
                    )

                mean_fov[i] = np.mean(epoch_registered, axis=0)

            with h5py.File(save_dir / f"{k}_paired_reg_mean_episodic_fov.h5", "w") as f:
                f.create_dataset("data", data=mean_fov)


def paired_plane_cached_movie(
    h5_file: Path,
    reg_df: pd.DataFrame,
    tmp_dir: Path = Path("../scratch/"),
    chunk_size: int = 5000,
    non_rigid: bool = True,
    block_size: list = [128, 128],
) -> Path:
    """Transform frames and save to h5 file

    Parameters
    ----------
    h5_file : Path
        movie path to the non-motion corrected movie
    reg_df : pandas.DataFrame
        registration DataFrame (from the csv file) for the pair associated with the non-motion
        corrected movie
    tmp_dir : Path, optional
        temporary directory to store the registered movie, by default Path("../scratch/")
    chunk_size : int, optional
        number of frames to process at a time, by default 5000
    non_rigid : bool, optional
        whether the motion transform is non-rigid, by default True
    block_size : list, optional
        block size for non-rigid registration, by default [128, 128]

    Returns
    -------
    Path to temporary h5 file
    """
    print("~~~~~~~~~~~")
    print(h5_file.is_file())
    with h5py.File(h5_file, "r") as f:
        print(f"~~~~~~~~~H5 File{h5_file}")
        data_length = f["data"].shape[0]
        start_frames = np.arange(0, data_length, chunk_size)
        end_frames = np.append(start_frames[1:], data_length)
        # assert that frames and shifts are the same length
        y_shifts = reg_df["y"].values
        x_shifts = reg_df["x"].values
        if non_rigid:
            assert "nonrigid_x" in reg_df.columns
            assert "nonrigid_y" in reg_df.columns
            Ly = f["data"].shape[1]
            Lx = f["data"].shape[2]
            block_size = (block_size[0], block_size[1])
            blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size)
            ymax1 = np.vstack(reg_df.nonrigid_y.values)
            xmax1 = np.vstack(reg_df.nonrigid_x.values)
        print(data_length)
        print(len(y_shifts))
        print(len(x_shifts))
        assert data_length == len(y_shifts) == len(x_shifts)
        for start_frame, end_frame in zip(start_frames, end_frames):
            r_frames = np.zeros_like(f["data"][start_frame:end_frame], dtype=np.int16)
            frame_group = f["data"][start_frame:end_frame]
            x_shift_group = x_shifts[start_frame:end_frame]
            y_shift_group = y_shifts[start_frame:end_frame]
            xmax1_group = xmax1[start_frame:end_frame]
            ymax1_group = ymax1[start_frame:end_frame]
            for frame_index, (frame, dy, dx) in enumerate(
                zip(frame_group, y_shift_group, x_shift_group)
            ):
                r_frames[frame_index] = shift_frame(frame=frame, dy=dy, dx=dx)
            if non_rigid:
                r_frames = nonrigid.transform_data(
                    r_frames,
                    yblock=blocks[0],
                    xblock=blocks[1],
                    nblocks=blocks[2],
                    ymax1=ymax1_group,
                    xmax1=xmax1_group,
                    bilinear=True,
                )
                # uint16 is preferrable, but suite2p default seems like int16, and other files are
                # in int16
                # Suite2p codes also need to be changed to work with uint16 (e.g., using
                # nonrigid_uint16 branch)
                # njit pre-defined data type
                # TODO: change all processing into uint16 in the future

            # save r_frames
            temp_path = tmp_dir / f'{h5_file.name.split(".")[0]}_registered_to_pair.h5'
            with h5py.File(temp_path, "a") as cache_f:
                if "data" not in cache_f.keys():
                    cache_f.create_dataset(
                        "data",
                        (data_length, 512, 512),
                        maxshape=(None, 512, 512),
                        chunks=(1000, 512, 512),
                    )
                    cache_f["data"].resize(
                        (f["data"].shape[0] + r_frames.shape[0]), axis=0
                    )
                    cache_f["data"][start_frame:end_frame] = r_frames
                else:
                    cache_f["data"].resize(
                        (f["data"].shape[0] + r_frames.shape[0]), axis=0
                    )
                    cache_f["data"][start_frame:end_frame] = r_frames
    return temp_path


def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Returns frame, shifted by dy and dx

    Parameters
    ----------
    frame: Ly x Lx
    dy: int
        vertical shift amount
    dx: int
        horizontal shift amount

    Returns
    -------
    frame_shifted: Ly x Lx
        The shifted frame

    # TODO: move to utils

    """
    assert frame.ndim == 2, "frame must be 2D"
    assert np.abs(dy) < frame.shape[0], "dy must be less than frame height"
    assert np.abs(dx) < frame.shape[1], "dx must be less than frame width"

    return np.roll(frame, (-dy, -dx), axis=(0, 1))


###############################################################################
# PLOTS
# Documented less, just quick QC plots
###############################################################################


def fig_paired_planes_registered_projections(projections_dict: dict):
    """Plot registered projections of paired planes

    Parameters
    ----------
    projections_dict : dict
        dictionary containing the following keys:
        - plane1_raw
        - plane1_original_registered
        - plane1_paired_registered
        - plane2_raw
        - plane2_original_registered
        - plane2_paired_registered
    """
    # get 99 percentile of all images to set vmax
    images = [v for k, v in projections_dict.items()]
    max_val = np.percentile(np.concatenate(images), 99.9)

    keys = [
        "plane1_raw",
        "plane1_original_registered",
        "plane1_paired_registered",
        "plane2_raw",
        "plane2_original_registered",
        "plane2_paired_registered",
    ]

    # check that all keys are in dict
    assert all(
        [k in projections_dict.keys() for k in keys]
    ), "missing keys in projections_dict"

    # subplots show all images
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    ax[0, 0].imshow(projections_dict["plane1_raw"], cmap="gray", vmax=max_val)
    ax[0, 0].set_title("plane 1 raw")
    ax[0, 1].imshow(
        projections_dict["plane1_original_registered"], cmap="gray", vmax=max_val
    )
    ax[0, 1].set_title("plane 1 orignal registered")
    ax[0, 2].imshow(
        projections_dict["plane1_paired_registered"], cmap="gray", vmax=max_val
    )
    ax[0, 2].set_title("plane 1 registered to plane 2")

    ax[1, 0].imshow(projections_dict["plane2_raw"], cmap="gray")
    ax[1, 0].set_title("plane 2 raw")
    ax[1, 1].imshow(
        projections_dict["plane2_original_registered"], cmap="gray", vmax=max_val
    )
    ax[1, 1].set_title("plane 2 original registered")
    ax[1, 2].imshow(
        projections_dict["plane2_paired_registered"], cmap="gray", vmax=max_val
    )
    ax[1, 2].set_title("plane 2 registered to plane 1")

    # turn off axis labels
    for i in range(2):
        for j in range(3):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.show()


def histogram_shifts(expt1_shifts, expt2_shifts):
    """
    Plot histograms of shifts for each plane

    Parameters
    ----------
    expt1_shifts : pd.DataFrame
        DataFrame containing the shifts for plane 1
    expt2_shifts : pd.DataFrame
        DataFrame containing the shifts for plane 2
    """
    e1y, e1x = expt1_shifts.y, expt1_shifts.x
    e2y, e2x = expt2_shifts.y, expt2_shifts.x

    fig, ax = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True)
    ax[0, 0].hist(e1y, bins=25)
    ax[0, 0].set_title("plane 1 y shifts")
    ax[0, 1].hist(e1x, bins=25)
    ax[0, 1].set_title("plane 1 x shifts")
    ax[1, 0].hist(e2y, bins=25)
    ax[1, 0].set_title("plane 2 y shifts")
    ax[1, 1].hist(e2x, bins=25)
    ax[1, 1].set_title("plane 2 x shifts")
    plt.tight_layout()
    plt.show()


def projection_process(data: np.ndarray, projection: str = "max") -> np.ndarray:
    """

    Parameters
    ----------
    data: np.ndarray
        nframes x nrows x ncols, uint16
    projection: str
        "max" or "avg"

    Returns
    -------
    proj: np.ndarray
        nrows x ncols, uint8

    """
    if projection == "max":
        proj = np.max(data, axis=0)
    elif projection == "avg":
        proj = np.mean(data, axis=0)
    else:
        raise ValueError('projection can be "max" or "avg" not ' f"{projection}")
    return normalize_array(proj)


def episodic_mean_fov(
    movie_fn, save_dir, max_num_epochs=10, num_frames=1000, save_webm=False
):
    """
    Calculate the mean FOV image for each epoch in a movie and saves it to an h5 file
    Parameters
    ----------
    movie_fn : str or Path
        Path to the movie file
        h5 file
    save_dir: Path
        Directory to store the movie
    max_num_epochs : int
        Maximum number of epochs to calculate the mean FOV image for
    num_frames : int
        Number of frames to average to calculate the mean FOV image
    save_webm: bool
        Save webm or not

    Returns
    -------
    Path to the mean FOV image h5 file
    """
    # Load the movie
    if not str(movie_fn).endswith(".h5"):
        raise ValueError("movie_fn must be an h5 file")
    if not save_dir.is_dir():
        raise (ValueError("save_dir must be a directory"))
    with h5py.File(movie_fn, "r") as f:
        data_length = f["data"].shape[0]
        num_epochs = min(max_num_epochs, data_length // num_frames)
        epoch_interval = data_length // (
            num_epochs + 1
        )  # +1 to avoid the very first frame (about half of each epoch)
        num_frames = min(num_frames, epoch_interval)
        start_frames = [num_frames // 2 + i * epoch_interval for i in range(num_epochs)]
        assert start_frames[-1] + num_frames < data_length
        avg_img = projection_process(f["data"], projection="avg")
        max_img = projection_process(f["data"], projection="max")
        # Calculate the mean FOV image for each epoch
        mean_fov = np.zeros((num_epochs, f["data"].shape[1], f["data"].shape[2]))
        for i in range(num_epochs):
            start_frame = start_frames[i]
            mean_fov[i] = projection_process(
                f["data"][start_frame : start_frame + num_frames], projection="avg"
            )
            # mean_fov[i] = np.mean(
            #     f["data"][start_frame : start_frame + num_frames], axis=0
            # )
    save_path = save_dir / f"{movie_fn.stem}_episodic_mean_fov.h5"
    webm_path = save_dir / f"{movie_fn.stem}_episodic_mean_fov.webm"
    for im, dest in zip(
        [avg_img, max_img],
        [
            save_dir / f"{movie_fn.stem}_avg_img.png",
            save_dir / f"{movie_fn.stem}_max_img.png",
        ],
    ):
        plt.imsave(dest, im, cmap="gray")

    with h5py.File(save_path, "w") as f:
        f.create_dataset("data", data=mean_fov)
    if save_webm:
        norm_array = normalize_array(mean_fov)
        encode_video(norm_array, str(webm_path), 5)
    return save_path
