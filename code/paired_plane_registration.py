from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from suite2p.registration import nonrigid
import shutil

# NOTE: currently this module works in the Session level, someone may want to calculat per experiment
# TODO: implement per experiment level


def get_s2p_motion_transform(csv_path: Path) -> pd.DataFrame:
    """Get suite2p motion transform for experiment
    Also correct for data type in nonrigid columns (from str to np.array)

    Parameters
    ----------
    csv_path : Path
        path to suite2p rigid motion transform

    Returns
    -------
    pandas.DataFrame
        # TODO LOW: add more context about DF

    """

    reg_df = pd.read_csv(csv_path)
    if "nonrigid_x" in reg_df.columns:
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
    num_frames_to_avg=1000,
):
    """Generate mean episodic FOVs registered to the paired experiment for both experiments in the pair
    Create a full paired registered movie in a temp directory to get torn down at end of capsule processing

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
    num_frames_to_avg : int
        Number of frames to average to calculate the mean FOV image
    
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
        paired_plane_data[oeid]["raw_movie_fp"] = input_dir / oeid / f"{oeid}.h5"
    # to tie the paired suite2p rigid motion transform to the correct oeid
    paired_plane_data[oeids[0]]["paired_motion_df"] = get_s2p_motion_transform(
        input_dir / oeids[-1] / f"{oeids[-1]}_motion_transform.csv"
    )
    paired_plane_data[oeids[-1]]["paired_motion_df"] = get_s2p_motion_transform(
        input_dir / oeids[0] / f"{oeids[0]}_motion_transform.csv"
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
    # But since this is about paired plane registration, they all must have motion_corrected_movie_filepath

    for k in paired_plane_data.keys():
        with h5py.File(paired_plane_data[k]["raw_movie_fp"], "r") as f:
            data_length = f["data"].shape[0]
            assert data_length == paired_plane_data[k]["paired_motion_df"].shape[0]

            num_epochs = min(max_num_epochs, data_length // num_frames_to_avg)
            epoch_interval = data_length // (
                num_epochs + 1
            )  # +1 to avoid the very first frame (about half of each epoch)
            num_frames = min(num_frames_to_avg, epoch_interval)
            start_frames = [num_frames // 2 + i * epoch_interval for i in range(num_epochs)]
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
                    nonrigid_y = paired_plane_data[k]["paired_motion_df"]["nonrigid_y"].values[
                        start_frame : start_frame + num_frames
                    ]
                    nonrigid_x = paired_plane_data[k]["paired_motion_df"]["nonrigid_x"].values[
                        start_frame : start_frame + num_frames
                    ]
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

def paired_plane_cached_movie(h5_file: Path,
                              reg_df: pd.DataFrame,
                              tmp_dir: Path = Path("../scratch/"),
                              chunk_size=500
                              )
    """Transform frames and save to h5 file

    Parameters
    ----------
    h5_file : Path
        movie path to the non-motion corrected movie
    reg_df : pandas.DataFrame
        registration DataFrame (from the csv file) for the pair associated with the non-motion corrected movie
    save_path : Path, optional
        path to save transformed h5 file, by default None
    return_rframes : bool, optional
        return registered frames, by default False

    Returns
    -------
    Path to temporary h5 file
    """

    with h5py.File(h5_file, "r") as f:
        data_length = f['data'].shape[0]
        no_of_chunks = data_length // chunk_size
        # assert that frames and shifts are the same length
        y_shifts = reg_df['y'].values
        x_shifts = reg_df['x'].values
        run_nonrigid = False
        if 'nonrigid_x' in reg_df.columns:
            run_nonrigid = True
            # from default parameters:
            # TODO: read this from the log file
            Ly = 512
            Lx = 512
            block_size = (128, 128)
            blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size)
            ymax1 = np.vstack(reg_df.nonrigid_y.values)
            xmax1 = np.vstack(reg_df.nonrigid_x.values)
        assert len(data_length) == len(y_shifts) == len(x_shifts)
        if run_nonrigid:
            assert len(data_length) == ymax1.shape[0] == xmax1.shape[0]
        for i in range(no_of_chunks):
            if i == no_of_chunks:
                r_frames = np.zeros_like(f['data'][i * chunk_size:])
            else:
                r_frames = np.zeros_like(f['data'][i * chunk_size: i * (chunk_size - 1)])
            for i, (frame, dy, dx) in enumerate(zip(data_length, y_shifts, x_shifts)):
                r_frames[i] = shift_frame(frame=frame, dy=dy, dx=dx)
            if run_nonrigid:
                r_frames = nonrigid.transform_data(r_frames, yblock=blocks[0], xblock=blocks[1], nblocks=blocks[2],
                                                ymax1=ymax1, xmax1=xmax1, bilinear=True)
                # uint16 is preferrable, but suite2p default seems like int16, and other files are in int16
                # Suite2p codes also need to be changed to work with uint16 (e.g., using nonrigid_uint16 branch)
                # njit pre-defined data type
                # TODO: change all processing into uint16 in the future
                r_frames = r_frames.astype(np.int16)

            # save r_frames
            temp_path = tmp_dir / f'{h5_file.name.split(".")[0]}_registered_to_pair.h5'
            with h5py.File(temp_path, 'w') as f:
                f.create_dataset('data', data=r_frames)
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
    assert all([k in projections_dict.keys() for k in keys]), "missing keys in projections_dict"

    # subplots show all images
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    ax[0, 0].imshow(projections_dict["plane1_raw"], cmap="gray", vmax=max_val)
    ax[0, 0].set_title("plane 1 raw")
    ax[0, 1].imshow(projections_dict["plane1_original_registered"], cmap="gray", vmax=max_val)
    ax[0, 1].set_title("plane 1 orignal registered")
    ax[0, 2].imshow(projections_dict["plane1_paired_registered"], cmap="gray", vmax=max_val)
    ax[0, 2].set_title("plane 1 registered to plane 2")

    ax[1, 0].imshow(projections_dict["plane2_raw"], cmap="gray")
    ax[1, 0].set_title("plane 2 raw")
    ax[1, 1].imshow(projections_dict["plane2_original_registered"], cmap="gray", vmax=max_val)
    ax[1, 1].set_title("plane 2 original registered")
    ax[1, 2].imshow(projections_dict["plane2_paired_registered"], cmap="gray", vmax=max_val)
    ax[1, 2].set_title("plane 2 registered to plane 1")

    # turn off axis labels
    for i in range(2):
        for j in range(3):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.show()


def histogram_shifts(expt1_shifts, expt2_shifts):
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


def episodic_mean_fov(movie_fn, save_dir, max_num_epochs=10, num_frames_to_avg=1000):
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
    num_frames_to_avg : int
        Number of frames to average to calculate the mean FOV image

    """
    # Load the movie
    if not str(movie_fn).endswith(".h5"):
        raise ValueError("movie_fn must be an h5 file")
    if not save_dir.is_dir():
        raise (ValueError("save_dir must be a directory"))
    with h5py.File(movie_fn, "r") as f:
        data_length = f["data"].shape[0]
        num_epochs = min(max_num_epochs, data_length // num_frames_to_avg)
        epoch_interval = data_length // (
            num_epochs + 1
        )  # +1 to avoid the very first frame (about half of each epoch)
        num_frames = min(num_frames_to_avg, epoch_interval)
        start_frames = [num_frames // 2 + i * epoch_interval for i in range(num_epochs)]
        assert start_frames[-1] + num_frames < data_length

        # Calculate the mean FOV image for each epoch
        mean_fov = np.zeros((num_epochs, f["data"].shape[1], f["data"].shape[2]))
        for i in range(num_epochs):
            start_frame = start_frames[i]
            mean_fov[i] = np.mean(f["data"][start_frame : start_frame + num_frames_to_avg], axis=0)
    save_path = save_dir / f"{movie_fn.stem}_mean_fov.h5"
    with h5py.File(save_path, "w") as f:
        f.create_dataset("data", data=mean_fov)