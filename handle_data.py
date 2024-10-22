from pathlib import Path
import requests

import numpy as np


def acquire_map_data(dest_path, source_url_template):
    """Load map data from a file, downloading it if necessary."""
    need_to_dl = False
    if not dest_path.exists():
        print(f"File {dest_path} does not exist; downloading.")
        need_to_dl = True
    elif dest_path.stat().st_size < 1024:  # If the file is less than 1KB, it's a placeholder file
        print(f"File {dest_path} has placeholder file; redownloading.")
        need_to_dl = True
    else:
        print(f"File {dest_path} exists.")
    fn = dest_path.name
    if need_to_dl:
        response = requests.get(source_url_template.format(fn=fn))
        with open(dest_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {fn}")


def format_freq(freq):
    return "{:.0f}".format(freq).zfill(3)
def format_real(real):
    return "{:.0f}".format(real).zfill(5)


def get_planck_obs_data(detector, assets_directory):
    planck_obs_fn = "{instrument}_SkyMap_{frequency}_{obs_nside}_{rev}_full.fits"
    url_template_maps = "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID={fn}"
    # Setup to get maps... this is all naming convention stuff
    if detector in [30, 44, 70]:
        instrument = "LFI"
        use_freq_str = format_freq(detector) + "-BPassCorrected"
        rev = "R3.00"
        obs_nside = 1024
    else:
        instrument = "HFI"
        use_freq_str = format_freq(detector)
        rev = "R3.01"
        obs_nside = 2048
    if detector == 353:
        use_freq_str = format_freq(detector) + "-psb"

    obs_map_fn = planck_obs_fn.format(instrument=instrument, frequency=use_freq_str, rev=rev, obs_nside=obs_nside)
    dest_path = Path(assets_directory) / obs_map_fn
    acquire_map_data(dest_path, url_template_maps)  # Download the data if it doesn't exist. Do this ahead of time.
    return dest_path


def get_planck_noise_data(detector, assets_directory, realization=0):
    """
    Get the filename for the Planck noise data, downloading it if necessary.

    Parameters
    ----------
    realization : int
        The realization number for the noise map. Default is 0. There are 300 available.
    """
    ring_cut = "full"
    planck_noise_fn_template = "ffp10_noise_{frequency}_{ring_cut}_map_mc_{realization}.fits"
    url_template_sims = "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID={fn}"

    fn = planck_noise_fn_template.format(frequency=format_freq(detector), 
                                         ring_cut=ring_cut, 
                                         realization=format_real(realization))
    fn = Path(assets_directory) / fn
    acquire_map_data(fn, url_template_sims)
    return fn


def get_map_dtype(m: np.ndarray):
    """
    Get the data type of a map in a format compatible
    with numba and mpi4py.

    Args:
        m (np.ndarray): Numpy array representing the map.

    Returns:
        np.dtype: The data type of the map.
    """
    # From PySM3 template.py's read_map function, with minimal alteration:
    dtype = m.dtype
    # numba only supports little endian
    if dtype.byteorder == ">":
        dtype = dtype.newbyteorder()
    # mpi4py has issues if the dtype is a string like ">f4"
    if dtype == np.dtype(np.float32):
        dtype = np.dtype(np.float32)
    elif dtype == np.dtype(np.float64):
        dtype = np.dtype(np.float64)
    # End of used portion
    return dtype