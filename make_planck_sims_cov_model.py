from pathlib import Path
from itertools import product
import logging

import numpy as np
import healpy as hp
from sklearn.decomposition import PCA
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from tqdm import tqdm  # For progress bars

from handle_data import get_map_dtype, get_planck_obs_data, get_planck_noise_data


logger = logging.getLogger("handle_data")
logger.setLevel(logging.DEBUG)


DATA_ROOT = "/bigdata/cmb_project/data/Assets/"
# ASSETS_DIRECTORY = f"{DATA_ROOT}/Assets/Planck/"
PLANCK_NOISE_DIR = f"{DATA_ROOT}/Planck_Noise/"

DETECTORS = [30, 44, 70, 100, 143, 217, 353, 545, 857]
N_PLANCK_SIMS = 100


def get_lmax_for_nside(nside):
    """Helper function: Max ell for a given nside; to be considered a parameter"""
    return 3 * nside - 1


def get_field_unit(fits_fn, hdu, field_idx):
    """
    Get the unit associated with a specific field from the header of the 
    specified HDU (Header Data Unit) in a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
        hdu (int): The index of the HDU.
        field_idx (int): The index of the field.

    Returns:
        str: The unit of the field.
    """
    with fits.open(fits_fn) as hdul:
        try:
            field_num = field_idx + 1
            unit = hdul[hdu].header[f"TUNIT{field_num}"]
        except KeyError:
            unit = ""
    return unit


def get_ps_data(detector):
    if detector in [30, 44, 70]:
        nside = 1024
    else:
        nside = 2048
    lmax = get_lmax_for_nside(nside)  # Defined above as 3*Nside-1
    # Getting power spectra for 100 maps at 100 GHz takes ~50 minutes
    src_cls = []
    maps_means = []
    for i in tqdm(range(N_PLANCK_SIMS)):
        src_map_fn = get_planck_noise_data(detector=detector, assets_directory=PLANCK_NOISE_DIR, realization=i, progress=True)
        t_src_map = hp.read_map(src_map_fn) * 1e6
        maps_means.append(np.mean(t_src_map))
        src_cls.append(hp.anafast(t_src_map, lmax=lmax))

    # Determine parameters for approximating the distribution of power spectra
    # Use log scaling for the power spectra; otherwise it's dominated by low ells
    log_src_cls = np.log10(src_cls)

    # We want to find the components that explain the majority of the variance
    #   We don't have enough maps to fully determine the distribution, but a full
    #   covariance matrix is overkill anyways. PCA gives a good, concise summary.
    pca = PCA().fit(log_src_cls)

    # We need the mean, the components (eigenvectors), and the variance (eigenvalues)
    #   These are surrogates for the full covariance matrix
    mean_ps = pca.mean_
    components = pca.components_  
    variance = pca.explained_variance_

    # We need the mean and standard deviation of the maps_means so we can adjust the monopole as needed
    maps_mean = np.mean(maps_means)
    maps_sd = np.std(maps_means)
    maps_unit = get_field_unit(src_map_fn, hdu=1, field_idx=0)

    # Save the results; delete the variables so we know we test loading them
    np.savez(f"noise_model_{detector}GHz.npz", 
             mean_ps=mean_ps, 
             components=components, 
             variance=variance, 
             maps_mean=maps_mean, 
             maps_sd=maps_sd,
             maps_unit=maps_unit)


if __name__ == "__main__":
    for this_det in DETECTORS:
        get_ps_data(this_det)
        logger.info(f"Done with {this_det} GHz")
    logger.info("All done")
