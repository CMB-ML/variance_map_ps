from pathlib import Path

import numpy as np
import healpy as hp

import pysm3.units as u
from astropy.io import fits

from cmbml.utils.get_maps import get_planck_noise_data


DATA_ROOT = "/shared/data/Assets"
PLANCK_NOISE_DIR = f"{DATA_ROOT}/PlanckNoise/"

DETECTORS = [30, 44, 70, 100, 143, 217, 353, 545, 857]
N_NOISE_SIMS = 100
DO_FIELDS = "TQU"
LOCAL_DIR = Path("noise_avgs")
OVERWRITE_EXISTING = True
OUTPUT_NAME_TEMPLATE = "avg_noise_map_{det}_{fields}_{n}.fits"

NSIDE_LOOKUP = {30: 1024, 44: 1024, 70: 1024, 100: 2048, 143: 2048, 217: 2048, 353: 2048, 545: 2048, 857: 2048}


def get_field_unit(fits_fn, hdu=1, field_idx=0):
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
            unit_str = hdul[hdu].header[f"TUNIT{field_num}"]
        except KeyError:
            unit_str = ""
    if unit_str in ["Kcmb", "K_CMB"]:
        return u.K_CMB
    elif unit_str in ["uKcmb", "uK_CMB"]:
        return u.uK_CMB
    elif unit_str in ["MJy/sr", "Mjy/sr"]:
        return u.MJy / u.sr
    elif unit_str in ["K_CMB^2", "(K_CMB)^2", "Kcmb^2", "(Kcmb)^2"]:
        return u.K_CMB ** 2
    elif unit_str in ["uK_CMB^2", "(uK_CMB)^2", "uKcmb^2", "(uKcmb)^2"]:
        return u.uK_CMB ** 2
    elif unit_str in ["(MJy/sr)^2", "MJy/sr^2", "(Mjy/sr)^2", "Mjy/sr^2"]:
        return (u.MJy / u.sr) ** 2
    return unit_str


def make_avg_maps(det):
    if DO_FIELDS == "T" or det in [545, 857]:
        use_field = [0]
        column_names = ["TEMPERATURE"]
        avg_noise_map = np.zeros((hp.nside2npix(NSIDE_LOOKUP[det])))
    elif DO_FIELDS == "TQU":
        use_field = [0,1,2]
        column_names = ["TEMPERATURE", "Q_STOKES", "U_STOKES"]
        avg_noise_map = np.zeros((3,hp.nside2npix(NSIDE_LOOKUP[det])))

    for i in range(N_NOISE_SIMS):
        nse_fn = get_planck_noise_data(detector=det, assets_directory=PLANCK_NOISE_DIR, realization=i)
        if i == 0:
            unit = get_field_unit(nse_fn, hdu=1, field_idx=0)
        nse = hp.read_map(nse_fn, field=use_field)
        avg_noise_map += nse / N_NOISE_SIMS

    avg_noise_map = avg_noise_map * unit

    if DO_FIELDS == "T":
        column_units = [str(avg_noise_map.unit)]
    elif DO_FIELDS == "TQU":
        column_units = [*[str(avg_noise_map.unit)]*3]

    hp.write_map(out_file_path(det), avg_noise_map,
                column_names=column_names, 
                column_units=column_units,
                dtype=np.float32,
                overwrite=OVERWRITE_EXISTING)


def out_file_path(det):
    return LOCAL_DIR / OUTPUT_NAME_TEMPLATE.format(det=det, fields=DO_FIELDS, n=N_NOISE_SIMS)


def main():
    LOCAL_DIR.mkdir(exist_ok=True)
    for det in DETECTORS:
        if out_file_path(det).exists() and not OVERWRITE_EXISTING:
            print(f"File for detector {det} already exists. Skipping.")
            continue
        elif out_file_path(det).exists() and OVERWRITE_EXISTING:
            print(f"File for detector {det} already exists. Overwriting.")
        make_avg_maps(det)


if __name__ == "__main__":
    main()
