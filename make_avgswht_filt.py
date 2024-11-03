import logging
import sys
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pysm3.units as u

from cmbml.utils.handle_data import (
    get_planck_obs_data, 
    get_planck_noise_data, 
    get_planck_hm_data, 
    get_map_dtype
    )
from cmbml.utils.fits_inspection import get_field_unit

# from system_config import ASSETS_DIRECTORY as ASSETS_DIRECTORY

DATA_ROOT = "/shared/data/Assets"
ASSETS_DIRECTORY = f"{DATA_ROOT}/Planck/"


# Logging was helpful when debugging my handle_data module
logging.basicConfig(
    level=logging.WARNING,  # If DEBUG, there's a bunch of PySM3 and Matplotlib stuff
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

CENTER_FREQS = {
    30: 28.4 * u.GHz,      # Value from Planck DeltaBandpassTable & Planck 2018 I, Table 4
    44: 44.1 * u.GHz,      # Value from Planck DeltaBandpassTable & Planck 2018 I, Table 4
    70: 70.4 * u.GHz,      # Value from Planck DeltaBandpassTable & Planck 2018 I, Table 4
    100: 100.89 * u.GHz,   # Value from Planck DeltaBandpassTable
    143: 142.876 * u.GHz,  # Value from Planck DeltaBandpassTable
    217: 221.156 * u.GHz,  # Value from Planck DeltaBandpassTable
    353: 357.5 * u.GHz,    # Value from Planck DeltaBandpassTable
    545: 555.2 * u.GHz,    # Value from Planck DeltaBandpassTable
    857: 866.8 * u.GHz,    # Value from Planck DeltaBandpassTable
}

def get_xxcov_field_num(detector, field_str):
    if detector not in [30, 44, 70, 100, 143, 217, 353, 545, 857]:
        raise ValueError(f"Detector {detector} not recognized")
    field_str = field_str.lower()
    lower_field_nums = dict(ii=4, iq=5, iu=6, qq=7, qu=8, uu=9)
    upper_field_nums = dict(ii=2)  # These detectors only have intensity data
    if detector in [545, 857]:
        if field_str not in upper_field_nums.keys():
            raise ValueError(f"Field {field_str} not available for detector {detector}")
        res = upper_field_nums[field_str]
    else:
        if field_str not in ['ii', 'iq', 'iu', 'qq', 'qu', 'uu']:
            raise ValueError(f"Field {field_str} not available for detector {detector}")
        res = lower_field_nums[field_str]
    return res

def _change_variance_map_resolution(m, nside_out):
    # For variance maps, because statistics
    power = 2

    # From PySM3 template.py's read_map function, with minimal alteration (added 'power'):
    m_dtype = get_map_dtype(m)
    nside_in = hp.get_nside(m)
    if nside_out < nside_in:  # do downgrading in double precision, per healpy instructions
        m = hp.ud_grade(m.astype(np.float64), power=power, nside_out=nside_out)
    elif nside_out > nside_in:
        m = hp.ud_grade(m, power=power, nside_out=nside_out)
    m = m.astype(m_dtype, copy=False)
    # End of used portion

    return m

def get_unit_from_str(unit_str):
    if unit_str in ['uK_CMB']:
        return u.uK_CMB
    elif unit_str in ['Kcmb', 'K_CMB']:
        return u.K_CMB
    elif unit_str in ['MJy/sr']:
        return u.MJy / u.sr
    else:
        raise ValueError(f"Unit {unit_str} not recognized")

def get_sqrt_unit_from_str(unit_str):
    if unit_str in ['Kcmb^2', '(K_CMB)^2', 'K_CMB^2']:
        return u.K_CMB
    elif unit_str in ['uK_CMB^2', '(uK_CMB)^2']:
        return u.uK_CMB
    elif unit_str in ['(MJy/sr)^2', '(Mjy/sr)^2', 'MJy/sr^2']:
        return u.MJy / u.sr
    else:
        raise ValueError(f"Unit {unit_str} not recognized")
    
def get_scale_map(det, nside_out):
    obs_fn = get_planck_obs_data(detector=det, assets_directory=ASSETS_DIRECTORY)
    use_field = get_xxcov_field_num(det, 'II')
    II_cov_map = hp.read_map(obs_fn, hdu=1, field=use_field)
    II_cov_map_512 = _change_variance_map_resolution(II_cov_map, nside_out)
    scale_map = np.sqrt(II_cov_map_512)

    var_map_unit = get_field_unit(obs_fn, hdu=1, field_idx=use_field)
    var_map_unit = get_sqrt_unit_from_str(var_map_unit)
    scale_map = scale_map * var_map_unit
    scale_map = u.Quantity(scale_map, unit=var_map_unit)

    return scale_map

def make_tgt_noise_params(det, n_sims):
    data = np.load(f"noise_models/noise_model_{det}GHz.npz")

    src_mean_ps     = data['mean_ps']
    src_components  = data['components']
    src_variance    = data['variance']

    src_mean_maps   = data['maps_mean']
    src_sd_maps     = data['maps_sd']
    src_map_unit    = data['maps_unit']

    num_components = len(src_variance)

    std_devs = np.sqrt(src_variance)

    if n_sims == 1:
        reduced_samples = np.random.normal(0, std_devs, (num_components,))
        # Reconstruct power spectra in log10 space
        tgt_log_ps = reduced_samples @ src_components + src_mean_ps
        # Convert out of log10 space
        tgt_cls = 10**tgt_log_ps
    else:
        reduced_samples = np.random.normal(0, std_devs, (n_sims, num_components))
        # Reconstruct power spectra in log10 space
        tgt_log_ps = reduced_samples @ src_components + src_mean_ps
        # Convert out of log10 space
        tgt_cls = 10**tgt_log_ps

    src_map_unit = get_unit_from_str(src_map_unit)

    return tgt_cls, src_mean_maps, src_sd_maps, src_map_unit

def make_filter_noise_map(detector, n_sims, seed=42, nside_out=512, return_dict=False):
    # Set parameters
    np.random.seed(seed)
    lmax = 3*nside_out-1
    rng = np.random.default_rng(seed)

    # Get target noise parameters
    tgt_cl, tgt_dist_mean, tgt_dist_sd, tgt_unit = make_tgt_noise_params(detector, n_sims=n_sims)

    tgt_cl = tgt_cl * tgt_unit**2
    min_dist_t = (tgt_dist_mean - 3*tgt_dist_sd)              * tgt_unit  # outside this range, adjust monopole
    max_dist_t = (tgt_dist_mean + 3*tgt_dist_sd)              * tgt_unit

    # Get target means and sign
    tgt_means = np.random.normal(tgt_dist_mean, tgt_dist_sd, size=(n_sims, tgt_dist_mean.size)) * tgt_unit
    tgt_signs = np.sign(tgt_means)

    # Make aniso white noise maps
    scale_map = get_scale_map(detector, nside_out)
    wht_nse_map = rng.normal(size=(n_sims, scale_map.size)) * scale_map

    # Check for unit consistency (both target and white should be in source units, either K_CMB or MJy/sr)
    assert tgt_unit == wht_nse_map.unit, f"Stopping. tgt_unit is {tgt_unit}, wht_nse_map.unit is {wht_nse_map.unit}"

    # Reverse sign of aniso white noise map if needed
    wht_means  = wht_nse_map.mean(axis=1)
    wht_signs  = np.sign(wht_means).reshape(tgt_signs.shape)
    flip_signs = wht_signs * tgt_signs

    wht_nse_map = wht_nse_map * flip_signs

    # if return_dict:
    #     res = dict(
    #         tgt_cls=tgt_cl,
    #         tgt_means=tgt_means,
    #         wht_maps=wht_nse_map,
    #         wht_cls=[],
    #         wht_means=wht_nse_map.mean(axis=1),
    #         filters=[],
    #         filtered_maps=[],
    #         wht_avg=None,
    #         filt_avg=None
    #     )
    # else:
    #     res = []

    wht_avg = np.zeros(scale_map.size)
    filt_avg = np.zeros(scale_map.size)

    # Make map filter, non-vectorized because healpy (improvements welcomed!)
    for i in range(n_sims):
        wht_nse_alms  = hp.map2alm(wht_nse_map[i], lmax=lmax)
        wht_nse_cl    = hp.alm2cl(wht_nse_alms) * wht_nse_map.unit**2
        map_filter    = np.sqrt(tgt_cl[i][:lmax+1] / wht_nse_cl)

        # Filter map
        filtered_alms = hp.almxfl(wht_nse_alms, map_filter)
        filtered_map  = hp.alm2map(filtered_alms, nside=nside_out) * tgt_unit
        # filtered_cl   = hp.anafast(filtered_map, lmax=lmax) * filtered_map.unit**2

        wht_avg += wht_nse_map[i] / n_sims
        filt_avg += filtered_map / n_sims

        # if return_dict:
        #     res['wht_cls'].append(wht_nse_cl)
        #     res['filters'].append(map_filter)
        #     res['filtered_maps'].append(filtered_map)
        # else:
        #     res.append(filtered_map)

    hp.write_map(f"noise_avgs/wht_avg_{detector}GHz_{n_sims}sims_seed{seed}.fits", wht_avg, overwrite=True)
    hp.write_map(f"noise_avgs/filt_avg_{detector}GHz_{n_sims}sims_seed{seed}.fits", filt_avg, overwrite=True)

    return res


res = make_filter_noise_map(217, n_sims=100, seed=42, nside_out=512, return_dict=True)
