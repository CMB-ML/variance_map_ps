# Overview

This is a temporary repository and results will be integrated into CMB-ML.

The purpose is to demonstrate an implementation of noise generation which is (1) spatially anisotropic, (2) spatially correlated, (3) field correlated, all at the target output resolution.

We had (1) implemented for CMB-ML v0.1.0.

The next step, (2), is derived from the method given in Appendix A of Planck CO Revisited (Ghosh, S., et al.: A&A, 688, A54).

The final step, (3), has yet to be fully implemented.

# Method of Appendix A of Planck CO Revisited (Temperature only)

The method of Appendix A of Planck CO Revisited was written for a more complicated situation. It turns out that the process is simple. The method for temperature maps only follows. For each frequency, separately:

1. Obtain variance maps from Planck's observations 
    - E.g. In HFI_SkyMap_100_2048_R3.01_full.fits, the field associated with II_COV
2. Obtain a noise simulation map from Planck
    - E.g. ffp10_noise_100_full_map_mc_00000.fits
3. Change the resolution of the variance map to the target resolution (from 1.)
4. Per pixel, get the standard deviation (square root of the values) (from 3.)
5. Generate Gaussian noise ($\sim \mathcal{N}(0,1)$) per pixel
6. Scale that noise (from 5.) with the standard deviation (from 4.) - this is spatially anisotropic white noise (1).
7. Determine the power spectrum of the noise simulation map, $C_{\ell}^{(n)}$ (from 2.)
8. Determine the $a_{\ell m}$'s and power spectrum of the anisotropic white noise map $C_\ell^{(w)}$ (from 6.)
    - We currently replace the $C_0^{(w)}$ and $C_1^{(w)}$ with the mean of the $C_\ell^{(w)}$'s. This prevents wildly high correlation imposed by the filter.
9. Create the filter, $\sqrt{C_{\ell}^{(n)}/ \; C_{\ell}^{(w)}}$ (from 7., 8.)
10. Smooth the filter with a simple box-car (we currently use length 2). (from step 9.)
11. Apply the filter to the $a_{\ell m}$'s (from 10.)
12. Convert the $a_{\ell m}$'s back to pixel space.

# Contents of Note

Three notebooks should be looked at:
- Spatially anisotropic noise (1) [variance_map_ps.ipynb](variance_map_ps.ipynb)
    - A minimal implementation has facilities for downloading required files
    - Also compares results for $N_{side}=2048$ and $N_{side}=512$
- Spatially correlated noise (2) [variance_map_ps_imp_I_downscale.ipynb](variance_map_ps_imp_I_downscale.ipynb)
    - Includes comparison to expected values from white noise
- Spatially correlated noise (2) [variance_map_ps_imp_I_multiple.ipynb](variance_map_ps_imp_I_multiple.ipynb)
    - Generates several noise realizations for comparison
- Field correlated noise (3) [variance_map_ps_imp_IQU.ipynb](variance_map_ps_imp_IQU.ipynb)
    - INCOMPLETE
    - Requires finding a square root and inverse matrix per pixel
    - May need to be optimized with PyTorch for expediency when creating simulations
        - Currently takes ~5 minutes to determine these values for all pixels
- Finding the average power spectrum of Planck Noise [planck_sims_mean_cl.ipynb](planck_sims_mean_cl.ipynb)

# TODO

- Figure out what to do for multiple maps
- Finish IQU
- Move contents to lost_in_space or CMB-ML tutorials
- Rewrite introduction to whatever final versions of the notebooks we share

# References

From [Planck Wiki](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Simulation_data):
- Noise maps can be accessed have direct links with this format:
`http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=<filename>`
- Observation maps (including half-mission maps) instead use:
`http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=<filename>`

The above links are incorporated below.

If a GUI is preferred, go to [Planck Legacy Archive](https://pla.esac.esa.int/#maps), then follow these navigation steps (direct links don't work there):
- Maps (Leftmost icon)
- Advanced Search & Map Operations (Second wide row)
- Simulated Map Search (tab at top of this section)
- Choose Noise from list of categories at left
- Period: "full" (not "Full", which is empty with all other options)
- Instrument: "HFI" or "LFI"
- Frequency: All work

In Planck 2018, I, Table 4, we find the temperature noise level for the 100 GHz detector: $1.29 \mu K_{CMB} deg$. The table also contains levels for all detectors and the polarization noise levels. We use the following values for white noise in this notebook...

TODO: Find the citation for when the noise simulations were generated. It's around here somewhere...

# Installing

Instructions were not specifically retained.

Installing (updated for torch, tqdm):
- previously, created a conda environment with healpy, jupyter, matplotlib
- `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia`
- `conda install tqdm`

