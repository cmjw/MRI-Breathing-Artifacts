# Overview
A set of procedures to demonstrate MRI breathing artifact, binning, and reconstruction.

## Requirements
- Matplotlib
- Numpy
- H5PY for .mat file processing
- ASTRA toolbox

## Data
This program uses a sampled portion of the XCAT dataset (Segars et al. 2010) to simulate breathing motion.
The XCAT dataset should be located at ./data/sampling_300ms_compressed.mat. (not included in this repository)

## References
- Segars, W. P., Sturgeon, G. M., Ward, D. J., Ratnanather, J. T., Miller, M. I., & Tsui, B. M. W. (2010). The new XCAT series of digital phantoms for multi-modality imaging. IEEE Nuclear Science Symposium & Medical Imaging Conference, 2392–2395. https://doi.org/10.1109/NSSMIC.2010.5874215
