# Overview
A set of procedures to demonstrate MRI breathing artifact, binning, and reconstruction. 
- We simulate the breathing pattern of the patient using a 4D dataset of a simulated abdominal model.
- Using golden angle radial sampling, we have written forward and backward process routines using the ASTRA library.
- We use binning to categorize similar breathing phases to reconstruct a set of images with reduced motion artifact.
- Finally, we use the total variation noise reduction algorithm to reduce the streaking artifact in the radon domain during iterative reconstruction.
- We execute this process for each bin to produce one reconstructed image per bin.

## Requirements
- Matplotlib
- Numpy
- H5PY for .mat file processing
- ASTRA toolbox

## Data
This program uses a sampled portion of the XCAT dataset (Segars et al. 2010) to simulate breathing motion.
The XCAT dataset should be located at ./data/sampling_300ms_compressed.mat. (not included in this repository)

## Future Improvements:

- Simulate irregular breathing patterns
- 3D radial sampling of multiple slices
- Optimize selection of number of bins, and TV parameters
- Improved sampling scheme
- Temporal reconstruction algorithms - frames/slices closer in time should have smaller differences in their reconstructed images

## References
- Chu, M.-L., Chang, H.-C., Chung, H.-W., Bashir, M. R., Cai, J., Zhang, L., Sun, D., & Chen, N.-K. (2017). Free-breathing abdominal MRI improved by repeated k-t-subsampling and artifact-minimization (ReKAM). Medical Physics, 45(1), 178–190. https://doi.org/10.1002/mp.12674
- Feng, L. (2022). Golden‐angle radial MRI: Basics, advances, and applications. Journal of Magnetic Resonance Imaging, 56(1), 45–62. https://doi.org/10.1002/jmri.28187
- Kojima, S., Shinohara, H., Hashimoto, T., Hirata, M., & Ueno, E. (2015). Iterative image reconstruction that includes a total variation regularization for radial MRI. Radiological Physics and Technology, 8(2), 295–304. https://doi.org/10.1007/s12194-015-0320-7
- Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. Physica D: Nonlinear Phenomena, 60(1), 259–268. https://doi.org/10.1016/0167-2789(92)90242-F
- Segars, W. P., Sturgeon, G. M., Ward, D. J., Ratnanather, J. T., Miller, M. I., & Tsui, B. M. W. (2010). The new XCAT series of digital phantoms for multi-modality imaging. IEEE Nuclear Science Symposium & Medical Imaging Conference, 2392–2395. https://doi.org/10.1109/NSSMIC.2010.5874215