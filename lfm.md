# main.py
```python
# main.py
import os
import numpy as np
from modules.data_loader import load_lfm_data
from modules.geometry_calculator import calculate_geometry_and_operators
from modules.image_corrector import correct_image
from modules.volume_reconstructor import reconstruct_volume
from modules.visualizer import plot_error_convergence, plot_comparison

# --- Configuration ---
data_path = './fly-muscles-GFP' # Update path as needed
depth_range = [-300, 300]
depth_step = 150
new_spacing_px = 15
super_res_factor = 5
lanczos_window_size = 4
filter_flag = True
# niter = 100
niter = 1
use_gpu = True # Set to False if Cupy is not available

# --- Main Execution ---
try:
    # 1. Load Data
    print("Loading data...")
    white_image, config_dict, raw_lenslet_image = load_lfm_data(data_path)

    # 2. Calculate Geometry and Operators
    print("Calculating geometry and operators...")
    Camera, LensletCenters, Resolution, LensletGridModel, NewLensletGridModel, H, Ht = \
        calculate_geometry_and_operators(config_dict, white_image, depth_range, depth_step, super_res_factor, new_spacing_px)

    # 3. Correct Image
    print("Correcting image...")
    LFimage, imgSize, volumeSize = correct_image(raw_lenslet_image, white_image, LensletCenters, Resolution, LensletGridModel, NewLensletGridModel, use_gpu)

    # 4. Reconstruct Volume
    print("Starting reconstruction...")
    recon_volume_np, error_metrics = reconstruct_volume(
        LFimage, H, Ht, LensletCenters, Resolution, Camera, imgSize, np.array(volumeSize[:2]), volumeSize,
        niter=niter, filter_flag=filter_flag, lanczos_window_size=lanczos_window_size, use_gpu=use_gpu
    )

    # 5. Visualize Results
    print("Visualizing results...")
    plot_error_convergence(error_metrics)
    center_slice = recon_volume_np[:, :, recon_volume_np.shape[2] // 2]
    plot_comparison(raw_lenslet_image, center_slice)

    print("Reconstruction complete. Outputs saved.")

except Exception as e:
    print(f"An error occurred during execution: {e}")
    import traceback
    traceback.print_exc()

```

#  modules/data_loader.py
```python
# modules/data_loader.py
import os
import tifffile
import yaml 

def load_lfm_data(data_path: str) -> tuple:
    """
    Loads the required LFM data files.

    Args:
        data_path (str): Path to the directory containing 'calib.tif', 'config.yaml', and the image file.

    Returns:
        tuple: A tuple containing (white_image, config_dict, raw_lenslet_image).
               white_image: Calibration image (numpy array).
               config_dict: Configuration parameters from YAML file (dict).
               raw_lenslet_image: Raw light field image (numpy array).
    """
    fname_calib = os.path.join(data_path, 'calib.tif')
    fname_config = os.path.join(data_path, 'config.yaml')
    fname_img = os.path.join(data_path, 'example_fly.tif')

    if not os.path.exists(fname_calib) or not os.path.exists(fname_config) or not os.path.exists(fname_img):
        raise FileNotFoundError(f"Required data files not found in {data_path}. Check README.")

    white_image = tifffile.imread(fname_calib)
    with open(fname_config, 'r') as f:
        config_dict = yaml.safe_load(f)
    raw_lenslet_image = tifffile.imread(fname_img)

    return white_image, config_dict, raw_lenslet_image
```

# modules/geometry_calculator.py
```python
import numpy as np
from pyolaf.geometry import LFM_computeGeometryParameters, LFM_setCameraParams_v2
from pyolaf.lf import LFM_computeLFMatrixOperators
def calculate_geometry_and_operators(config_dict: dict, white_image: np.ndarray, depth_range: list, depth_step: float, super_res_factor: int, new_spacing_px: int):
    """
    Calculates camera parameters, geometry, and forward/backward projection operators.

    Args:
        config_dict (dict): Configuration dictionary loaded from YAML.
        white_image (np.ndarray): Calibration image.
        depth_range (list): [min_depth, max_depth] in mm.
        depth_step (float): Depth step in mm.
        super_res_factor (int): Super-resolution factor.
        new_spacing_px (int): New lenslet spacing in pixels.

    Returns:
        tuple: (Camera, LensletCenters, Resolution, LensletGridModel, NewLensletGridModel, H, Ht)
    """
    Camera = LFM_setCameraParams_v2(config_dict, new_spacing_px)
    LensletCenters, Resolution, LensletGridModel, NewLensletGridModel = \
        LFM_computeGeometryParameters(
            Camera, white_image, depth_range, depth_step, super_res_factor, False)
    H, Ht = LFM_computeLFMatrixOperators(Camera, Resolution, LensletCenters)

    return Camera, LensletCenters, Resolution, LensletGridModel, NewLensletGridModel, H, Ht
```

# modules/image_corrector.py
```python
import numpy as np
import cupy
from pyolaf.transform import LFM_retrieveTransformation, format_transform, get_transformed_shape, transform_img

def correct_image(lenslet_image: np.ndarray, white_image: np.ndarray, lenslet_centers: dict, resolution: dict, lenslet_grid_model: np.ndarray, new_lenslet_grid_model: np.ndarray, use_gpu: bool = True):
    """
    Applies geometric correction and normalization to the input lenslet image.

    Args:
        lenslet_image (np.ndarray): Raw lenslet image to be corrected.
        white_image (np.ndarray): Calibration image.
        lenslet_centers (dict): Output from geometry calculation.
        resolution (dict): Output from geometry calculation.
        lenslet_grid_model (np.ndarray): Output from geometry calculation.
        new_lenslet_grid_model (np.ndarray): Output from geometry calculation.
        use_gpu (bool): Whether to use Cupy for computation.

    Returns:
        tuple: (corrected_normalized_gpu_array, transformed_shape, volume_size)
    """
    xp = cupy if use_gpu else np

    # Obtain transformation
    FixAll = LFM_retrieveTransformation(lenslet_grid_model, new_lenslet_grid_model)
    trans = format_transform(FixAll)
    imgSize = get_transformed_shape(white_image.shape, trans)
    imgSize = imgSize + (1 - np.remainder(imgSize, 2)) # Ensure even size

    texSize = np.ceil(np.multiply(imgSize, resolution['texScaleFactor'])).astype('int32')
    texSize = texSize + (1 - np.remainder(texSize, 2)) # Ensure even size

    ndepths = len(resolution['depths'])
    volumeSize = np.append(texSize, ndepths).astype('int32')

    # Setup and correct image
    img = xp.array(lenslet_image, dtype='float32')
    new = transform_img(img, trans, lenslet_centers['offset'])
    newnorm = (new - xp.min(new)) / (xp.max(new) - xp.min(new))
    LFimage = newnorm

    return LFimage, imgSize, volumeSize
```

# modules/visualizer.py
```python
import matplotlib.pyplot as plt
import numpy as np

def normalize_to_uint8(img):
    """Helper function to normalize an image to uint8."""
    img = img - img.min()
    img = img / (img.max() + 1e-8)  # Avoid division by zero
    return (img * 255).astype(np.uint8)

def plot_error_convergence(error_metrics, save_path='deconv_error_convergence.png'):
    """Plots the error metrics over iterations."""
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(error_metrics) + 1), error_metrics, 'o-', label='LF MAE')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Absolute Error (vs onesForward)')
    plt.title('Deconvolution Error Convergence')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved error convergence plot to '{save_path}'")
    plt.close() # Close figure to free memory

def plot_comparison(input_img, output_slice, save_path='comparison.png'):
    """Plots and saves a comparison between input and output slices."""
    input_norm = normalize_to_uint8(input_img)
    output_norm = normalize_to_uint8(output_slice)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_norm, cmap='gray')
    axes[0].set_title('Input (Lenslet Image)')
    axes[0].axis('off')

    axes[1].imshow(output_norm, cmap='gray')
    axes[1].set_title('Output (Reconstructed Center Slice)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to '{save_path}'")
    plt.close() # Close figure to free memory
```

# modules/volume_reconstructor.py
```python
import numpy as np
import cupy
from time import time
from pyolaf.project import LFM_forwardProject, LFM_backwardProject
from pyolaf.aliasing import lanczosfft, LFM_computeDepthAdaptiveWidth
from cupy.fft import fftshift, ifft2, fft2

def reconstruct_volume(LFimage, H, Ht, lenslet_centers, resolution, camera, imgSize, texSize, volumeSize, niter=100, filter_flag=True, lanczos_window_size=4, use_gpu=True):
    """
    Performs iterative deconvolution (Richardson-Lucy) to reconstruct the volume.

    Args:
        LFimage: The corrected light field image (Cupy/Numpy array).
        H, Ht: Forward and backward projection operators.
        lenslet_centers (dict): Geometry information.
        resolution (dict): Resolution information.
        camera (dict): Camera information.
        imgSize (tuple): Size of the transformed image.
        texSize (tuple): Size of the texture space.
        volumeSize (tuple): Size of the final volume.
        niter (int): Number of iterations.
        filter_flag (bool): Whether to apply anti-aliasing filter.
        lanczos_window_size (int): Window size for the filter.
        use_gpu (bool): Whether to use Cupy.

    Returns:
        tuple: (reconstructed_volume_numpy, error_metrics)
    """
    xp = cupy if use_gpu else np
    if use_gpu:
        mempool = cupy.get_default_memory_pool()

    crange = camera['range'] # Assuming 'range' is in resolution dict
    initVolume = xp.ones(volumeSize, dtype='float32')

    print('Precomputing partial forward and back projections...')
    onesForward = LFM_forwardProject(H, initVolume, lenslet_centers, resolution, imgSize, crange, step=8)
    onesBack = LFM_backwardProject(Ht, onesForward, lenslet_centers, resolution, texSize, crange, step=8)

    print('Run deconvolution on image...')
    LFimage = xp.asarray(LFimage)
    reconVolume = xp.asarray(xp.copy(initVolume))
    error_metrics = []
    t1 = time()

    # Build anti-aliasing filter kernels
    widths = LFM_computeDepthAdaptiveWidth(camera, resolution) # Note: Assumes camera is in resolution
    kernelFFT = lanczosfft(volumeSize, widths, lanczos_window_size)

    for i in range(niter):
        if i == 0:
            LFimageGuess = onesForward
        else:
            LFimageGuess = LFM_forwardProject(H, reconVolume, lenslet_centers, resolution, imgSize, crange, step=10)
        if use_gpu:
            mempool.free_all_blocks()

        errorLFimage = LFimage / LFimageGuess * onesForward
        errorLFimage[~xp.isfinite(errorLFimage)] = 0

        err_metric = xp.mean(xp.abs(errorLFimage - onesForward)).item()
        error_metrics.append(err_metric)
        print(f"Iter {i+1}/{niter}, LF error (MAE): {err_metric:.6f}")

        errorBack = LFM_backwardProject(Ht, errorLFimage, lenslet_centers, resolution, texSize, crange, step=10)
        if use_gpu:
            mempool.free_all_blocks()

        errorBack = errorBack / onesBack
        errorBack[~xp.isfinite(errorBack)] = 0

        reconVolume = reconVolume * errorBack

        if filter_flag:
            for j in range(errorBack.shape[2]):
                reconVolume[:, :, j] = xp.abs(fftshift(ifft2(kernelFFT[:,:,j] * fft2(reconVolume[:,:,j]))))

        reconVolume[~xp.isfinite(reconVolume)] = 0
        if use_gpu:
            mempool.free_all_blocks()

    t2 = time()
    print(f'Time for reconstruction: {t2 - t1:.2f} seconds')

    if use_gpu:
        reconVolume_np = cupy.asnumpy(reconVolume)
    else:
        reconVolume_np = reconVolume

    return reconVolume_np, error_metrics
```