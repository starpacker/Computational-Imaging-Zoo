# main.py
```python
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Any, Sequence

def open_image(path) -> torch.Tensor:
    """Load image (single or multi-frame) → torch tensor [n_frames, h, w]."""

    img = Image.open(path)
    frames = []

    for i in range(getattr(img, "n_frames", 1)):
        img.seek(i)
        arr = np.array(img).astype(np.float32)
        frames.append(torch.from_numpy(arr))

    return torch.stack(frames, dim=0)  # [n, h, w]


def save_image(path, data: torch.Tensor) -> None:
    """Save a torch tensor as multi-frame (if n>1) image.

    data: [n_frames, h, w] torch tensor (float or uint8)
    """
    
    data = data.detach().cpu().numpy()
    print(data)
    # convert float -> uint8
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)

    frames = [Image.fromarray(f) for f in data]
    
    # 保存为多帧TIFF图像
    if len(frames) == 1:
        # 如果只有一个帧，直接保存
        frames[0].save(path, format='TIFF')
    else:
        # 如果有多个帧，保存为多页TIFF
        frames[0].save(
            path,
            format='TIFF',
            save_all=True,
            append_images=frames[1:],
            loop=0  # 无限循环（对于动画TIFF）
        )


def sum_psf(a, axis: Sequence[int] = (1, 2), keepdims: bool = True,):
    return torch.sum(a, dim=axis, keepdim=keepdims)


def make_circle_mask(radius: int) -> np.ndarray:
    """Create a circular mask.

    Args:
        radius: The radius of the circle.

    Returns:
        A circular mask.
    """
    y, x = np.ogrid[: 2 * radius, : 2 * radius]
    circle_mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius**2
    return circle_mask.astype(np.float32)  # [2 * radius, 2 * radius]


def crop_and_apply_circle_mask(
    data: np.ndarray,  # [k, n, n]
    center,
    radius: int,
) -> np.ndarray:
    """Crop the image and apply a circular mask.

    Args:
        data: The image data.
        center: The center of the circle.
        radius: The radius of the circle.

    Returns:
        The cropped and masked image.
    """
    center = [int(x) for x in center]
    radius = int(radius)
    circle_mask = np.expand_dims(make_circle_mask(radius), axis=0)  # [1, 2 * radius, 2 * radius]
    sub_O = data[
        :, center[0] - radius : center[0] + radius, center[1] - radius : center[1] + radius
    ]  # [k, 2 * radius, 2 * radius]
    return sub_O * circle_mask  # [k, 2 * radius, 2 * radius]

from inverse_algorithm import reconstruct as inverse_reconstruction

def main(
    img, psf, out,
    lens_radius,
    num_iters=10,
    normalize_psf=False,
    lens_center=None,
):
    img = open_image(img)
    psf = open_image(psf)

    if normalize_psf:
        psf = psf / sum_psf(psf)

    recon = inverse_reconstruction(
        img, psf,
        recon_kwargs=dict(num_iter=num_iters)
    )
    if lens_center == None:
        lens_center =(img.shape[-2] // 2, img.shape[-1] // 2)
    cropped = crop_and_apply_circle_mask(
        recon, lens_center, lens_radius
    )
    # print(cropped.shape)
    save_image(out, cropped)
```

# inverse.py
```python
import torch
import numpy as np
from pathlib import Path
from time import time
from typing import Optional, Union


ArrayLike = Union[np.ndarray, torch.Tensor]


# ----------------------------------------------------
#  Utility FFT wrappers (simplify their interface)
# ----------------------------------------------------
def rfft2(a: torch.Tensor):
    return torch.fft.rfft2(a, dim=(-2, -1))


def irfft2(a: torch.Tensor, s=None):
    return torch.fft.irfft2(a, s=s, dim=(-2, -1))


def flip(a: torch.Tensor):
    return torch.flip(a, dims=(-2, -1))


# ----------------------------------------------------
#        Richardson–Lucy Single Iteration
# ----------------------------------------------------
def rl_step(
    data: torch.Tensor,      # [k, n, n]
    image: torch.Tensor,     # [1, n, n]
    psf_fft: torch.Tensor,   # [k, n, n/2+1]
    psft_fft: torch.Tensor   # [k, n, n/2+1]
) -> torch.Tensor:
    """One RL iteration."""

    # denominator = PSF * data (convolution)
    conv = irfft2(psf_fft * rfft2(data))  # [k, n, n]
    denominator = conv.sum(dim=0, keepdim=True)  # [1, n, n]

    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-12)

    # RL error term
    img_err = image / denominator  # [1, n, n]

    # data * conv( error, reverse_psf )
    update = irfft2(rfft2(img_err) * psft_fft)  # [k, n, n]

    return data * torch.fft.fftshift(update, dim=(-1, -2))


# ----------------------------------------------------
#      Core RL Loop (independent of any class)
# ----------------------------------------------------
def richardson_lucy_core(
    image: torch.Tensor,   # [1, n, n]
    psf: torch.Tensor,     # [k, n, n]
    num_iter: int
) -> torch.Tensor:

    psf_fft = rfft2(psf)             # [k, n, n/2+1]
    psft_fft = rfft2(flip(psf))      # [k, n, n/2+1]

    # initialization
    data = torch.full_like(psf, fill_value=0.5)

    for _ in range(num_iter):
        data = rl_step(data, image, psf_fft, psft_fft)

    return data


# ----------------------------------------------------
#            High-level RL wrapper
# ----------------------------------------------------
def richardson_lucy(
    image: ArrayLike,
    psf: ArrayLike,
    num_iter: int = 10,
) -> torch.Tensor:

    # convert to torch
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image.astype(np.float32))
    if isinstance(psf, np.ndarray):
        psf = torch.from_numpy(psf.astype(np.float32))

    # Ensure correct shape
    if image.ndim == 2:
        image = image.unsqueeze(0)  # [1, h, w]
    if psf.ndim == 2:
        psf = psf.unsqueeze(0)      # [1, h, w]

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    psf = psf.to(device)

    result = richardson_lucy_core(image, psf, num_iter)

    return result.cpu()


# ----------------------------------------------------
#     Reconstruction entry point (public API)
# ----------------------------------------------------
def reconstruct(
    image: Union[ArrayLike, str, Path],
    psf: Union[ArrayLike, str, Path],
    recon_kwargs: Optional[dict] = None,
) -> torch.Tensor:

    if recon_kwargs is None:
        recon_kwargs = {}

    # If paths provided, they must be loaded by user before calling this
    if isinstance(image, (str, Path)):
        raise ValueError("`image` is path, but reconstruct() expects tensor or numpy array.")
    if isinstance(psf, (str, Path)):
        raise ValueError("`psf` is path, but reconstruct() expects tensor or numpy array.")

    t0 = time()
    result = richardson_lucy(image, psf, **recon_kwargs)
    print(f"Reconstruction took {time() - t0:.3f} seconds.")

    return result

```