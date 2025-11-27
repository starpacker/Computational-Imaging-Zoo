# utils.py
```python
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import torch

def save_model_with_required_grad(model, save_path):
    tensors_to_save = []
    
    # Traverse through model parameters and append tensors with require_grad=True to the list
    for param_name, param_tensor in model.named_parameters():
        if param_tensor.requires_grad:
            tensors_to_save.append(param_tensor)
    
    # Save the list of tensors
    torch.save(tensors_to_save, save_path)

def load_model_with_required_grad(model, load_path):
    # Load the list of tensors
    tensors_to_load = torch.load(load_path)
    
    # Traverse through model parameters and load tensors from the list
    for param_name, param_tensor in model.named_parameters():
        if param_tensor.requires_grad:
            param_tensor.data = tensors_to_load.pop(0).data

newcolors = np.vstack(
    (
        np.flipud(mpl.colormaps['magma'](np.linspace(0, 1, 128))),
        mpl.colormaps['magma'](np.linspace(0, 1, 128)),
    )
)
newcmp = ListedColormap(newcolors, name='magma_cyclic')
```

# network.py
```python
import torch
import torch.nn as nn


class G_Renderer(nn.Module):
    def __init__(
        self, in_dim=32, hidden_dim=32, num_layers=2, out_dim=1, use_layernorm=False
    ):
        super().__init__()
        act_fn = nn.ReLU()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        layers.append(act_fn)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class G_FeatureTensor(nn.Module):
    def __init__(self, x_dim, y_dim, num_feats=32, ds_factor=1):
        super().__init__()
        self.x_dim, self.y_dim = x_dim, y_dim
        x_mode, y_mode = x_dim // ds_factor, y_dim // ds_factor
        self.num_feats = num_feats

        self.data = nn.Parameter(
            2e-4 * torch.rand((x_mode, y_mode, num_feats)) - 1e-4, requires_grad=True
        )

        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()

        xs = xy * torch.tensor([x_mode, y_mode], device=xs.device).float()
        indices = xs.long()
        self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

        self.x0 = nn.Parameter(
            indices[:, 0].clamp(min=0, max=x_mode - 1), requires_grad=False
        )
        self.y0 = nn.Parameter(
            indices[:, 1].clamp(min=0, max=y_mode - 1), requires_grad=False
        )
        self.x1 = nn.Parameter((self.x0 + 1).clamp(max=x_mode - 1), requires_grad=False)
        self.y1 = nn.Parameter((self.y0 + 1).clamp(max=y_mode - 1), requires_grad=False)

    def sample(self):
        return (
            self.data[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.data[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )

    def forward(self):
        return self.sample()


class G_Tensor(G_FeatureTensor):
    def __init__(self, im_size, num_feats=32, ds_factor=1):
        super().__init__(im_size, im_size, num_feats=num_feats, ds_factor=ds_factor)
        self.renderer = G_Renderer(in_dim=num_feats)

    def forward(self):
        feats = self.sample()
        return self.renderer(feats)


class G_Tensor3D(nn.Module):
    def __init__(
        self, x_mode, y_mode, z_dim, z_min, z_max, num_feats=32, use_layernorm=False
    ):
        super().__init__()
        self.x_mode, self.y_mode, self.num_feats = x_mode, y_mode, num_feats
        self.data = nn.Parameter(
            2e-4 * torch.randn((self.x_mode, self.y_mode, self.num_feats)),
            requires_grad=True,
        )
        self.renderer = G_Renderer(in_dim=self.num_feats, use_layernorm=use_layernorm)
        self.x0 = None

        self.z_mode = z_dim
        self.z_data = nn.Parameter(
            torch.randn((self.z_mode, self.num_feats)), requires_grad=True
        )
        self.z_min = z_min
        self.z_max = z_max
        self.z_dim = z_dim

    def create_coords(self, x_dim, y_dim, x_max, y_max):
        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        xs = xy * torch.tensor([x_max, y_max], device=xs.device).float()
        indices = xs.long()
        self.x_dim, self.y_dim = x_dim, y_dim
        self.xy_coords = nn.Parameter(
            xy[None],
            requires_grad=False,
        )

        if self.x0 is not None:
            device = self.x0.device
            self.x0.data = (indices[:, 0].clamp(min=0, max=x_max - 1)).to(device)
            self.y0.data = indices[:, 1].clamp(min=0, max=y_max - 1).to(device)
            self.x1.data = (self.x0 + 1).clamp(max=x_max - 1).to(device)
            self.y1.data = (self.y0 + 1).clamp(max=y_max - 1).to(device)
            self.lerp_weights.data = (xs - indices.float()).to(device)
        else:
            self.x0 = nn.Parameter(
                indices[:, 0].clamp(min=0, max=x_max - 1),
                requires_grad=False,
            )
            self.y0 = nn.Parameter(
                indices[:, 1].clamp(min=0, max=y_max - 1),
                requires_grad=False,
            )
            self.x1 = nn.Parameter(
                (self.x0 + 1).clamp(max=x_max - 1), requires_grad=False
            )
            self.y1 = nn.Parameter(
                (self.y0 + 1).clamp(max=y_max - 1), requires_grad=False
            )
            self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

    def normalize_z(self, z):
        return (self.z_dim - 1) * (z - self.z_min) / (self.z_max - self.z_min)

    def sample(self, z):
        z = self.normalize_z(z)
        z0 = z.long().clamp(min=0, max=self.z_dim - 1)
        z1 = (z0 + 1).clamp(max=self.z_dim - 1)
        zlerp_weights = (z - z.long().float())[:, None]

        xy_feat = (
            self.data[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.data[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )
        z_feat = (
            self.z_data[z0] * (1.0 - zlerp_weights) + self.z_data[z1] * zlerp_weights
        )
        z_feat = z_feat[:, None].repeat(1, xy_feat.shape[0], 1)

        feat = xy_feat[None].repeat(z.shape[0], 1, 1) * z_feat
        # feat = torch.cat([feat, self.xy_coords], -1)

        return feat

    def forward(self, z):
        feat = self.sample(z)

        out = self.renderer(feat)
        b = z.shape[0]
        w, h = self.x_dim, self.y_dim
        out = out.view(b, 1, w, h)

        return out


class FullModel(nn.Module):
    def __init__(
        self, w, h, num_feats, x_mode, y_mode, z_min, z_max, ds_factor, use_layernorm
    ):
        super().__init__()
        self.img_real = G_Tensor3D(
            x_mode=x_mode,
            y_mode=y_mode,
            z_dim=5,
            z_min=z_min,
            z_max=z_max,
            num_feats=num_feats,
            use_layernorm=use_layernorm,
        )
        self.img_imag = G_Tensor3D(
            x_mode=x_mode,
            y_mode=y_mode,
            z_dim=5,
            z_min=z_min,
            z_max=z_max,
            num_feats=num_feats,
            use_layernorm=use_layernorm,
        )
        self.w, self.h = w, h
        self.init_scale_grids(ds_factor=ds_factor)

        # In case we want to learn the pupil function
        # self.Pupil0 = nn.Parameter(Pupil0, requires_grad=True)

        # In case we want to learn the shift of each LED
        # xF, yF = torch.meshgrid(torch.arange(-w // 2, w // 2), torch.arange(-h // 2, h // 2), indexing="ij")
        # self.xF = nn.Parameter(xF, requires_grad=False)
        # self.yF = nn.Parameter(yF, requires_grad=False)
        # self.shift = nn.Parameter(
        #    torch.zeros(n_views, 2), requires_grad=True
        # )

    def init_scale_grids(self, ds_factor):
        self.img_real.create_coords(
            x_dim=self.w // ds_factor,
            y_dim=self.h // ds_factor,
            x_max=self.img_real.x_mode,
            y_max=self.img_real.y_mode,
        )
        self.img_imag.create_coords(
            x_dim=self.w // ds_factor,
            y_dim=self.h // ds_factor,
            x_max=self.img_imag.x_mode,
            y_max=self.img_imag.y_mode,
        )
        self.ds_factor = ds_factor
        self.us_module = nn.Upsample(scale_factor=ds_factor, mode="bilinear")

    def get_shift_grid(self, led_num):
        w = self.w
        shift = self.shift[led_num].unsqueeze(1).unsqueeze(1)
        grid = torch.exp(
            -1j
            * 2
            * torch.pi
            * (self.xF[None] * shift[..., 0] + self.yF[None] * shift[..., 1])
            / w
        )
        return grid

    def forward(self, dz):
        img_real = self.img_real(dz)
        img_imag = self.img_imag(dz)
        img_real = self.us_module(img_real).squeeze(1)
        img_imag = self.us_module(img_imag).squeeze(1)

        return img_real, img_imag
```

# main.py
```python
import os
import tqdm
import mat73
import scipy.io as sio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn.functional as F

from network import FullModel

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from PIL import Image

def save_gif(imgs, filename='amplitude_stack.gif', duration=0.1):
    # 将每帧转为 uint8
    frames = []
    for img in imgs:
        img_uint8 = (img * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_uint8))
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=int(duration*1000), loop=0)



def get_sub_spectrum(img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, mag):
    O = torch.fft.fftshift(torch.fft.fft2(img_complex))
    # print(O.shape)
    to_pad_x = (spectrum_mask.shape[-2] * mag - O.shape[-2]) // 2
    to_pad_y = (spectrum_mask.shape[-1] * mag - O.shape[-1]) // 2
    O = F.pad(O, (to_pad_x, to_pad_x, to_pad_y, to_pad_y, 0, 0), "constant", 0)

    O_sub = torch.stack(
        [O[:, x_0[i] : x_1[i], y_0[i] : y_1[i]] for i in range(len(led_num))], dim=1
    )
    O_sub = O_sub * spectrum_mask
    o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
    oI_sub = torch.abs(o_sub)

    return oI_sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=15, type=int) 
    parser.add_argument("--lr_decay_step", default=6, type=int) 
    parser.add_argument("--num_feats", default=32, type=int)
    parser.add_argument("--num_modes", default=512, type=int)
    parser.add_argument("--c2f", default=False, action="store_true")
    parser.add_argument("--fit_3D", default=True, action="store_true")
    parser.add_argument("--layer_norm", default=False, action="store_true")
    parser.add_argument("--amp", default=True, action="store_true")
    parser.add_argument("--sample", default="BloodSmearTilt", type=str)
    parser.add_argument("--color", default="g", type=str)
    parser.add_argument("--is_system", default="Linux", type=str) # "Windows". "Linux"

    args = parser.parse_args()

    fit_3D = args.fit_3D
    num_epochs = args.num_epochs
    num_feats = args.num_feats
    num_modes = args.num_modes
    lr_decay_step = args.lr_decay_step
    use_c2f = args.c2f
    use_layernorm = args.layer_norm
    use_amp = args.amp
    
    sample = args.sample
    color = args.color
    is_os = args.is_system
    
    sample_list = ["BloodSmearTilt", "sheepblood", "WU1005", "Siemens"]
    color_list = ['r', 'g', 'b']
    if sample not in sample_list: 
        print("Error message: sample name is wrong.")
        print("Avaliable sample names: ['BloodSmearTilt', 'sheepblood', 'WU1005', 'Siemens'] ")
    if color not in color_list:
        print("Error message: color name is wrong.")
        print("Avaliable color names: ['r', 'g', 'b']")

    vis_dir = f"./vis/feat{num_feats}"

    
    if fit_3D:
        vis_dir += "_3D"
        os.makedirs(f"{vis_dir}/vid", exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Load data
    
    if fit_3D:
        data_struct = mat73.loadmat(f"data/{sample}/{sample}_{color}.mat")
    
        I = data_struct["I_low"].astype("float32")
    
        # Select ROI
        I = I[0:int(num_modes*2), 0:int(num_modes*2), :] 
    
        # Raw measurement sidelength
        M = I.shape[0]
        N = I.shape[1]
        ID_len = I.shape[2]
    
        # NAx NAy
        NAs = data_struct["na_calib"].astype("float32")
        NAx = NAs[:, 0]
        NAy = NAs[:, 1]
    
        # LED central wavelength
        if color == "r":
            wavelength = 0.632  # um
        elif color == "g":
            wavelength = 0.5126  # um
        elif color == "b":
            wavelength = 0.471  # um
    
        # Distance between two adjacent LEDs (unit: um)
        D_led = 4000
        # free-space k-vector
        k0 = 2 * np.pi / wavelength
        # Objective lens magnification
        mag = data_struct["mag"].astype("float32")
        # Camera pixel pitch (unit: um)
        pixel_size = data_struct["dpix_c"].astype("float32")
        # pixel size at image plane (unit: um)
        D_pixel = pixel_size / mag
        # Objective lens NA
        NA = data_struct["na_cal"].astype("float32")
        # Maximum k-value
        kmax = NA * k0
    
        # Calculate upsampliing ratio
        MAGimg = 2
        # Upsampled pixel count
        MM = int(M * MAGimg)
        NN = int(N * MAGimg)
    
        # Define spatial frequency coordinates
        Fxx1, Fyy1 = np.meshgrid(np.arange(-NN / 2, NN / 2), np.arange(-MM / 2, MM / 2))
        Fxx1 = Fxx1[0, :] / (N * D_pixel) * (2 * np.pi)
        Fyy1 = Fyy1[:, 0] / (M * D_pixel) * (2 * np.pi)
    
        # Calculate illumination NA
        u = -NAx
        v = -NAy
        NAillu = np.sqrt(u**2 + v**2)
        order = np.argsort(NAillu)
        u = u[order]
        v = v[order]
    
        # NA shift in pixel from different LED illuminations
        ledpos_true = np.zeros((ID_len, 2), dtype=int)
        count = 0
        for idx in range(ID_len):
            Fx1_temp = np.abs(Fxx1 - k0 * u[idx])
            ledpos_true[count, 0] = np.argmin(Fx1_temp)
            Fy1_temp = np.abs(Fyy1 - k0 * v[idx])
            ledpos_true[count, 1] = np.argmin(Fy1_temp)
            count += 1
        # Raw measurements
        Isum = I[:, :, order] / np.max(I)

    else:
        if sample == 'Siemens':
            data_struct = sio.loadmat(f"data/{sample}/{sample}_{color}.mat")
            MAGimg = 3

        else:
            data_struct = mat73.loadmat(f"data/{sample}/{sample}_{color}.mat")
            MAGimg = 2

            
        I = data_struct["I_low"].astype("float32")
    
        # Select ROI
        I = I[0:int(num_modes), 0:int(num_modes), :] #######################
    
        # Raw measurement sidelength
        M = I.shape[0]
        N = I.shape[1]
        ID_len = I.shape[2]
    
        # NAx NAy
        NAs = data_struct["na_calib"].astype("float32")
        NAx = NAs[:, 0]
        NAy = NAs[:, 1]
    
        # LED central wavelength
        if color == "r":
            wavelength = 0.632  # um
        elif color == "g":
            wavelength = 0.5126  # um
        elif color == "b":
            wavelength = 0.471  # um
    
        # Distance between two adjacent LEDs (unit: um)
        D_led = 4000
        # free-space k-vector
        k0 = 2 * np.pi / wavelength
        # Objective lens magnification
        mag = data_struct["mag"].astype("float32")
        # Camera pixel pitch (unit: um)
        pixel_size = data_struct["dpix_c"].astype("float32")
        # pixel size at image plane (unit: um)
        D_pixel = pixel_size / mag
        # Objective lens NA
        NA = data_struct["na_cal"].astype("float32")
        # Maximum k-value
        kmax = NA * k0
    
        # Calculate upsampliing ratio
        # MAGimg = 2
        # Upsampled pixel count
        MM = int(M * MAGimg)
        NN = int(N * MAGimg)
    
        # Define spatial frequency coordinates
        Fxx1, Fyy1 = np.meshgrid(np.arange(-NN / 2, NN / 2), np.arange(-MM / 2, MM / 2))
        Fxx1 = Fxx1[0, :] / (N * D_pixel) * (2 * np.pi)
        Fyy1 = Fyy1[:, 0] / (M * D_pixel) * (2 * np.pi)
    
        # Calculate illumination NA
        u = -NAx
        v = -NAy
        NAillu = np.sqrt(u**2 + v**2)
        order = np.argsort(NAillu)
        u = u[order]
        v = v[order]
    
        # NA shift in pixel from different LED illuminations
        ledpos_true = np.zeros((ID_len, 2), dtype=int)
        count = 0
        for idx in range(ID_len):
            Fx1_temp = np.abs(Fxx1 - k0 * u[idx])
            ledpos_true[count, 0] = np.argmin(Fx1_temp)
            Fy1_temp = np.abs(Fyy1 - k0 * v[idx])
            ledpos_true[count, 1] = np.argmin(Fy1_temp)
            count += 1
        # Raw measurements
        Isum = I[:, :, order] / np.max(I)


    # Define angular spectrum
    if sample == 'Siemens':
        kxx, kyy = np.meshgrid(Fxx1[0,:M], Fxx1[0,:N])   
    else:
        kxx, kyy = np.meshgrid(Fxx1[:M], Fxx1[:N])
    kxx, kyy = kxx - np.mean(kxx), kyy - np.mean(kyy)
    krr = np.sqrt(kxx**2 + kyy**2)
    mask_k = k0**2 - krr**2 > 0
    kzz_ampli = mask_k * np.abs(
        np.sqrt((k0**2 - krr.astype("complex64") ** 2))
    )  
    kzz_phase = np.angle(np.sqrt((k0**2 - krr.astype("complex64") ** 2)))
    kzz = kzz_ampli * np.exp(1j * kzz_phase)

    # Define Pupil support
    Fx1, Fy1 = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-M / 2, M / 2))
    Fx2 = (Fx1 / (N * D_pixel) * (2 * np.pi)) ** 2
    Fy2 = (Fy1 / (M * D_pixel) * (2 * np.pi)) ** 2
    Fxy2 = Fx2 + Fy2
    Pupil0 = np.zeros((M, N))
    Pupil0[Fxy2 <= (kmax**2)] = 1

    Pupil0 = (
        torch.from_numpy(Pupil0).view(1, 1, Pupil0.shape[0], Pupil0.shape[1]).to(device)
    )
    kzz = torch.from_numpy(kzz).to(device).unsqueeze(0)
    Isum = torch.from_numpy(Isum).to(device)

    if fit_3D:
        # Define depth of field of brightfield microscope for determine selected z-plane
        DOF = (
            0.5 / NA**2 #+ pixel_size / mag / NA
        )  # wavelength is emphrically set as 0.5 um
        # z-slice separation (emphirically set)
        delta_z = 0.8 * DOF
        # z-range
        z_max = 20.0
        z_min = -20.0
        # number of selected z-slices
        num_z = int(np.ceil((z_max - z_min) / delta_z))
        # print(num_z)
        
        # print(num_z)
    else:
        z_min = 0.0
        z_max = 1.0

    # Define LED Batch size
    led_batch_size = 1
    cur_ds = 1
    if use_c2f:
        c2f_sche = (
            [4] * (num_epochs // 5)
            + [2] * (num_epochs // 5)
            + [1] * (num_epochs // 5)
        )
        cur_ds = c2f_sche[0]

    model = FullModel(
        w=MM,
        h=MM,
        num_feats=num_feats,
        x_mode=num_modes,
        y_mode=num_modes,
        z_min=z_min,
        z_max=z_max,
        ds_factor=cur_ds,
        use_layernorm=use_layernorm,
    ).to(device)

    optimizer = torch.optim.Adam(
        lr=1e-3,
        params=filter(lambda p: p.requires_grad, model.parameters()),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step, gamma=0.1
    )

    t = tqdm.trange(num_epochs)
    for epoch in t:
        led_idices = list(np.arange(ID_len))  # list(np.random.permutation(ID_len)) #
        # _fill = len(led_idices) - (len(led_idices) % led_batch_size)
        # led_idices = led_idices + list(np.random.choice(led_idices, _fill, replace=False))
        if fit_3D:
            dzs = (
                (torch.randperm(num_z - 1)[: num_z // 2] + torch.rand(num_z // 2))
                * ((z_max - z_min) // (num_z - 1))
            ).to(device) + z_min
            if epoch % 2 == 0:
                dzs = torch.linspace(z_min, z_max, num_z).to(device)
        else:
            dzs = torch.FloatTensor([0.0]).to(device)

        if use_c2f and c2f_sche[epoch] < model.ds_factor:
            model.init_scale_grids(ds_factor=c2f_sche[epoch])
            print(f"ds_factor changed to {c2f_sche[epoch]}")
            model_fn = torch.jit.trace(model, dzs[0:1])

        if epoch == 0:
            if is_os == "Windows":
                model_fn = torch.jit.trace(model, dzs[0:1])
            elif is_os == "Linux":
                model_fn = torch.compile(model, backend="inductor")
            else:
                raise NotImplementedError


        for dz in dzs:
            dz = dz.unsqueeze(0)

            for it in range(ID_len // led_batch_size):  # + 1
                model.zero_grad()
                dfmask = torch.exp(
                    1j
                    * kzz.repeat(dz.shape[0], 1, 1)
                    * dz[:, None, None].repeat(1, kzz.shape[1], kzz.shape[2])
                )
                led_num = led_idices[it * led_batch_size : (it + 1) * led_batch_size]
                dfmask = dfmask.unsqueeze(1).repeat(1, len(led_num), 1, 1)
                spectrum_mask_ampli = Pupil0.repeat(
                    len(dz), len(led_num), 1, 1
                ) * torch.abs(dfmask)
                spectrum_mask_phase = Pupil0.repeat(len(dz), len(led_num), 1, 1) * (
                    torch.angle(dfmask) + 0
                )  # 0 represent Pupil0 Phase
                spectrum_mask = spectrum_mask_ampli * torch.exp(
                    1j * spectrum_mask_phase
                )

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    img_ampli, img_phase = model_fn(dz)
                    img_complex = img_ampli * torch.exp(1j * img_phase)
                    uo, vo = ledpos_true[led_num, 0], ledpos_true[led_num, 1]
                    x_0, x_1 = vo - M // 2, vo + M // 2
                    y_0, y_1 = uo - N // 2, uo + N // 2

                    oI_cap = torch.sqrt(Isum[:, :, led_num])
                    oI_cap = (
                        oI_cap.permute(2, 0, 1).unsqueeze(0).repeat(len(dz), 1, 1, 1)
                    )
                    

                    oI_sub = get_sub_spectrum(
                        img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, MAGimg
                    )
                    l1_loss = F.smooth_l1_loss(oI_cap, oI_sub)
                    loss = l1_loss
                    mse_loss = F.mse_loss(oI_cap, oI_sub)

                loss.backward()

                psnr = 10 * -torch.log10(mse_loss).item()
                t.set_postfix(Loss=f"{loss.item():.4e}", PSNR=f"{psnr:.2f}")
                optimizer.step()

        scheduler.step()
    
        if (epoch+1) % 10 == 0 or ( epoch % 2 == 0 and epoch < 20) or epoch == num_epochs:
            if epoch == num_epochs - 1:  # 假设epoch从0开始计数
                np.save(f"{vis_dir}/last_amplitude.npy", img_ampli[0].float().cpu().detach().numpy())
                print(f"Saved last epoch amplitude data to {vis_dir}/last_amplitude.npy")
            amplitude = (img_ampli[0].float()).cpu().detach().numpy() 
            phase = (img_phase[0].float()).cpu().detach().numpy()

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

            im = axs[0].imshow(amplitude, cmap="gray")
            axs[0].axis("image")
            axs[0].set_title("Reconstructed amplitude")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            im = axs[1].imshow(phase , cmap="gray") # - phase.mean()
            axs[1].axis("image")
            axs[1].set_title("Reconstructed phase")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            plt.savefig(f"{vis_dir}/e_{epoch}.png")

        if fit_3D and (epoch % 5 == 0 or epoch == num_epochs) and epoch > 0:
            dz = torch.linspace(z_min, z_max, 61).to(device).view(61)
            with torch.no_grad():
                out = []
                for z in torch.chunk(dz, 32):
                    img_ampli, img_phase = model(z)
                    _img_complex = img_ampli * torch.exp(1j * img_phase)
                    out.append(_img_complex)
                img_complex = torch.cat(out, dim=0)
            _imgs = img_complex.abs().cpu().detach().numpy()
            # Save amplitude
            imgs = (_imgs - _imgs.min()) / (_imgs.max() - _imgs.min())
            save_gif(imgs, 'recon_amplitude.gif')
            
        
```