# main.py
```python
import torch
import argparse
import os
import glob
from solve_rytov import solve_rytov
import yaml

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--physics_config', type=str)
    parser.add_argument('--save_name', default=None, type = str)
    args = parser.parse_args()

    physics_config = load_yaml(args.physics_config)
    if args.save_name != None:
        paths = list(glob.iglob(args.data_config))
        paths.sort()
        print(paths)
        for pth in paths:
            sub_data_config = load_yaml(pth)
            if(sub_data_config['save_name']==args.save_name):
                # print(1)
                solve_rytov(sub_data_config, physics_config)
            else:
                continue
    else:
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("is_available        =", torch.cuda.is_available())
        print("device_count        =", torch.cuda.device_count())
        data_config = load_yaml(args.data_config)
        solve_rytov(data_config, physics_config)

```

# solve_rytov.py
```python
import numpy as np
import os
import warnings
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch_dct as dct
from torch import fft
from tqdm import tqdm

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = get_device()

def holo_ref_finder(input, Bound):
    idx_arr = []
    batch_num = 2
    for i in range(batch_num):
        batch_size = input.shape[0]//batch_num
        start_idx = i * batch_size
        end_idx = (i+1)*batch_size if (i+1)*batch_size else None

        size_input = input[start_idx:end_idx].shape
        # Spectrum = torch.fft.fft2(torch.fft.fftshift(input, dim = (1, 2))).abs()
        Spectrum = torch.fft.fft2(input[start_idx:end_idx]).abs()
        # plt.imsave('test_plot/spectrum.png', clear(torch.log(Spectrum[0, ...] + 1e-8)))
        SpectrumLeft = Spectrum[:, 0:size_input[1] // 2, :]
        SpectrumLeft[:, 0:Bound[0], 0:Bound[1]] = 0
        SpectrumLeft[:, 0:Bound[0], size_input[2] - Bound[1] - 1:] = 0
        SpectrumLeft[:, 0:3, :] = 0
        SpectrumLeft[:, :, 0:3] = 0
        SpectrumLeft[:, :, -3:] = 0
        SpecCum1 = torch.sum(SpectrumLeft, 2)
        SpecCum2 = torch.sum(SpectrumLeft, 1)
        # print(SpecCum1.shape)
        # print(SpecCum2.shape)
        index1 = torch.argmax(SpecCum1, dim=1)
        # print(index1.shape)
        index2 = torch.argmax(SpecCum2, dim=1)
        # index2 = index2 - size_input[2] * torch.floor((index2+1) / (size_input[2] / 2))
        index2 = index2 - size_input[2]
        idx_arr.append(torch.stack([index1, index2], 1))
    idx_arr = torch.concat(idx_arr, 0).to(torch.float32)
    return idx_arr

def holo_filter(input, kRef, Bound, size_Spec):
    # input = torch.from_numpy(input)
    out_arr = []
    batch_num = 2
    for i in range(batch_num):
        batch_size = input.shape[0]//batch_num
        start_idx = i * batch_size
        end_idx = (i+1)*batch_size if (i+1)*batch_size else None
        size_input = input[start_idx:end_idx].shape
        ax = torch.arange(0, size_input[1], device=input.device)
        ay = torch.arange(0, size_input[2], device=input.device)
        X, Y = torch.meshgrid(ax, ay)
        shiftInput = input[start_idx:end_idx] * torch.exp(2.0j * np.pi * (Y * kRef[1] / size_input[1] + X * kRef[0] / size_input[2])).unsqueeze(0)
        ax = torch.arange(-size_Spec[1] // 2, size_Spec[1] // 2, device=input.device)
        ay = torch.arange(-size_Spec[0] // 2, size_Spec[0] // 2, device=input.device)
        #print(ax)
        Sx, Sy = torch.meshgrid(ax, ay)
        # print(shiftInput.shape)
        Spectrum = fft.fftshift(fft.fft2(fft.fftshift(shiftInput)))
        start1 = (size_input[1] - size_Spec[0]) // 2
        end1 = (size_input[1] - size_Spec[0]) // 2 + size_Spec[0]
        start2 = (size_input[2] - size_Spec[1]) // 2
        end2 = (size_input[2] - size_Spec[1]) // 2+ size_Spec[1]
        output = Spectrum[:, start1:end1, start2:end2]
        output[:, ((Sy / Bound[1])**2 + (Sx / Bound[0])**2 > 1)] = 0
        output = fft.ifftshift(fft.ifft2(fft.ifftshift(output)))
        out_arr.append(output)
    out_arr = torch.concat(out_arr, 0)
    return out_arr

def wrap_to_Pi(psi):
    psiWrap = torch.remainder(psi, 2*np.pi)
    mask = torch.abs(psiWrap) > np.pi
    psiWrap[mask] -= 2*np.pi * torch.sign(psiWrap[mask])
    return psiWrap

def phase_unwrap(psi):
    '''
    single do PhaseUnwrap
    '''
    psi_size = psi.shape
    dx = torch.zeros((psi_size[0], psi_size[1], psi_size[2] + 1), device=psi.device)
    dy = torch.zeros((psi_size[0], psi_size[1] + 1, psi_size[2]), device=psi.device)
    dx[:, :, 1:-1] = wrap_to_Pi(torch.diff(psi, 1, 2))
    dy[:, 1:-1, :] = wrap_to_Pi(torch.diff(psi, 1, 1))
    rho = torch.diff(dx, 1, 2) + torch.diff(dy, 1, 1)
    dctRho = dct.dct_2d(rho, norm='ortho')
    # print(torch.isnan(dctRho).sum())
    ax = torch.arange(0, psi_size[1], device=psi.device)
    ay = torch.arange(0, psi_size[2], device=psi.device)
    X, Y = torch.meshgrid(ax, ay)
    temp = (torch.cos(np.pi * Y / psi_size[2]) + torch.cos(np.pi * X / psi_size[1]) - 2)
    dctPhi = dctRho / 2.0 / temp
    dctPhi[:, 0,0] = 0.
    phi = dct.idct_2d(dctPhi, norm='ortho')
    return phi

def least_square(x, y):
    length = x.shape[-1]
    return (length * torch.sum(x * y, axis = -1, keepdims=True) - torch.sum(x, axis = -1, keepdims=True) * torch.sum(y, axis = -1, keepdims=True)) / (length * torch.sum(x**2, axis = -1, keepdims=True) - torch.sum(x, axis = -1, keepdims=True)**2)

def holo_VISA(input, kScanIn, itertime = 5):
    size_input = input.shape[1:]
    ax = torch.arange(0, size_input[0], device=input.device)
    ay = torch.arange(0, size_input[1], device=input.device)
    X, Y = torch.meshgrid(ax, ay)
    X = X.unsqueeze(0)
    Y = Y.unsqueeze(0)
    for i in range(itertime):
        shiftInput = input * torch.exp(2.0j * np.pi * (Y * kScanIn[:, 1].unsqueeze(-1).unsqueeze(-1) / size_input[0] + X * kScanIn[:, 0].unsqueeze(-1).unsqueeze(-1) / size_input[1]))
        Phs = shiftInput.angle()
        # print(torch.isnan(Phs).sum())
        Phs = phase_unwrap(Phs)
        # print(torch.isnan(Phs).sum())
        x = torch.arange(0, size_input[0], 1, device=input.device).unsqueeze(0)
        y = torch.mean(Phs, axis = 2)
        alpha1 = least_square(x, y) * size_input[0]
        y = torch.mean(Phs, axis = 1)
        alpha2 = least_square(x, y) * size_input[0]
        # print(alpha1.shape)
        kScanIn = kScanIn - torch.concatenate([alpha1, alpha2], -1) / 2 / np.pi
    Amp = shiftInput.abs()
    kScanOut = kScanIn
    return Amp, Phs  , kScanOut

def compute_wavefront(amp, phs, k):
    nx = amp.shape[1]
    ny = amp.shape[2]
    xs = torch.linspace(-nx//2, nx//2-1, nx, dtype=torch.float32, device=amp.device)
    ys = torch.linspace(-ny//2, ny//2-1, ny, dtype=torch.float32, device=amp.device)
    #print(xs)
    XS, YS = torch.meshgrid(xs, ys, indexing='ij')
    XS = XS.unsqueeze(0)
    YS = YS.unsqueeze(0)
    kx = -k[:, 0].unsqueeze(-1).unsqueeze(-1)
    ky = -k[:, 1].unsqueeze(-1).unsqueeze(-1)
    print(kx.shape)
    # OrthoSlicer3D(phs.cpu().numpy()).show()
    field = amp * torch.exp(1j * phs) * torch.exp(2j * np.pi * (XS * kx / nx + YS * ky / ny))
    return field

def phase_residual_correction_with_mask(phase_img, mask, order=1):
    height, width = phase_img.shape[1:]
    device = phase_img.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device).float(),
        torch.arange(width, device=device).float(),
        indexing='ij'
    )
    valid_indices = mask > 0
    valid_x = x[valid_indices]
    valid_y = y[valid_indices]
    valid_values = phase_img[:, valid_indices]
    valid_x = valid_x / width
    valid_y = valid_y / height

    # 构建设计矩阵
    design_matrix = []

    # 添加X的多项式项
    for i in range(1, order + 1):
        design_matrix.append(valid_x ** i)

    # 添加Y的多项式项
    for i in range(1, order + 1):
        design_matrix.append(valid_y ** i)
    # 添加常数项
    design_matrix.append(torch.ones_like(valid_x))

    # 将列表转换为设计矩阵
    design_matrix = torch.stack(design_matrix, dim=1)

    # 使用最小二乘法求解多项式系数 (A^T A)^-1 A^T b
    ATA = torch.matmul(design_matrix.T, design_matrix).unsqueeze(0).repeat(valid_values.shape[0], 1, 1)  # 批次化
    # ATb = torch.matmul(design_matrix.T, valid_values)#批次化
    ATb = torch.einsum('ij,kj->ki', design_matrix.T, valid_values)
    # 增加少量正则化以确保可逆性
    epsilon = 1e-10
    ATA_reg = ATA + epsilon * torch.eye(ATA.shape[1], device=device).unsqueeze(0)
    coefficients = torch.linalg.solve(ATA_reg, ATb)
    # print(coefficients.shape)
    # 构建背景模型
    background = torch.zeros_like(phase_img)
    # 应用X多项式项
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    for i in range(order):
        background += coefficients[:, i].unsqueeze(-1).unsqueeze(-1) * (x / width) ** (i + 1)
    # 应用Y多项式项
    for i in range(order):
        background += coefficients[:, order + i].unsqueeze(-1).unsqueeze(-1) * (y / height) ** (i + 1)
    # 应用常数项
    background += coefficients[:, -1].unsqueeze(-1).unsqueeze(-1)
    # 从原始相位中减去背景
    # print(background.shape)
    return background

def holo_process_with_mask(odt_data, mask, device, VISATimes=5):
    raw_Samp = odt_data.sp
    raw_Back = odt_data.bg
    size_Spec = odt_data.spec_size
    para_NStack = odt_data.angle_num
    kBoundPixel = [odt_data.km_bound_pixel, odt_data.km_bound_pixel]
    conjFlag = odt_data.conjugate_flag
    raw_Back = raw_Back.to(device)
    raw_Samp = raw_Samp.to(device)
    k_ScanSamp = holo_ref_finder(raw_Samp, kBoundPixel) * conjFlag
    k_ScanBack = holo_ref_finder(raw_Back, kBoundPixel) * conjFlag
    k_RefSamp = torch.mean(k_ScanSamp, 0)
    k_RefBack = torch.mean(k_ScanBack, 0)
    k_ScanSamp = k_ScanSamp - k_RefSamp
    k_ScanBack = k_ScanBack - k_RefBack
    # print(k_ScanBack)
    # print(k_ScanSamp)
    batch_num = para_NStack // odt_data.batch_size
    u_Samp_amp = []
    u_Samp_phs = []
    u_Back_amp = []
    u_Back_phs = []
    full_k_ScanSamp = []
    full_k_ScanBack = []
    print(raw_Samp.shape)
    print(k_ScanBack.shape)
    for i in tqdm(range(odt_data.batch_size)):
        start_idx = i * batch_num
        end_idx = (i + 1) * batch_num if (i + 1) * batch_num < para_NStack else None
        u_Samp = holo_filter(raw_Samp[start_idx:end_idx, ...], k_RefSamp, kBoundPixel, size_Spec)
        u_Back = holo_filter(raw_Back[start_idx:end_idx, ...], k_RefSamp, kBoundPixel, size_Spec)
        sub_u_Back_amp, sub_u_Back_phs, sub_k_ScanBack = holo_VISA(u_Back, k_ScanBack[start_idx:end_idx, :],
                                                                   VISATimes)
        sub_u_Samp_amp, sub_u_Samp_phs, sub_k_ScanSamp = holo_VISA(u_Samp, k_ScanSamp[start_idx:end_idx, :],
                                                                   VISATimes)
        residual_phs = sub_u_Samp_phs - sub_u_Back_phs
        background = phase_residual_correction_with_mask(residual_phs, mask)
        sub_u_Samp_phs = sub_u_Samp_phs - background
        sub_k_ScanSamp = sub_k_ScanBack

        u_Samp_amp.append(sub_u_Samp_amp)
        u_Samp_phs.append(sub_u_Samp_phs)
        full_k_ScanSamp.append(sub_k_ScanSamp)
        full_k_ScanBack.append(sub_k_ScanBack)
        u_Back_amp.append(sub_u_Back_amp)
        u_Back_phs.append(sub_u_Back_phs)
    u_Samp_amp = torch.concat(u_Samp_amp, 0)
    u_Samp_phs = torch.concat(u_Samp_phs, 0)
    u_Back_amp = torch.concat(u_Back_amp)
    u_Back_phs = torch.concat(u_Back_phs)
    k_ScanSamp = torch.concat(full_k_ScanSamp)
    k_ScanBack = torch.concat(full_k_ScanBack)
    temp = k_ScanSamp[1:-1, :] * odt_data.spec_pixel_size / odt_data.km
    # print(temp)
    # print(odt_data.km)
    temp = torch.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2)[-96:].mean().item()
    # print(temp)
    print(f'scan with theta {np.arcsin(temp) * 180 / np.pi}')
    meanAmp = u_Back_amp.mean()
    u_Samp_amp = u_Samp_amp / meanAmp
    u_Back_amp = u_Back_amp / meanAmp
    # OrthoSlicer3D(u_Samp_amp.cpu().numpy()).show()
    # OrthoSlicer3D(u_Samp_phs.cpu().numpy()).show()
    # OrthoSlicer3D(u_Back_amp.cpu().numpy()).show()
    # OrthoSlicer3D(u_Back_phs.cpu().numpy()).show()
    ###########################################################################
    ## remove ref
    ###########################################################################
    mean_log_back_amp = torch.log(u_Back_amp).mean(axis=0)
    mean_back_phs = u_Back_phs.mean(axis=0)
    # plt.imshow(torch.exp(mean_log_back_amp).cpu().numpy(), cmap='gray')
    # plt.show()
    # plt.imshow(torch.exp(mean_back_phs).cpu().numpy(), cmap='gray')
    # plt.show()
    u_Samp_amp = torch.exp(torch.log(u_Samp_amp) - mean_log_back_amp)
    u_Back_amp = torch.exp(torch.log(u_Back_amp) - mean_log_back_amp)
    u_Samp_phs = u_Samp_phs - mean_back_phs
    u_Back_phs = u_Back_phs - mean_back_phs
    meanAmp = u_Back_amp.mean()

    u_Samp_amp = u_Samp_amp / meanAmp
    u_Back_amp = u_Back_amp / meanAmp
    print('u back phase mean', u_Back_phs.mean())
    print('u samp phase mean', u_Samp_phs.mean())
    # tifffile.imsave('test_phase.tif', u_Samp_phs.cpu().numpy())
    ############################################################################
    ## background correction
    ############################################################################

    odt_data.sp_amp = u_Samp_amp
    odt_data.sp_phs = u_Samp_phs
    odt_data.bg_amp = u_Back_amp
    odt_data.bg_phs = u_Back_phs
    odt_data.k_scan_sp = k_ScanSamp
    odt_data.k_scan_bg = k_ScanBack
    del odt_data.sp
    del odt_data.bg
    print(f'complex field shape {odt_data.sp_amp.shape}')
    print('sp phs mean', odt_data.sp_phs.mean())
    print('bg phs mean', odt_data.bg_phs.mean())
    u_out = compute_wavefront(odt_data.sp_amp, odt_data.sp_phs, k_ScanSamp)
    u_in = compute_wavefront(odt_data.bg_amp, odt_data.bg_phs,
                             k_ScanSamp)  # 必须要用同一个波矢，不然不匹配，k samp和k scan毕竟还是有细微差别的
    # OrthoSlicer3D((u_out - u_in).abs().cpu().numpy()).show()
    check_phase_correction = (u_out - u_in).abs().mean(axis=0)
    # plt.imshow(check_phase_correction.cpu().numpy())
    # plt.colorbar()
    # plt.show()
    # OrthoSlicer3D(torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(u_out)))) + 1e-8).cpu().numpy()).show()
    return odt_data, u_out, u_in, k_ScanSamp, k_ScanBack, check_phase_correction

from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import glob
import tifffile
from skimage.transform import downscale_local_mean

class physics_parameter:
    pass

def read_tiff(path):
    img_all = []
    paths = list(glob.iglob(path))
    paths.sort()
    total_frames = 0
    for p in paths:
        print(p)
        with tifffile.TiffFile(p) as tif:
            n_pages = len(tif.pages)
            # 该文件少于 3 帧则跳过
            if n_pages <= 2:
                print(f"  -> {p} 只有 {n_pages} 帧，跳过")
                continue
            # 只读取从第3帧起的所有帧
            arr = tif.asarray(key=slice(2, n_pages))  # 等价于 pages[2:]
        # 统一为 float32
        arr = arr.astype(np.float32, copy=False)

        # 确保形状为 (frames, H, W, ...)
        if arr.ndim == 2:
            arr = arr[None, ...]
        total_frames += arr.shape[0]
        img_all.append(arr)

    if not img_all:
        raise ValueError("没有可用帧（所有文件都不足 3 帧或未匹配到路径）。")

    # 在第0维拼接所有文件的“第3帧及之后”
    img_all = np.concatenate(img_all, axis=0) if len(img_all) > 1 else img_all[0]
    print(f"image shape {img_all.shape}，总帧数(从各文件第3帧开始计算)：{total_frames}")
    return torch.from_numpy(img_all)


class ODT_data(Dataset):
    def __init__(self, data_config, physics_config):
        self.physics_param = physics_parameter()
        print(data_config)
        sp_raw, bg_raw = self.read_raw(data_config)
        self.batch_size = data_config['batch_size']
        self.len = sp_raw.shape[0]
        self.sp = sp_raw
        self.bg = bg_raw
        self.wavelength = physics_config['wavelength']
        self.physics_param.wavelength = physics_config['wavelength']
        self.pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
        self.physics_param.pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
        self.n_medium = physics_config['n_medium']
        self.physics_param.n_medium = physics_config['n_medium']
        self.NA = physics_config['NA']
        self.physics_param.NA = physics_config['NA']
        self.conjugate_flag = physics_config['conjugate_flag']
        self.angle_num = physics_config['angle_num']
        self.crop_size = data_config['crop_size']
        self.spec_size = data_config['spec_size']
        self.domain_size = data_config['domain_size']
        self.angle_downsample = physics_config['angle_downsample']
        if(physics_config['angle_downsample']):
            num = physics_config['angle_downsample']
            self.sp = self.sp[::num, ...]
            self.bg = self.bg[::num, ...]
            self.angle_num = self.angle_num // num
            print(f'image size downsample to {self.sp.shape}')
        if(self.crop_size[0] < self.sp.shape[1]):
            idx_x = self.sp.shape[1]//2 + data_config['shift_idx'][0]
            idx_y = self.sp.shape[2]//2 + data_config['shift_idx'][1]
            self.sp = self.sp[:, idx_x - self.crop_size[0]//2:idx_x + self.crop_size[0]//2, idx_y - self.crop_size[1]//2:idx_y + self.crop_size[1]//2 ]
            self.bg = self.bg[:, idx_x - self.crop_size[0]//2:idx_x + self.crop_size[0]//2 , idx_y - self.crop_size[1]//2:idx_y + self.crop_size[1]//2 ]
            print(f'image size crop to {self.sp.shape}')
            
        

        self.exp_name = data_config['expriment_name']
        self.effective_NA = np.minimum(self.NA, self.n_medium)
        self.spec_pixel_size = 2 * np.pi / (self.pixelsize * self.crop_size[0])
        self.k_bound_pixel = np.ceil(self.effective_NA * 2 * np.pi / self.wavelength / self.spec_pixel_size).astype(int)
        self.km_bound_pixel = np.ceil(self.n_medium * 2 * np.pi / self.wavelength / self.spec_pixel_size).astype(int)
        self.km = self.n_medium * 2 * np.pi / self.wavelength
        self.physics_param.km = self.n_medium * 2 * np.pi / self.wavelength
        self.physics_param.k_bound_pixel = np.ceil(self.effective_NA * 2 * np.pi / self.wavelength / self.spec_pixel_size).astype(int)
        self.physics_param.spec_pixel_size = 2 * np.pi / (self.pixelsize * self.crop_size[0])
        print('retrieved pixel size', self.pixelsize * self.crop_size[0] / self.domain_size[-1])


    def read_raw(self, path_config):
        print('reading sample image')
        sp = read_tiff(path_config['sp_path'])
        print('reading background image')
        bg = read_tiff(path_config['bg_path'])

        return sp, bg
    
    def __len__(self):
            return self.len

    def __getitem__(self, index):
        sp = self.sp[index]
        bg = self.bg[index]
        return sp, bg

def ind2sub(ind_arr, shape):
    Y = torch.floor(ind_arr / shape[0])
    X = torch.fmod(ind_arr, shape[1])
    return X.long(), Y.long()

def load_binary_mask_float32(image_path, threshold=128, mask_type='fiji'):
    # 以灰度模式读取图片
    img_pil = Image.open(image_path).convert('L')
    # 转换为numpy数组
    img_array = np.array(img_pil)

    # 应用阈值进行二值化 (大于阈值为1.0，否则为0.0)
    if(mask_type=='fiji'):
        binary_mask =1- (img_array > threshold).astype(np.float32)
    elif(mask_type=='bg'):
        binary_mask = (img_array > threshold).astype(np.float32)

    return binary_mask

def RytovRec(ODT_data, exp):
    u_Samp_amp = ODT_data.sp_amp
    u_Samp_phs = ODT_data.sp_phs
    u_Back_amp = ODT_data.bg_amp
    u_Back_phs = ODT_data.bg_phs
    km = ODT_data.km

    u_Rytov = torch.log(u_Samp_amp / u_Back_amp) + 1.0j * (u_Samp_phs - u_Back_phs)
    # OrthoSlicer3D(u_Rytov.imag.numpy()).show()
    del ODT_data.sp_amp
    del ODT_data.sp_phs
    del ODT_data.bg_amp
    del ODT_data.bg_phs
    size_input = u_Rytov.shape
    size_Joint = ODT_data.domain_size
    u_Rytov = torch.fft.fftshift(torch.fft.fftshift(u_Rytov, 1), 2)
    u_Rytov = torch.fft.fft2(u_Rytov)
    u_Rytov = torch.fft.fftshift(torch.fft.fftshift(u_Rytov, 1), 2)
    print('u_Rytov shape', u_Rytov.shape)
    f_Joint = torch.zeros(size_Joint, dtype = torch.cfloat, device=u_Rytov.device)
    f_count = torch.zeros(size_Joint, dtype=torch.int)
    Ax = torch.linspace(-size_input[1] // 2, size_input[1] // 2 - 1, size_input[1])
    Ay = torch.linspace(-size_input[1] // 2, size_input[1] // 2- 1, size_input[1])
    kSizeZ = ODT_data.spec_pixel_size
    kSize = ODT_data.spec_pixel_size
    kScan = ODT_data.k_scan_sp.cpu()
    norm_coef = size_Joint[0] * size_Joint[1] * size_Joint[2] * kSizeZ / size_input[1] / size_input[2] / (4 * np.pi * np.pi)
    kScanZ = torch.sqrt(km**2 - (kScan[:, 0] * kSize)**2 - (kScan[:, 1] * kSize)**2)
    index_map = torch.arange(0, size_input[1] * size_input[2]).reshape(size_input[1], size_input[2]).transpose(1,0)
    
    NStack = ODT_data.angle_num
    kBound = ODT_data.k_bound_pixel
    print('Nstack', NStack)
    print('kScanz shape', kScanZ.shape)
    for i in tqdm(range(kScanZ.shape[0])):
        SAx = Ax - kScan[i, 0]
        SAy = Ay - kScan[i, 1]
        Sx, Sy = torch.meshgrid(SAx, SAy)
        Mindex = torch.lt(torch.square(Sx / kBound) + torch.square(Sy / kBound),  1)
        kz = torch.sqrt(1 - (Sx[Mindex] / kBound)**2 - (Sy[Mindex] / kBound)**2) * km
        I3 = (torch.fmod((kz - kScanZ[i]) // kSizeZ + size_Joint[0] // 2, size_Joint[0]))

        I3 = I3.int()
        I1, I2 = ind2sub(index_map[Mindex], [size_input[1], size_input[2]])
        
        f_Joint[I3.long(), I1.long(), I2.long()] += u_Rytov[i, I1.long(), I2.long()] * kz.to(u_Rytov.device)
        
        f_count[I3.long(), I1.long(), I2.long()] += 1
    f_Joint = f_Joint.cpu()
    f_Joint[f_count > 0] = f_Joint[f_count > 0] / f_count[f_count > 0]
    # torch.save(f_Joint*norm_coef, 'data/f_Joint.pt')
    f_count[f_count > 0] = 1
    f = -torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(f_Joint))) * 1.0j * norm_coef
    print('f real min', f.real.min(), 'f real max', f.real.max())
    print('f imag min', f.imag.min(), 'f imag max', f.imag.max())
    
    f = f * 4 * np.pi #这样就可以无缝调用模拟数据的散射势转RI函数，数值可能还是对不太上，就不管了吧
    
    deltaRI = (ODT_data.n_medium * torch.sqrt(1 +  f  * (ODT_data.wavelength / (ODT_data.n_medium * 2 * np.pi))**2)) - ODT_data.n_medium
    deltaRI = deltaRI.real

    f_count[f_count > 0.5] = 1
    f_count[f_count < 0.5] = 0
    f_count_hermite = f_count + torch.flip(f_count, dims=[0, 1, 2])
    f_Joint = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(f)))
    resolution = 1 / np.asarray(size_Joint) * np.max(np.asarray(size_Joint))
    return deltaRI, f_Joint, norm_coef, f, f_count, (torch.log(u_Samp_amp / u_Back_amp) + 1.0j * (u_Samp_phs - u_Back_phs)), resolution


def solve_rytov(data_config, physics_config):
    data_pack = ODT_data(data_config, physics_config)
    exp = data_pack.exp_name
    data_name = data_config['data_name']
    ###################################################
    ## rytov recon
    ###################################################
    warnings.warn('please check mask path')
    mask_path = os.path.join(data_config['save_path'], data_config['expriment_name'], 'phase_mask', data_name + '.png')
    print(mask_path)
    shift_idx = np.asarray(data_config['shift_idx'])
    spec_size = np.asarray(data_config['spec_size'])
    if os.path.exists(mask_path):
        # 正常加载现有 mask
        mask = load_binary_mask_float32(mask_path, mask_type=data_config['mask_type'])
    else:
        print(f"Mask file not found: {mask_path}, using default mask in memory.")
        # 直接在内存里创建默认 mask（全黑）
        mask = np.zeros(spec_size, dtype=np.float32)  # 全黑掩码
    
    mask = mask[mask.shape[0]//2 + shift_idx[0]//2 - spec_size[0]//2:mask.shape[0]//2 +shift_idx[0]//2 + spec_size[0]//2, shift_idx[1]//2 - spec_size[1]//2+ mask.shape[1]//2:shift_idx[1]//2 + spec_size[1]//2 + mask.shape[1]//2]
    
    data_pack, u_out, u_in, k_scan_samp, k_scan_back, check_phase_correction = holo_process_with_mask(data_pack, mask, device)
    RI_rytov, V_rytov_fft, norm_coef, f_target, f_count, rytov_phs, resolution = RytovRec(data_pack, exp)


    save_pth = os.path.join(data_config['save_path'], data_config['expriment_name'], data_config['save_name'])
    # OrthoSlicer3D(delta_RI[30:-30,30:-30,30:-30].detach().cpu().numpy()).show()
    os.makedirs(os.path.join(save_pth, data_name, 'scatter_potential'), exist_ok=True)
    os.makedirs(os.path.join(save_pth, 'rytov'), exist_ok=True)
    os.makedirs(os.path.join(save_pth, data_name, 'wavefront_remove_ref'), exist_ok=True)
    np.save(os.path.join(save_pth,data_name, 'scatter_potential', 'potential_fft.npy'), V_rytov_fft.detach().cpu().numpy())
    np.save(os.path.join(save_pth,data_name, 'scatter_potential', 'potential_mask.npy'), f_count.detach().cpu().numpy())
    np.save(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'u_in.npy'), u_in.detach().cpu().numpy())
    np.save(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'u_out.npy'), u_out.detach().cpu().numpy())
    np.save(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'k_samp.npy'), k_scan_samp.detach().cpu().numpy())
    np.save(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'k_back.npy'), k_scan_back.detach().cpu().numpy())
    np.save(os.path.join(save_pth, data_name, 'wavefront_remove_ref', 'rytov_phs.npy'), rytov_phs.detach().cpu().numpy())
    os.makedirs(os.path.join(save_pth,'rytov'), exist_ok=True)

```