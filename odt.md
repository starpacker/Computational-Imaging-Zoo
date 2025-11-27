# bpm algorithm

# bpm_algorithm.py
```python
import numpy as np
import torch
from tqdm import tqdm
from functools import partial

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = get_device()

def compute_bpm_grad(init_delta_ri, u_in, u_out, batch_size, cos_factor, dz, domain_size, k0, p_kernel, device):
    
    ol_factor = k0 * dz / cos_factor.unsqueeze(-1)
    p_kernel = p_kernel.unsqueeze(0)
    bp_kernel = p_kernel.conj()
    grad = torch.zeros_like(init_delta_ri)
    s = torch.zeros((batch_size,)+init_delta_ri.shape, dtype=torch.cfloat, device=device)
    delta_ri = init_delta_ri
    loss=0
    num_batches = u_in.shape[0]//batch_size
    # print(u_in.shape)
    for i in range(num_batches):
        print(f"{i}/{num_batches}")
        start_idx = i * batch_size
        end_idx = (i + 1)*batch_size if (i + 1) * batch_size < u_in.shape[0] else None
        sub_u_in = u_in[start_idx:end_idx, ...]
        sub_u_out = u_out[start_idx:end_idx, ...]
        u = sub_u_in
        for m in range(init_delta_ri.shape[0]):
            u = torch.fft.ifft2(torch.fft.fft2(u) * p_kernel)#广播机制
            s[:, m, ...] = u.conj()
            u = u * torch.exp(1j * ol_factor[start_idx:end_idx, ...] * delta_ri[m, ...].unsqueeze(0))
        r = u - sub_u_out
        loss +=r.abs().mean()
        for m in reversed(range(init_delta_ri.shape[0])):
            r = r * torch.exp(-1j * ol_factor[start_idx:end_idx, ...] * delta_ri[m, ...].unsqueeze(0))
            s[:, m, ...] = -1j * ol_factor[start_idx:end_idx, ...] * s[:, m, ...] * r
            r = torch.fft.ifft2(torch.fft.fft2(r) * bp_kernel)
        grad = grad + s.sum(axis=0)
    grad = grad / u_in.shape[0]
    loss = loss/(u_in.shape[0]//batch_size)
    print('loss', loss)
    return grad, loss

def angular_spectrum_kernel(domain_size, spec_pixel_size, pixel_size, km):
    assert domain_size[1] % 2 == 0 and domain_size[2] % 2 == 0, "domain_size[1] and domain_size[2] must be even"
    kx = (torch.linspace((-domain_size[1] // 2+1), (domain_size[1] // 2 ), domain_size[1]) - 1) * spec_pixel_size
    ky = (torch.linspace((-domain_size[2] // 2+1), (domain_size[2] // 2 ), domain_size[2]) - 1) * spec_pixel_size
    # print(torch.linspace((-domain_size[2] // 2 + 1), (domain_size[2] // 2 ), domain_size[2]))
    [Ky, Kx] = torch.meshgrid(ky, kx, indexing='ij')
    K2 = Kx**2 + Ky**2
    Kz = torch.sqrt(-K2 + km**2 + 0j)
    Kz[-K2 + km**2<0]=0.
    kernel = torch.exp(1j * Kz * pixel_size[0])
    return  (torch.fft.fftshift(kernel))

def soft_threshold(z, thres = 1e-3):
    return torch.sign(z)*torch.max(torch.abs(z) - thres,torch.zeros_like(z))

def operator_grad(x, pixel_size_z=1):
    '''
    x: shape (Nx, Ny, Nz)
    '''
    g = torch.zeros(x.shape + (3,), dtype = torch.float32, device=device)
    g[:-1, :, :, 0] = x[1:, :, :] - x[:-1, :, :] / pixel_size_z
    g[:, :-1, :, 1] = x[:, 1:, :] - x[:, :-1, :]
    g[:, :, :-1, 2] = x[:, :, 1:] - x[:, :, :-1]
    return g

def projector_threshold(x, ROI, a = 0, b=100, ):
    '''
    a < x < b
    '''
    x[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]]=torch.clamp(x[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]], min = a, max=b) #+ x.imag[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]]
    return x

def operator_div(g, pixel_size_z=1):
    x = torch.zeros(g.shape[:-1], dtype = torch.float32, device=device)
    tmp = x.clone()
    tmp[1:-1, :, :] = (g[1:-1, :, :, 0] - g[:-2, :, :, 0])
    tmp[0, :, :] = g[0, :, :, 0]
    tmp[-1, :, :] = - g[-2, :, :, 0] 
    x += tmp / pixel_size_z
    tmp[:, 1:-1, :] = g[:, 1:-1, :, 1] - g[:, :-2, :, 1]
    tmp[:, 0, :] = g[:, 0, :, 1]
    tmp[:, -1, :] = - g[:, -2, :, 1]
    x += tmp
    tmp[:, :, 1:-1] = g[:, :, 1:-1, 2] - g[:, :, :-2, 2]
    tmp[:, :, 0] = g[:, :, 0, 2]
    tmp[:, :, -1] = - g[:, :, -2, 2]
    x += tmp
    return - x

def projector_grad(g):
    '''
    g: shape(Nx, Ny, Nz, 3), isotropic case
    '''
    norm = torch.linalg.norm(g, dim = -1)
    norm[norm < 1] = 1
    norm = norm.reshape(g.shape[:-1] + (1,))
    g /= norm
    return g

def indentical_map(x, *args):
    return x

def make_regulizer(tv_param, value_range_param, sparse_param, ROI=[None, None, None, None, None, None], step_size=0):
    value_range_regu = partial(projector_threshold,ROI=ROI, a = value_range_param[0], b=value_range_param[1])
    sparse_regu = partial(soft_threshold, thres = sparse_param * step_size) if sparse_param != None else indentical_map
    # print(sparse_param)
    if  tv_param[0] == None:
        return lambda x: sparse_regu(value_range_regu(x))
    else:
        def fista_regu(z):
            tau = tv_param[0]
            step_num = tv_param[1]
            g_1 = operator_grad(z)
            d = g_1.clone()
            q_1 = 1
            gamma = 1 / (12 * tau)
            for i in (range(step_num)):
                g_2 = projector_grad(d + gamma * operator_grad(value_range_regu(z - tau * operator_div(d))))
                x = (value_range_regu(z - tau * operator_div(g_2)))
                q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
                del d
                d = g_2 + ((q_1 - 1) / q_2) * (g_2 - g_1)
                q_1 = q_2
                g_1 = g_2
            return sparse_regu(x)
        return fista_regu




def solve_bpm(data_config, physics_config, reconstruction_config):
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    crop_size = data_config['crop_size']
    spec_size = data_config['spec_size']
    data_name = data_config['data_name']
    domain_size = data_config['domain_size']

    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    u_in = np.load('data_folder/u_in.npy')
    u_in = torch.from_numpy(u_in).to(device)
    u_out = np.load('data_folder/u_out.npy')  # .real
    u_out = torch.from_numpy(u_out).to(device)
    k_scan_samp = np.load('data_folder/k_samp.npy')
    temp = k_scan_samp * spec_pixel_size / km
    bpm_cosFactor = np.cos(np.arcsin(np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2)))
    bpm_cosFactor = torch.from_numpy(bpm_cosFactor.reshape(-1, 1)).to(device)

    crop_z = data_config['crop_z']
    region_z = domain_size[0]
    bg_z = region_z // 2
    if crop_z != None:
        region_z = data_config['region_z']
        bg_z = data_config['bg_z']
    p_kernel = angular_spectrum_kernel(domain_size, spec_pixel_size, resolution, km)#.to(device)
    p_kernel = p_kernel.to(device)
    u_inlet = torch.fft.ifft2(torch.fft.fft2(u_in) * (p_kernel ** (region_z - bg_z)).conj())
    u_outlet = torch.fft.ifft2(torch.fft.fft2(u_out) * (p_kernel ** (bg_z)))

    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    ROI = data_config['ROI']
    batch_size = reconstruction_config['batch_size']

    regu_func = make_regulizer(reconstruction_config['tv_param'], reconstruction_config['value_range_param'], reconstruction_config['sparse_param'][0])


    init_delta_ri = torch.zeros((region_z, domain_size[1], domain_size[2]), dtype=torch.float32, device=device)
    print('RI shape', init_delta_ri.shape)
    s = init_delta_ri.clone()
    q_1 = 1
    x_1 = init_delta_ri.clone()
    pbar = tqdm(range(n_iter))
    for i in pbar:
        # compute_bpm_grad
        grad, loss = compute_bpm_grad(s, u_inlet, u_outlet, batch_size, bpm_cosFactor, resolution[0], domain_size, k0,
                                      p_kernel, device)
        grad = grad.real
        pbar.set_postfix({'loss': loss.item()}, refresh=False)
        with torch.no_grad():
            z = s - grad * step_size
            # f = torch.clamp(f, min = 0., max=1000)
            x_2 = regu_func(z)
            q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            print(i, ((x_2 - x_1)**2).sum().sqrt())    # print output
            x_1 = x_2
            q_1 = q_2
        # f = f.detach()
    delta_ri = s
    vmin = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].min()
    vmax = delta_ri[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]].max()
    print(vmin)
    print(vmax)

```

# main.py
```python
import argparse
import yaml
from bpm_algorithm import solve_bpm 

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--physics_config', type=str)
    parser.add_argument('--save_name', default=None, type = str)
    parser.add_argument('--reconstruction_config', type=str)
    args = parser.parse_args()

    physics_config = load_yaml(args.physics_config)
    reconstruction_config = load_yaml(args.reconstruction_config)
    data_config = load_yaml(args.data_config)
    
    solve_bpm(data_config, physics_config, reconstruction_config)
```