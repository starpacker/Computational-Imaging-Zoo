# iterative_deconv/iterative_deconv.py
```python
import math
import warnings
import numpy as np
from numpy import zeros


xp = np
def iterative_deconv(data,kernel,iteration,rule):
    # print(1,data.shape)
    if xp is not np:
        data = xp.asarray(data)
        kernel = xp.asarray(kernel)

    if data.ndim > 2:
        data_de = xp.zeros((data.shape[0], data.shape[1],data.shape[2]), dtype = 'float32')
        # for i in range(0, data.shape[0]):
        #     data_de[i, :, :] = (deblur_core(data[i, :,:], kernel, iteration, rule)).real
        for i in range(0, data.shape[2]):
            data_de[:, :, i] = (deblur_core(data[:, :,i], kernel, iteration, rule)).real
    else:
        data_de = (deblur_core(data, kernel, iteration, rule)).real

    if xp is not np:
        data_de = xp.asnumpy(data_de)

    return data_de

def deblur_core(data, kernel, iteration, rule):
    # print(data.shape)
    #data = cp.asnumpy(data)
    kernel = xp.array(kernel)
    kernel = kernel / sum(sum(kernel))
    kernel_initial = kernel
    [dx,dy] = data.shape

    B = math.floor(min(dx,dy)/6)
    data = xp.pad(data, [int(B),int(B)], 'edge')
    yk = data
    xk = zeros((data.shape[0], data.shape[1]), dtype = 'float32')
    vk = zeros((data.shape[0], data.shape[1]), dtype = 'float32')
    otf = psf2otf(kernel_initial, data.shape)

    if rule == 2: 
    #LandWeber deconv
        t = 1
        gamma1 = 1
        for i in range(0,iteration):

            if i == 0:
                xk_update = data

                xk = data + t*xp.fft.ifftn(xp.conj(otf)) * (xp.fft.fftn(data) - (otf *xp.fft.fftn(data)))
            else:
                gamma2 = 1/2*(4 * gamma1*gamma1 + gamma1**4)**(1/2) - gamma1**2
                beta = -gamma2 *(1 - 1 / gamma1)
                yk_update = xk + beta * (xk - xk_update)
                yk = yk_update + t * xp.fft.ifftn(xp.conj(otf) * (xp.fft.fftn(data) - (otf * xp.fft.fftn(yk_update))))
                yk = xp.maximum(yk, 1e-6, dtype = 'float32')
                gamma1 = gamma2
                xk_update = xk
                xk = yk

    elif rule == 1:
        
    #Richardson-Lucy deconv

        for iter in range(0, iteration):

            xk_update = xk
            rliter1 = rliter(yk, data, otf)

            xk = yk * ((xp.fft.ifftn(xp.conj(otf) * rliter1)).real) / ( (xp.fft.ifftn(xp.fft.fftn(xp.ones(data.shape)) * otf)).real)

            xk = xp.maximum(xk, 1e-6, dtype = 'float32')

            vk_update = vk

            vk =xp.maximum(xk - yk, 1e-6 , dtype = 'float32')

            if iter == 0:
                alpha = 0
                yk = xk
                yk = xp.maximum(yk, 1e-6,dtype = 'float32')
                yk = xp.array(yk)

            else:

                alpha = sum(sum(vk_update * vk))/(sum(sum(vk_update * vk_update)) + 1e-10)
                alpha = xp.maximum(xp.minimum(alpha, 1), 1e-6, dtype = 'float32')
               # start = time.clock()
                yk = xk + alpha * (xk - xk_update)
                yk = xp.maximum(yk, 1e-6, dtype = 'float32')
                yk[xp.isnan(yk)] = 1e-6
                #end = time.clock()
               # print(start, end)
                #K=np.isnan(yk)

    yk[yk < 0] = 0
    yk = xp.array(yk, dtype = 'float32')
    data_decon = yk[B + 0:yk.shape[0] - B, B + 0: yk.shape[1] - B]

    return data_decon

def cart2pol(x, y):
    rho = xp.sqrt(x ** 2 + y ** 2)
    phi = xp.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * xp.cos(phi)
    y = rho * xp.sin(phi)
    return (x, y)

def psf2otf(psf, outSize):
    psfSize = xp.array(psf.shape)
    outSize = xp.array(outSize)
    padSize = xp.array(outSize - psfSize)
    psf = xp.pad(psf, ((0, int(padSize[0])), (0, int(padSize[1]))), 'constant')
    for i in range(len(psfSize)):
        psf = xp.roll(psf, -int(psfSize[i] / 2), i)
    otf = xp.fft.fftn(psf)
    nElem = xp.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * xp.log2(psfSize[k]) * nffts
    if xp.max(xp.abs(xp.imag(otf))) / xp.max(xp.abs(otf)) <= nOps * xp.finfo(xp.float32).eps:
        otf = xp.real(otf)
    return otf

def rliter(yk,data,otf):
    rliter = xp.fft.fftn(data / xp.maximum(xp.fft.ifftn(otf * xp.fft.fftn(yk)), 1e-6))
    return rliter
```

# iterative_deconv/kernel.py
```python
import numpy as np
import math
from numpy import log


def Gauss(sigma):
    sigma = np.array(sigma,dtype = 'float32')
    s=sigma.size
    if s==1:
       sigma=[sigma,sigma]
    sigma = np.array(sigma,dtype = 'float32')
    psfN = np.ceil(sigma / math.sqrt(8 * log(2)) * math.sqrt(-2 * log(0.0002))) + 1
    N = psfN * 2 + 1
    sigma = sigma / (2 * math.sqrt(2 * log(2)))
    dim = len(N)
    if dim > 1:
        N[1] = np.maximum(N[0], N[1])
        N[0] = N[1]
    if dim == 1:
        x = np.arange(-np.fix(N / 2), np.ceil(N / 2),dtype='float32')
        PSF = np.exp(-0.5 * (x * x) / (np.dot(sigma, sigma)))
        PSF = PSF / PSF.sum()
        center = N / 2 + 1
        return PSF
    if dim == 2:
        m = N[0]
        n = N[1]
        x = np.arange(-np.fix((n / 2)), np.ceil((n / 2)),dtype='float32')
        y = np.arange(-np.fix((m / 2)), np.ceil((m / 2)),dtype='float32')
        X, Y = np.meshgrid(x, y)
        s1 = sigma[0]
        s2 = sigma[1]
        PSF = np.exp(-(X * X) / (2 * np.dot(s1, s1)) - (Y * Y) / (2 * np.dot(s2, s2)))
        PSFsum = PSF.sum()
        PSF = PSF / PSFsum
        center = [m / 2 + 1, n / 2 + 1]
        return PSF
    if dim == 3:
        m = N[0]
        n = N[1]
        k = N[2]
        x = np.arange(-np.fix(n / 2), np.ceil(n / 2),dtype='float32')
        y = np.arange(-np.fix(m / 2), np.ceil(m / 2),dtype='float32')
        z = np.arange(-np.fix(k / 2), np.ceil(k / 2),dtype='float32')
        [X, Y, Z] = np.meshgrid(x, y, z)
        s1 = sigma[0]
        s2 = sigma[1]
        s3 = sigma[2]
        PSF = np.exp(-(X * X) / (2 * s1 * s1) - (Y * Y) / (2 * s2 * s2) - (Z * Z) / (2 * s3 ** 2))
        PSFsum = PSF.sum()
        PSF = PSF / PSFsum
        return PSF
```

# sparse_hessian_recon/operation.py
```python
import numpy as np
xp = np

def operation_xx(gsize):
    delta_xx = xp.array([[[1, -2, 1]]], dtype = 'float32')
    xxfft = xp.fft.fftn(delta_xx, gsize) * xp.conj(xp.fft.fftn(delta_xx, gsize))
    return xxfft

def operation_xy(gsize):
    delta_xy = xp.array([[[1, -1], [-1, 1]]], dtype = 'float32')
    xyfft = xp.fft.fftn(delta_xy, gsize) * xp.conj(xp.fft.fftn(delta_xy, gsize))
    return xyfft

def operation_xz(gsize):
    delta_xz = xp.array([[[1, -1]], [[-1, 1]]], dtype = 'float32')
    xzfft = xp.fft.fftn(delta_xz,gsize) * xp.conj(xp.fft.fftn(delta_xz, gsize))
    return xzfft

def operation_yy(gsize):
    delta_yy = xp.array([[[1], [-2], [1]]], dtype = 'float32')
    yyfft = xp.fft.fftn(delta_yy,gsize) * xp.conj(xp.fft.fftn(delta_yy, gsize))
    return yyfft

def operation_yz(gsize):
    delta_yz = xp.array([[[1], [-1]], [[-1], [1]]], dtype = 'float32')
    yzfft = xp.fft.fftn(delta_yz,gsize) * xp.conj(xp.fft.fftn(delta_yz, gsize))
    return yzfft

def operation_zz(gsize):
    delta_zz = xp.array([[[1]], [[-2]], [[1]]], dtype = 'float32')
    zzfft = xp.fft.fftn(delta_zz,gsize) * xp.conj(xp.fft.fftn(delta_zz, gsize))
    return zzfft

```

# sparse_hessian_recon/sparse_hessian_recon.py
```python
import gc
from .operation import *
from .sparse_iteration import *
import numpy as np
xp = np
cp = None
def sparse_hessian(f, iteration_num = 1000, fidelity = 150, sparsity = 10, contiz = 0.5 , mu = 1):
    '''
    function g = SparseHessian_core(f,iteration_num,fidelity,sparsity,iteration,contiz,mu)
    -----------------------------------------------
    Source code for argmin_g { ||f-g ||_2^2 +||gxx||_1+||gxx||_1+||gyy||_1+lamdbaz*||gzz||_1+2*||gxy||_1
     +2*sqrt(lamdbaz)||gxz||_1+ 2*sqrt(lamdbaz)|||gyz||_1+2*sqrt(lamdbal1)|||g||_1}
     f : ndarray
       Input image (can be N dimensional).
     iteration_num:  int, optional
        the iteration of sparse hessian {default:100}
     fidelity : int, optional
       fidelity {default: 150}
     contiz  : int, optional
       continuity along z-axial {example:1}
     sparsity :  int, optional
        sparsity {example:15}
    ------------------------------------------------
    Output:
      g
    '''
    if xp is not cp:
        contiz = np.sqrt(contiz)
        f1 = f
    else:
        contiz = cp.sqrt(contiz)
        f1 = cp.asarray(f, dtype = 'float32')
    flage = 0
    # f = cp.divide(f,cp.max(f[:]))
    f_flag = f.ndim
    if f_flag == 2:
        contiz = 0
        flage = 1
        f = xp.zeros((3,f.shape[0], f.shape[1]), dtype = 'float32')
        f = xp.array(f)
        for i in range(0,3):
            f[i,:,:] = f1
        
    elif f_flag > 2:
        if f1.shape[0] < 3:
            contiz = 0
            f = xp.zeros((3, f.shape[1], f.shape[2]), dtype = 'float32')
            f[0:f1.shape[0],:,:] = f1
            for i in range(f1.shape[0], 3):
                f[i, :, :] = f[1,:,:]
        else:
             f = f1
    imgsize = xp.shape(f)

    print("Start the Sparse deconvolution...")
    ## calculate derivate
    xxfft = operation_xx(imgsize)
    yyfft = operation_yy(imgsize)
    zzfft = operation_zz(imgsize)
    xyfft = operation_xy(imgsize)
    xzfft = operation_xz(imgsize)
    yzfft = operation_yz(imgsize)

    operationfft = xxfft + yyfft + (contiz**2)*zzfft+ 2*xyfft +2*(contiz)*xzfft + 2*(contiz)*yzfft
    normlize = (fidelity/mu) + (sparsity**2) + operationfft
    del xxfft,yyfft,zzfft,xyfft,xzfft,yzfft,operationfft
    gc.collect()
    # xp.clear_memo()
    ## initialize b
    bxx = xp.zeros(imgsize,dtype='float32')
    byy = bxx
    bzz = bxx
    bxy = bxx
    bxz = bxx
    byz = bxx
    bl1 = bxx
    ## initialize g
    g_update = xp.multiply(fidelity / mu, f)
    ## iteration
    tol = 1e-4 
    residual_prev = xp.inf
    for iter in range(0, iteration_num):

        g_update = xp.fft.fftn(g_update)

        if iter == 0:
            g = xp.fft.ifftn(g_update / (fidelity / mu)).real

        else:
            g = xp.fft.ifftn(xp.divide(g_update, normlize)).real


        g_update =xp.multiply((fidelity / mu), f)

        Lxx,bxx = iter_xx(g, bxx, 1, mu)
        g_update = g_update + Lxx
        del Lxx
        gc.collect()

        Lyy,byy = iter_yy(g, byy, 1, mu)
        g_update = g_update + Lyy
        del Lyy
        gc.collect()

        Lzz,bzz = iter_zz(g, bzz, contiz**2, mu)
        g_update = g_update + Lzz
        del Lzz
        gc.collect()        

        Lxy,bxy = iter_xy(g, bxy, 2, mu)
        g_update = g_update + Lxy
        del Lxy
        gc.collect()        

        Lxz,bxz = iter_xz(g, bxz, 2 * contiz, mu)
        g_update = g_update + Lxz
        del Lxz
        gc.collect()        

        Lyz,byz = iter_yz(g, byz, 2 * contiz, mu)
        g_update = g_update + Lyz
        del Lyz
        gc.collect()

        Lsparse,bl1 = iter_sparse(g, bl1, sparsity, mu)
        g_update = g_update + Lsparse
        del Lsparse
        gc.collect()
        # 计算当前残差（L2 范数）
        

        # 检查残差变化是否足够小（相对变化）
        if iter % 20 == 0:
            residual = xp.linalg.norm(f - g)
            rel_change = abs(residual - residual_prev) / (residual_prev + 1e-12)
            if rel_change < tol:
                print(f"Converged at iteration {iter}: residual change = {rel_change:.2e}")
                break

            residual_prev = residual

        if iter % 20 ==0:

            print('%d iterations done\r' % iter)

    g[g < 0] = 0


    del bxx,byy,bzz,bxy,byz,bl1,f,normlize,g_update
    gc.collect()

    return g[1, :, :] if flage else g
```

# sparse_hessian_recon/sparse_iterations.py
```python
import  warnings
import numpy as np
xp = np
def forward_diff(data, step, dim):
    # data --- input image(gpu array!!!)
    # step
    # dim --- determine which is the dimension to calculate derivate
    # dim = 0 --> z axis
    # dim = 1 --> y axis
    # dim = 2 --> x axis

    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = xp.zeros(3, dtype = 'float32')
    temp1 = xp.zeros(size + 1, dtype = 'float32')
    temp2 = xp.zeros(size + 1, dtype = 'float32')

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp1[
    int(position[0]):int(size[0]),
    int(position[1]):int(size[1]),
    int(position[2]):int(size[2])
    ] = data
    temp2[
    int(position[0]):int(size[0]),
    int(position[1]):int(size[1]),
    int(position[2]):int(size[2])
    ] = data

    size[dim] = size[dim] - 1
    temp2[0:size[0], 0:size[1], 0:size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] + 1

    out = temp1[
    int(position[0]):int(size[0]),
    int(position[1]):int(size[1]),
    int(position[2]):int(size[2])] 
    return -out


def back_diff(data, step, dim):
    # data --- input image(gpu array!!!)
    # step
    # dim --- determine which is the dimension to calculate derivate
    # dim = 0 --> z axis
    # dim = 1 --> y axis
    # dim = 2 --> x axis
    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = np.zeros(3,dtype='float32')
    temp1 = xp.zeros(size + 1,dtype='float32')
    temp2 = xp.zeros(size + 1,dtype='float32')

    temp1[
    int(position[0]):int(size[0]),
    int(position[1]):int(size[1]),
    int(position[2]):int(size[2])
    ] = data
    temp2[
    int(position[0]):int(size[0]),
    int(position[1]):int(size[1]),
    int(position[2]):int(size[2])
    ] = data
    

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp2[
    int(position[0]):int(size[0]),
    int(position[1]):int(size[1]),
    int(position[2]):int(size[2])
    ] = data
    
    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] - 1
    out = temp1[0:size[0], 0:size[1], 0:size[2]]
    return out

def shrink(x, L):
    s = xp.abs(x)
    xs = xp.sign(x) * xp.maximum(s - 1 / L, 0) 
    return xs

def iter_xx(g, bxx, para, mu):
    """

    :param g: input image
    :param bxx: use to update dxx
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lxx,bxx
    """
    gxx = back_diff(forward_diff(g, 1, 1), 1, 1)
    dxx = shrink(gxx + bxx, mu)
    bxx = bxx + (gxx - dxx)
    Lxx = para * back_diff(forward_diff(dxx - bxx, 1, 1), 1, 1)
    return Lxx, bxx


def iter_xy(g, bxy, para, mu):
    """

    :param g: input image
    :param bxy: use to update dxy
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lxy,bxy
    """
    gxy = forward_diff(forward_diff(g, 1, 1), 1, 2)
    dxy = shrink(gxy + bxy, mu)
    bxy = bxy + (gxy - dxy)
    Lxy = para * back_diff(back_diff(dxy - bxy, 1, 2), 1, 1)
    return Lxy, bxy


def iter_xz(g,bxz,para,mu):
    """

    :param g: input image
    :param bxz: use to update dxz
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lxz,bxz
    """
    gxz = forward_diff(forward_diff(g, 1, 1), 1, 0)
    dxz = shrink(gxz + bxz, mu)
    bxz = bxz + (gxz - dxz)
    Lxz = para * back_diff(back_diff(dxz - bxz, 1, 0), 1, 1)
    return Lxz, bxz

def iter_yy(g, byy, para, mu):
    """

    :param g: input image
    :param byy: use to update dxy
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lyy,byy
    """
    gyy = back_diff(forward_diff(g, 1, 2), 1, 2)
    dyy = shrink(gyy + byy, mu)
    byy = byy + (gyy - dyy)
    Lyy = para * back_diff(forward_diff(dyy - byy, 1, 2), 1, 2)
    return Lyy, byy

def iter_yz(g,byz,para,mu):
    """

    :param g: input image
    :param byz: use to update dxy
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lyz,byz
    """
    gyz = forward_diff(forward_diff(g, 1, 2), 1, 0)
    dyz = shrink(gyz + byz, mu)
    byz = byz + (gyz - dyz)
    Lyz = para * back_diff(back_diff(dyz - byz, 1, 0), 1, 2)
    return Lyz, byz

def iter_zz(g, bzz, para, mu):
    """

    :param g: input image
    :param bzz: use to update dxy
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lzz, bzz
    """
    gzz = back_diff(forward_diff(g, 1, 0), 1, 0)
    dzz = shrink(gzz + bzz, mu)
    bzz = bzz + (gzz - dzz)
    Lzz = para * back_diff(forward_diff(dzz - bzz, 1, 0), 1, 0)
    return Lzz, bzz

def iter_sparse(gsparse,bsparse,para,mu):
    """

    :param gsparse: input image
    :param bzz:  use to update dxy
    :param para: regularization parameters
    :param mu: parameter which is used to tune fidelity term
    :return: Lzz,bzz
    """

    dsparse = shrink(gsparse + bsparse, mu)
    bsparse = bsparse + (gsparse - dsparse)
    Lsparse = para * (dsparse - bsparse)
    return Lsparse, bsparse

```

# utils/background_estimation.py
```python
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.io import imread, imsave, imshow

def Low_frequency_resolve(coeffs, dlevel):
    cAn = coeffs[0]
    vec = []
    vec.append(cAn)
    for i in range(1, dlevel+1):
        (cH, cV, cD) = coeffs[i]
        [cH_x, cH_y] = cH.shape
        cH_new = np.zeros((cH_x, cH_y))
        t = (cH_new, cH_new, cH_new)
        vec.append(t)
    return vec


def rm_1(Biter, x, y):
    Biter_new = np.zeros((x, y), dtype=('uint8'))
    if x%2 and y%2 == 0:
        Biter_new[:, :] = Biter[0:x, :]
    elif x%2 == 0 and y%2:
        Biter_new[:, :]  = Biter[:, 0:y]
    elif x%2 and y%2:
        Biter_new[:, :]  = Biter[0:x, 0:y]
    else:
        Biter_new = Biter
    return Biter_new


def background_estimation(imgs, th = 1, dlevel = 7, wavename = 'db6', iter = 3):
    ''' Background estimation
        function Background = background_estimation(imgs,th,dlevel,wavename,iter)
        imgs: ndarray
            Input image (can be T × X × Y).
        th : int, optional
            if iteration {default:1}
        dlevel : int, optional
         decomposition level {default:7}
        wavename
         The selected wavelet function {default:'db6'}
        iter:  int, optional
         iteration {default:3}
        -----------------------------------------------
        Return:
         Background
    '''
    try:
        [t, x, y] = imgs.shape 
        Background = np.zeros((t, x, y))
        for taxial in range(t):
            img = imgs[taxial, :, :]
            for i in range(iter):
                initial = img
                res = initial
                coeffs = pywt.wavedec2(res, wavelet = wavename, level = dlevel)
                vec = Low_frequency_resolve(coeffs, dlevel)
                Biter = pywt.waverec2(vec, wavelet = wavename)
                Biter_new = rm_1(Biter, x, y)
                if th > 0:
                    eps = np.sqrt(np.abs(res))/2
                    ind = initial>(Biter_new+eps)
                    res[ind] = Biter_new[ind]+eps[ind]
                    coeffs1 = pywt.wavedec2(res, wavelet = wavename, level = dlevel)
                    vec = Low_frequency_resolve(coeffs1, dlevel)
                    Biter =  pywt.waverec2(vec, wavelet = wavename)
                    Biter_new = rm_1(Biter, x, y)
                    Background[taxial, :, :] = Biter_new
    except ValueError:
        [x, y] = imgs.shape 
        Background = np.zeros((x, y))
        for i in range(iter):
            initial = imgs
            res = initial
            coeffs = pywt.wavedec2(res, wavelet = wavename, level = dlevel)
            vec = Low_frequency_resolve(coeffs, dlevel)
            Biter = pywt.waverec2(vec, wavelet = wavename)
            Biter_new = rm_1(Biter, x, y)
            if th > 0:
                eps = np.sqrt(np.abs(res))/2
                ind = initial>(Biter_new+eps)
                res[ind] = Biter_new[ind]+eps[ind]
                coeffs1 = pywt.wavedec2(res, wavelet = wavename, level = dlevel)
                vec = Low_frequency_resolve(coeffs1, dlevel)
                Biter =  pywt.waverec2(vec, wavelet = wavename)
                Biter_new = rm_1(Biter, x, y)
                Background = Biter_new
    return Background
```

# utils/upsample.py
```python
import  warnings
import numpy as np

cp = None
xp = np
def spatial_upsample(SIMmovie,n=2):


    if xp is not cp:
        SIMmovie = SIMmovie
    else:
        SIMmovie = cp.asarray(SIMmovie)

    k = SIMmovie.ndim
    if k > 2:
        [sz,sx,sy] = SIMmovie.shape
        for frames in range(0, sz):
            y = xp.zeros((sz, sx*n,sy*n), dtype = 'float32')
            y = xp.array(y)
            y[frames, 0:sx*n:n, 0:sy*n:n] = SIMmovie[frames,:,:]
            y = xp.array(y)
        return y
    else:
        [sx, sy] = SIMmovie.shape
        y=xp.zeros((sx*n, sy*n), dtype = 'float32')
        y[0:sx * n:n, 0:sy * n:n] = SIMmovie
        return y


def fourier_upsample(imgstack, n = 2):
    ''' 
    Fourier interpolation
    -----------------------------------------------
    imgstack: ndarray
        input image (can be N dimensional).
     n: int, optional
        magnification times  {default: 2}
    ------------------------------------------------
    Output:
      imgfl
    '''
    if xp is not cp:
        imgstack = imgstack
    else:
        imgstack = cp.asarray(imgstack)

    n = n * xp.ones((1, 2))
    if imgstack.ndim < 3:
        z = 1
        sz = [imgstack.shape[0], imgstack.shape[1]]
        imgfl= xp.zeros(( int(n[0][0]) * int(sz[0]), int(n[0][0]) * int(sz[1])))
    else:
        z = imgstack.shape[0]
        sz = [imgstack.shape[1], imgstack.shape[2]]
        imgfl = xp.zeros((z, int(n[0][0]) * int(sz[0]), int(n[0][0]) * int(sz[1])))


    for i in range(0,z):
        if imgstack.ndim < 3:
            img = imgstack
        else:
            img = imgstack[i,:, :]
        imgsz = [img.shape[0], img.shape[1]]
        imgsz = xp.array(imgsz)
        if ((imgsz[0]%2))==1:
            sz = imgsz
        else:
            sz = imgsz-1
        sz = xp.array(sz)
        idx = xp.ceil(sz / 2) + 1 + (n - 1) * xp.floor(sz / 2)
        padsize = [img.shape[0]/2,img.shape[1]/2]
        padsize = xp.array(padsize )
        k = xp.ceil(padsize )
        f=xp.floor(padsize )


       # img = xp.pad(img, [int(k[0]), int(k[1])], 'symmetric')
        img = xp.pad(img, ((int(k[0]), 0), (int(k[1]), 0)), 'symmetric')

        img = xp.pad(img, ((0, int(f[0])), (0, int(f[1]))), 'symmetric')

        im_shape = n*(xp.array(img.shape))
        newsz = xp.floor(im_shape-(n - 1))
        imgl = fInterp_2D(img, newsz)
        if imgstack.ndim < 3:

            imgfl = imgl[int(idx[0][0]):int(n[0][0]) * int(imgsz[0])+int(idx[0][0]), int(idx[0][1]):int(idx[0][1]) + int(n[0][1]) *int(imgsz[1])]
        else:
            imgfl = xp.array(imgfl)

            imgfl[i,:,:] = imgl[int(idx[0][0]):int(n[0][0])*int(imgsz[0])+int(idx[0][0]), int(idx[0][1]):int(idx[0][1])+int(n[0][1])*int(imgsz[1])]

    return imgfl

def fInterp_2D(img, newsz):
    imgsz = img.shape
    imgsz = xp.array(imgsz)
    newsz = xp.array(newsz)
    if (xp.sum(newsz == 0)) >= 1:
        img_ip = []
    isgreater = newsz >= imgsz
    isgreater = isgreater.astype(int)
    isgreater = xp.array(isgreater)
    incr = xp.zeros((2, 1), dtype = 'float32')
    for iDim in range(0,2):
        if isgreater[0][iDim] == 1:
            incr[iDim] = 1
        else:
            incr = xp.floor(imgsz[iDim] / newsz[iDim]) + 1
    newsz[0][0] = int(newsz[0][0])
    a=newsz[0][0]
    b=newsz[0][1]
    nyqst = xp.ceil((imgsz + 1) / 2)
    B = float(a / imgsz[0] * b / imgsz[1])
    img = B * xp.fft.fft2(img)
    img_ip = xp.zeros((int(a), int(b)),dtype = 'complex')
    img_ip[0: int(nyqst[0]), 0: int(nyqst[1])]= img[0: int(nyqst[0]), 0: int(nyqst[1])]#xl, yl
    img_ip[a-(int(imgsz[0])-int(nyqst[0])):a, 0:int(nyqst[1])] = img[int(nyqst[0]):int(imgsz[0]),0:int(nyqst[1])]#xh, yl
    img_ip[0: int(nyqst[0]),a- (int(imgsz[1]) - int(nyqst[1])):a]= img[0: int(nyqst[0]),int( nyqst[1]): int(imgsz[1])]
    img_ip[a-(int(imgsz[0])- int(nyqst[0])):a, a- (int(imgsz[1])-int(nyqst[1])):a]=img[int(nyqst[0]):int(imgsz[0]), int(nyqst[1]):int(imgsz[1])]
    rm = xp.remainder(imgsz, 2)
    if int( rm[0]) == 0 & int(a )!=int(imgsz[0]):
        img_ip[int(nyqst[0]),:] = img_ip[int(nyqst[0]),:] / 2
        img_ip[int(nyqst[0] ) + int(a) - int(imgsz[0]),:] = img_ip[int(nyqst[0]),:]
    if int(rm[1]) == 0 &int(b) != int(imgsz[1]):
        img_ip[ :,int(nyqst[1])] = img_ip[ :,int(nyqst[1])] / 2
        img_ip[:,int(nyqst[1])+int(b)-imgsz[1]] = img_ip[:,int(nyqst[1])]
    img_ip = xp.array(img_ip)
    img_ip =(xp.fft.ifft2(img_ip)). real
    img_ip = img_ip[0: int(a):int(incr[0]), 0:int(b): int(incr[1])]
    return img_ip
```

# sparse_deconv.py
```python
# import cupy as cp
import numpy as np
import warnings
import time
import gc

from matplotlib import pyplot as plt

from .sparse_hessian_recon.sparse_hessian_recon import sparse_hessian
from .iterative_deconv.iterative_deconv import iterative_deconv
from .iterative_deconv.kernel import Gauss
from .utils.background_estimation import background_estimation
from .utils.upsample import spatial_upsample, fourier_upsample
xp = np
def sparse_deconv(img, sigma, sparse_iter = 100, fidelity = 150, sparsity = 10, tcontinuity = 0.5,
                          background = 1, deconv_iter = 7, deconv_type = 1, up_sample = 0):

    """Sparse deconvolution.
   	----------
   	It is an universal post-processing framework for 
   	fluorescence (or intensity-based) image restoration, 
   	including xy (2D), xy-t (2D along t axis), 
   	and xy-z (3D) images. 
   	It is based on the natural priori 
   	knowledge of forward fluorescent 
   	imaging model: sparsity and 
   	continuity along xy-t(z) axes.
   	----------
    Parameters
    ----------
    img : ndarray
       Input image (can be T × X × Y).
    sigma : 1/2/3 element(s) list
       The point spread function size in pixel.
       [x, y, z] dimension
       3D deconv feature is still in progress,
       now is plane-by-plane 2D deconv. 
    sparse_iter:  int, optional
         the iteration of sparse hessian {default: 100}
    fidelity : int, optional
       fidelity {default: 150}
    tcontinuity  : optional
       continuity along z-axial {default: 0.5}
    sparsity :  int, optional
        sparsity {default: 10}
    background:int, optional
        background estimation {default:1}:
        when background is none, 0
        when background is Weak background (High SNR), 1
        when background is Strong background (High SNR), 2
        when background is Weak background (Low SNR), 3
        when background is medium background (Low SNR), 4
        when background is Strong background (Low SNR), 5
    deconv_iter : int, optional
        the iteration of deconvolution {example:7}
    deconv_type : int, optional
       choose the different type deconvolution:
       0: No deconvolution       
       1: Richardson-Lucy deconvolution
       2: LandWeber deconvolution
    up_sample : int, optional
       choose the different type upsampling (x2) operation:
       0: No upsampling       
       1: Fourier upsampling
       2: Spatial upsampling (should decrease the fidelity & sparsity)

    Returns
    -------
    img_last : ndarray
       The sparse deconvolved image.

    Examples
    --------
    >>> from sparse_recon.sparse_deconv import sparse_deconv
	>>> from skimage import io
    >>> img = io.imread('test.tif')
	>>> img_recon = sparse_deconv(img, [5,5])
    References
    ----------
      [1] Weisong Zhao et al. Sparse deconvolution improves
      the resolution of live-cell super-resolution 
      fluorescence microscopy, Nature Biotechnology (2022),
      https://doi.org/10.1038/s41587-021-01092-2
    """
    if not sigma:
        print("The PSF's sigma is not given, turning off the iterative deconv...")
        deconv_type = 0
    img = np.array(img, dtype = 'float32')
    scaler = np.max(img)
    img = img / scaler

    if background == 1:
        backgrounds = background_estimation(img / 2.5)
        img = img - backgrounds
    elif background == 2:
        backgrounds = background_estimation(img / 2)
        img = img - backgrounds
    elif background == 3:
        medVal = np.mean(img) / 2.5
        img[img > medVal] = medVal
        backgrounds = background_estimation(img)
        img = img - backgrounds
    elif background== 4:
        medVal = np.mean(img)/2
        img[img> medVal] = medVal
        backgrounds = background_estimation(img)
        img = img - backgrounds
    elif background == 5:
        medVal = np.mean(img)
        img[img > medVal] = medVal
        backgrounds = background_estimation(img)
        img = img - backgrounds


    img = img / (img.max())
    img[img < 0] = 0

    if up_sample == 1:
        img = fourier_upsample(img)
    elif up_sample == 2:
        img = spatial_upsample(img)

    img = img / (img.max())

    start = time.process_time()

    gc.collect()
    # xp.clear_memo()

    img_sparse = sparse_hessian(img, sparse_iter, fidelity, sparsity, tcontinuity)
    end = time.process_time()
    print('sparse-hessian time %0.2fs' % (end - start))
    img_sparse = img_sparse / (img_sparse.max())
    if deconv_type == 0:
        img_last = img_sparse
        return scaler * img_last
    else:
        start = time.process_time()
        kernel = Gauss(sigma)
        img_last = iterative_deconv(img_sparse, kernel, deconv_iter, rule = deconv_type)
        end = time.process_time()
        print('deconv time %0.2fs' % (end - start))
        return scaler * img_last

```

# main.py
```python
from sparse_deconv import sparse_deconv
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
import numpy as np

# 配置参数
PSF_FACTOR = 280 /65
SPARSE_ITER = 1000    

# 输入和输出文件路径（请根据实际情况修改）
input_path = "001.tif"
output_path = "output.tif"
expected_output_path = "expected_output.tif"  # 用于对比的预期输出文件路径

# 读取图像
im = io.imread(input_path)

# 执行稀疏反卷积
img_recon = sparse_deconv(im, PSF_FACTOR, sparse_iter=SPARSE_ITER)

# 保存结果，保持原始数据类型
io.imsave(output_path, img_recon.astype(im.dtype))

print(f"✅ Processing complete! Result saved to: {output_path}")

# 加载预期输出图像
expected = io.imread(expected_output_path)

# 确保形状和数据类型匹配（如果不匹配，可能需要调整）
if img_recon.shape != expected.shape:
    raise ValueError("Reconstructed image and expected image must have the same shape!")

# 计算 PSNR
psnr = peak_signal_noise_ratio(expected, img_recon, data_range=expected.max() - expected.min())

# 计算 SSIM（结构相似性指数）
ssim_val = ssim(expected, img_recon, data_range=expected.max() - expected.min(), channel_axis=None if len(expected.shape) == 2 else 2)

# 计算 MSE（均方误差）
mse = np.mean((expected.astype(np.float64) - img_recon.astype(np.float64)) ** 2)

print(f"📊 PSNR: {psnr:.4f} dB")
print(f"📊 SSIM: {ssim_val:.4f}")
print(f"📊 MSE: {mse:.6f}")
```