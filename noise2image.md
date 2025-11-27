# main.py
```python
"""Training noise2image model on synthetic data of noise events."""

from argparse import ArgumentParser
import os
import random
import numpy as np
import torch
import numbers
from torchvision import transforms
from PIL import Image
from sklearn.linear_model import LinearRegression
import scipy
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage import io
import cv2
import numba as nb
from h5 import NPYEventsReader
from typing import Tuple



parser = ArgumentParser()
parser.add_argument("--pixel_bin", type=int, default=2, 
                    help="Pixel binning during the event aggregation.")
parser.add_argument("--aug_contrast", action='store_true', 
                    help="Augment image contrast during synthetic training.")

num_photon_scalar= 1.0050
num_time= 21.9882
eps_pos = 0.9040
eps_neg = 1.0235
bias_pr= 1.7023
illum_offset = 0.1930
constant_noise_neg = 0.1398
H_inv = np.array(
    [[1.05374157e+00, 1.29773432e-02, -4.31484473e+01],
     [-1.14301830e-03, 1.08660669e+00, -2.24931360e+01],
     [-1.49652303e-05, 3.06551912e-05, 9.99973086e-01]])

def _is_numpy_image(img):
    return isinstance(img, np.ndarray)

class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a probability of 0.5."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, pic):
        """
        Args:
            img (numpy array): Image to be flipped.
        Returns:
            numpy array: Randomly flipped image.
        """

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make it three channel
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        if random.random() < self.prob:
            return pic[:, ::-1, :]
        return pic

class RandomVerticalFlip(object):
    """Vertically flip the given numpy array randomly with a probability of 0.5 by default."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, pic):
        """
        Args:
            img (numpy array): Image to be flipped.
        Returns:
            numpy array: Randomly flipped image.
        """

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make it three channel
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        if random.random() < self.prob:
            return pic[::-1, :, :]
        return pic

class CenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(pic, output_size):
        """Get parameters for ``crop`` for center crop.
        Args:
            pic (np array): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to the crop for center crop.
        """

        h, w, c = pic.shape
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return i, j, th, tw

    def __call__(self, pic):
        """
        Args:
            pic (np array): Image to be cropped.
        Returns:
            np array: Cropped image.
        """

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make them 3
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        # get crop params: starting pixels and size of the crop
        i, j, h, w = self.get_params(pic, self.size)

        return pic[i:i + h, j:j + w, :]

class EventNoiseCountWrapper(object):
    def __init__(self, num_photon_scalar=50, num_time=100, eps_pos=0.1, eps_neg=0.1, bias_pr=0.0, illum_offset=0.0,
                 constant_noise_neg=0.0, polarity=True, pixel_bin=1, varying_eps=False, poisson_sample=False):
        self.event_obj = EventNoiseCount(num_photon_scalar=num_photon_scalar, num_time=num_time, eps_pos=eps_pos,
                                         eps_neg=eps_neg, bias_pr=bias_pr, illum_offset=illum_offset,
                                         constant_noise_neg=constant_noise_neg, polarity=polarity,
                                         pixel_bin=pixel_bin, output_numpy=True, varying_eps=varying_eps,
                                         poisson_sample=poisson_sample)
        self.pixel_bin = pixel_bin

    def __call__(self, input_mat):
        sample = Image.fromarray(input_mat[:, :, -1]*255)
        out = self.event_obj(sample)
        if self.pixel_bin > 1:
            img = input_mat[:, :, -1].reshape((input_mat.shape[0] // self.pixel_bin, self.pixel_bin, input_mat.shape[1] // self.pixel_bin, self.pixel_bin)).mean(axis=(1, 3))[..., np.newaxis]
        else:
            img = input_mat[:, :, -1, np.newaxis]
        out = np.concatenate([out.transpose((1, 2, 0)), img], axis=-1)
        return out

class EventNoiseCount(object):

    def __init__(self, num_photon_scalar=50, num_time=100, eps_pos=0.1, eps_neg=0.1, bias_pr=0.0, illum_offset=0.0,
                 constant_noise_neg=0.0, polarity=True, pixel_bin=1, output_numpy=False, varying_eps=False,
                 poisson_sample=False):
        self.num_photon_scalar = num_photon_scalar
        self.num_time = num_time
        self.eps_pos = eps_pos
        self.eps_neg = eps_neg
        self.bias_pr = bias_pr
        self.illum_offset = illum_offset
        self.constant_noise_neg = constant_noise_neg
        self.polarity = polarity
        self.pixel_bin = pixel_bin
        self.output_numpy = output_numpy
        self.varying_eps = varying_eps
        self.poission_sample = poisson_sample  # by default sample from negative binomial

        with np.load('lux_measurement.npz') as data:
            list_lux_close = data['list_lux_close']
            list_lux_far = data['list_lux_far']
            list_intensity = data['list_intensity']
        self.reg = LinearRegression().fit(list_intensity.reshape(-1, 1), list_lux_close.reshape(-1, 1))
        self.reg_far = LinearRegression().fit(list_intensity.reshape(-1, 1), list_lux_far.reshape(-1, 1))
        self.illuminance_level = self.reg.predict((np.arange(256) ** 2.2).reshape(-1, 1))
        self.illuminance_level_far = self.reg_far.predict((np.arange(256) ** 2.2).reshape(-1, 1))
        self.illuminance_level = self.illuminance_level / np.max(self.illuminance_level) * np.max(self.illuminance_level_far)

        with np.load('synthetic_param.npz') as f:
            self.negative_binomial_r_pos = f['r_pos']
            self.negative_binomial_r_neg = f['r_neg']
        self.rng = np.random.default_rng(seed=int(torch.randint(2**32, (1,))))

    def __call__(self, sample):

        # convert to grayscale, load into numpy
        im_arr = np.array(sample.convert('L')).astype(np.float32)
        im_arr = im_arr.reshape((sample.size[1], sample.size[0]))

        r_pos = np.interp(im_arr, np.linspace(0, 255, len(self.negative_binomial_r_pos)), self.negative_binomial_r_pos)
        r_neg = np.interp(im_arr, np.linspace(0, 255, len(self.negative_binomial_r_neg)), self.negative_binomial_r_neg)
        im_arr_gamma = np.squeeze(self.illuminance_level[im_arr.astype(np.int32)], axis=-1)

        # actual forward noisy model
        random_scale = self.rng.uniform(0.8, 1.2)  # 0.8, 1.2
        eps_pos = self.eps_pos if not self.varying_eps else self.eps_pos * random_scale
        eps_neg = self.eps_neg if not self.varying_eps else self.eps_neg * random_scale
        # p = (1 - scipy.special.erf(self.eps * ((im_arr_gamma + self.luminance_offset) * self.num_photon_scalar + self.bias_pr) / np.sqrt(2 * (im_arr_gamma + self.luminance_offset) * self.num_photon_scalar + 1e-9))) / 2
        p_pos = (1 - scipy.special.erf((np.exp(eps_pos) - 1) * ((im_arr_gamma + self.illum_offset) * self.num_photon_scalar + self.bias_pr) / np.sqrt(
            2 * (im_arr_gamma + self.illum_offset) * self.num_photon_scalar * (1 + np.exp(eps_pos)))))
        p_neg = (1 - scipy.special.erf((np.exp(eps_neg) - 1) * ((im_arr_gamma + self.illum_offset) * self.num_photon_scalar + self.bias_pr) / np.sqrt(
            2 * (im_arr_gamma + self.illum_offset) * self.num_photon_scalar * (1 + np.exp(eps_neg)))))

        if self.poission_sample:
            count_pos = self.rng.poisson((p_pos * self.num_time))
            count_neg = self.rng.poisson((p_neg * self.num_time))
        else:
            count_pos = self.rng.negative_binomial(r_pos, r_pos / (p_pos * self.num_time + r_pos))
            count_neg = self.rng.negative_binomial(r_neg, r_neg / (p_neg * self.num_time + r_neg))

        if self.constant_noise_neg > 0:
            noise_neg = self.constant_noise_neg
            count_neg += self.rng.poisson(noise_neg, size=p_neg.shape)

        if self.polarity:
            total_count = np.stack([count_pos, count_neg], axis=0)
        else:
            total_count = count_pos + count_neg
            total_count = total_count[np.newaxis]

        if self.pixel_bin > 1:
            total_count = total_count.reshape((total_count.shape[0], total_count.shape[1] // self.pixel_bin, self.pixel_bin,
                                               total_count.shape[2] // self.pixel_bin, self.pixel_bin)).mean(axis=(2, 4))
            im_arr = im_arr.reshape((im_arr.shape[0] // self.pixel_bin, self.pixel_bin,
                                     im_arr.shape[1] // self.pixel_bin, self.pixel_bin)).mean(axis=(1, 3))

        if self.output_numpy:
            return total_count.astype(np.float32)
        else:
            return torch.from_numpy(total_count.astype(np.float32)), torch.from_numpy(im_arr.astype(np.float32)[np.newaxis]/255), torch.tensor(1.0, dtype=torch.float32)

class EventCountNormalization(object):
    def __init__(self, integration_time_s=1):
        self.integration_time_s = integration_time_s

    def __call__(self, mat):
        num_channels = mat.shape[-1] - 1
        scalar = 1 / 20 * (num_channels / 2) / self.integration_time_s
        mat[..., :-1] *= scalar
        return mat

class AugmentImageContrast(object):
    def __init__(self, max_scale, min_scale, seed=19358):
        assert (min_scale >= 0) and (max_scale >= min_scale)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.rng = np.random.default_rng(seed)

    def __call__(self, input_mat):
        scale = self.rng.uniform(self.min_scale, self.max_scale)
        base_level = 0  # self.rng.uniform(0, 1 - scale)
        input_mat[:, :, -1] = input_mat[:, :, -1] * scale + base_level
        return input_mat

def calibrate_distortion(calib_img):
    gt_img = cv2.resize(cv2.imread('./calibration.png'), (1280, 720))
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)

    checker_size = (37, 20)
    ret, corners = cv2.findChessboardCorners(gt_img, checker_size)

    ret_m, corners_m = cv2.findChessboardCorners(calib_img, checker_size)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    corners2 = cv2.cornerSubPix(gt_img, corners, (11, 11), (-1, -1), criteria)
    corners_m2 = cv2.cornerSubPix(calib_img, corners_m, (11, 11), (-1, -1), criteria)

    # Compute the homography matrix
    H, _ = cv2.findHomography(corners2, corners_m2)

    # To reverse the transformation on another image (for example, 'm')
    # First, invert the homography matrix
    H_inv = np.linalg.inv(H)

    return H_inv

def undistort(img, H_inv_, dim_xy=(1280, 720)):
    H_inv_cur = H_inv_.copy()
    H_inv_cur[2, :2] = H_inv_cur[2, :2] * 720 / dim_xy[1]
    H_inv_cur[:2, 2] = H_inv_cur[:2, 2] * dim_xy[1] / 720
    return cv2.warpPerspective(img, H_inv_cur, dsize=dim_xy, flags=cv2.INTER_NEAREST).astype(np.float32)

@nb.jit(nopython=True)
def count_events(events_, dim_xy_):
    count = np.zeros((dim_xy_[1], dim_xy_[0]), dtype=np.float32)
    for e in events_:
        count[e[1], e[0]] += 1
    return count

def sparse_diff(x, y, t, shape: Tuple[int, int]=(720, 1280)):
    
    coordinate = x*shape[1] + y #convert to linear index
    stable_argsorted = torch.argsort(coordinate, stable=True) #argsort by linear index
    sorted_coordinate = coordinate[stable_argsorted] #sort coordinate
    sorted_t = t[stable_argsorted] #sort t by coordinate
    coord_switch = np.where(np.diff(sorted_coordinate) > 0)[0]+1 #find where coordinate changes
    t_diff = torch.diff(
        sorted_t,
        prepend=torch.zeros(1, device=x.device)
    ).to(dtype=sorted_t.dtype) #compute time difference between event times
    t_diff[coord_switch] = 0#set time difference to 0 where coordinate changes
    
    nevents = (
        -1 + #Subtract 1 to account for first event
        torch.sparse_coo_tensor(
            coordinate.unsqueeze(0),
            torch.ones_like(t_diff),
            size=(shape[0]*shape[1],)
        ).to_dense()
    ).clamp(min=1) #compute number of events per pixel. 

    means = torch.sparse_coo_tensor(
        coordinate.unsqueeze(0),
        t_diff,
        size=(shape[0]*shape[1],)
    ).to_dense() / nevents #compute mean time difference per pixel
    means = means[sorted_coordinate] #extract mean time difference per pixel in original order
    t_diff = (t_diff - means) ** 2 #compute squared difference from mean
    t_diff[coord_switch] = 0 #set squared difference to 0 where coordinate changes
        
    sp = torch.sparse_coo_tensor(
        coordinate.unsqueeze(0),
        t_diff,
        size=(shape[0]*shape[1],)
    ) #convert to sparse tensor for sum reduction
    
    return (sp.to_dense()/nevents.clamp(1)).reshape(shape)

class EventImagePairDataset(Dataset):
    def __init__(self, image_folder, event_folder,
                 integration_time_s=1, total_time_s=10, start_time_s=-1, time_bin=1, pixel_bin=1,
                 polarity=False, std_channel=False, n_limit=None, transform=None,
                 img_suffix='.jpg', calib_img_path=None):
        """
        :param image_folder: folder containing images
        :param event_folder: folder containing event recordings
        :param integration_time_s: integration time in seconds, if -1, randomly sample from [1, total_time_s]
        :param total_time_s: total time of the event recording in seconds
        :param start_time_s: start time of the event recording in seconds, if -1, randomly sample from [0, total_time_s - integration_time_s]
        :param time_bin: number of time bins to split the integration time into (for temporal resolution)
        :param pixel_bin: average pooling kernel size, if 1, no pooling
        :param polarity: whether to count polarity events separately
        :param std_channel: whether to add a channel of event interval standard deviation
        :param n_limit: limit the number of images to load
        :param transform: transform to apply to the image and event count matrix for data augmentation
        :param img_suffix: image file suffix. Default is '.jpg'
        :param calib_img_path: path to the calibration image
        """
        super().__init__()
        _image_files = [f for f in sorted(os.listdir(image_folder)) if f.endswith(img_suffix)]
        image_files, images, event_files, event_images = [], [], [], []
        self.event_dim_xy = [1280, 720]
        self.output_dim_xy = [1280 // pixel_bin, 720 // pixel_bin]
        self.integration_time_s = integration_time_s
        self.total_time_s = total_time_s
        self.start_time_s = start_time_s
        self.time_bin = time_bin
        self.pixel_bin = pixel_bin
        self.polarity = polarity
        self.std_channel = std_channel
        self.transform = transform

        for i in tqdm(range(len(_image_files) if n_limit is None else n_limit)):
            image = io.imread(os.path.join(image_folder, _image_files[i]))

            if image.shape[:2] == (1080, 1920):
                image_files.append(_image_files[i])
                event_files.append(os.path.join(event_folder, _image_files[i].split('.')[0]))
                
            else:
                print(f'Image {_image_files[i]} has invalid shape {image.shape}')
        self.image_files = image_files
        self.event_files = event_files
        self.folder = image_folder
        self.event_folder = event_folder

        if calib_img_path is not None:
            calib_img = cv2.resize(cv2.imread(calib_img_path), (1280, 720))
            calib_img = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)
            calib_img = cv2.equalizeHist(calib_img)
            calib_img[calib_img<127]=0
            calib_img[calib_img>=127]=255
            self.H_inv = calibrate_distortion(io.imread(calib_img_path))
        else:
            self.H_inv = H_inv

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        integration_time_s = self.integration_time_s if self.integration_time_s > 0 \
            else np.clip(np.random.normal(1.0, 4), 1.0, self.total_time_s).astype(np.float32)
        events = self.load_events(self.event_files[idx], integration_time_s)
        time_bin_start = np.linspace(events['t'][0], events['t'][-1], self.time_bin + 1)[:-1]
        bin_ind = np.searchsorted(events['t'], time_bin_start, side='right')
        events['x'] = events['x'] // self.pixel_bin
        events['y'] = events['y'] // self.pixel_bin
        count_bin = np.zeros((self.output_dim_xy[1], self.output_dim_xy[0],
                              self.time_bin * 2 if self.polarity else self.time_bin), dtype=np.float32)
        for i in range(self.time_bin):
            e = events[bin_ind[i]:(bin_ind[i + 1] if i + 1 < self.time_bin else None)]

            if self.polarity:
                count_pos = undistort(count_events(e[e['p'] == 1], nb.typed.List(self.output_dim_xy)), self.H_inv, self.output_dim_xy)
                count_neg = undistort(count_events(e[e['p'] == 0], nb.typed.List(self.output_dim_xy)), self.H_inv, self.output_dim_xy)
                count_bin[:, :, i * 2] = count_pos / self.pixel_bin / self.pixel_bin
                count_bin[:, :, i * 2 + 1] = count_neg / self.pixel_bin / self.pixel_bin
            else:
                count = undistort(count_events(e, nb.typed.List(self.output_dim_xy)), self.H_inv, self.output_dim_xy)
                count_bin[:, :, i] = count / self.pixel_bin / self.pixel_bin

        image = self.load_image(os.path.join(self.folder, self.image_files[idx]))

        if self.std_channel:
            event_std = sparse_diff(x=torch.from_numpy(events['y'].astype(np.int64)),
                                    y=torch.from_numpy(events['x'].astype(np.int64)),
                                    t=torch.from_numpy(events['t'].astype(np.int64)),
                                    shape=(self.output_dim_xy[1], self.output_dim_xy[0]))
            event_std = undistort(event_std.numpy()**0.5 * 1e-6, self.H_inv, self.output_dim_xy)
            count_bin = np.concatenate([count_bin, event_std[..., np.newaxis]], axis=-1)

        combined = np.concatenate([count_bin, image[..., np.newaxis]], axis=-1)
        if self.transform is not None:
            combined = self.transform(combined)

        return torch.from_numpy(combined[..., :-1].copy().transpose((2, 0, 1))), torch.from_numpy(combined[..., -1:].copy().transpose((2, 0, 1))), torch.tensor(integration_time_s, dtype=torch.float32)

    def load_events(self, event_file_path, integration_time_s):
        # h = H5EventsReader(event_file_path)
        h = NPYEventsReader(event_file_path)
        
        start_us = self.start_time_s * 1e6
        if self.start_time_s == -1:
            start_us = torch.randint(0, int((self.total_time_s - integration_time_s) * 1e6 + 1), (1,)).item()

        try:
            events = h.read_interval(start_us, start_us + integration_time_s * 1e6)
        except Exception as e:
            print(f'Error reading {event_file_path}, start {start_us}, as {e}')
            if integration_time_s < 3:
                start_us = (start_us + 2 * integration_time_s * 1e6) % int((self.total_time_s - integration_time_s) * 1e6)
            else:
                start_us = torch.randint(0, int((self.total_time_s - integration_time_s) * 1e6), (1,)).item()

            print('Trying again with start', start_us)
            events = h.read_interval(start_us, start_us + integration_time_s * 1e6)
        return events

    def load_image(self, image_file_path):
        image = io.imread(image_file_path, as_gray=True)
        image = cv2.resize(image, (self.event_dim_xy[0] // self.pixel_bin, self.event_dim_xy[1] // self.pixel_bin))
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.
        return image.astype(np.float32)

if __name__ == '__main__':
    INDIST_EVENT_PATH = './data/indist_events/'
    INDIST_IMAGE_PATH = './data/indist_images/'
    OOD_EVENT_PATH = './data/ood_DIV2K_events/'
    OOD_IMAGE_PATH = './data/ood_DIV2K_images/'
    args = parser.parse_args()

    input_size = (720 // args.pixel_bin // 8 * 8 * args.pixel_bin, 1280 // args.pixel_bin // 8 * 8 * args.pixel_bin)
    input_size_ds = [input_size[0]//args.pixel_bin, input_size[1]//args.pixel_bin]

    transformer = [RandomHorizontalFlip(),
                       RandomVerticalFlip(),
                       CenterCrop(input_size),
                       EventNoiseCountWrapper(num_photon_scalar=num_photon_scalar,
                                                    num_time=num_time, eps_pos=eps_pos,
                                                    eps_neg=eps_neg, bias_pr=bias_pr,
                                                    illum_offset=illum_offset,
                                                    constant_noise_neg=constant_noise_neg,
                                                    pixel_bin=args.pixel_bin, varying_eps=True),
                       EventCountNormalization()]
    if args.aug_contrast:
        transformer.insert(3, AugmentImageContrast(max_scale=1.3, min_scale=0.7))

    ds = EventImagePairDataset(image_folder=INDIST_IMAGE_PATH,
                                     event_folder=INDIST_EVENT_PATH,
                                     integration_time_s=1, total_time_s=10, start_time_s=5,
                                     time_bin=1, pixel_bin=1, polarity=True, std_channel=False,
                                     transform=transforms.Compose(transformer))
```