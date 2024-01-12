import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import pdb
from training import augmentation
import operator 
def is_num(a):
    return isinstance(a, int) or isinstance(a, float)


def delta(x1, x2):
    delta_ = x2 - x1
    return delta_ // 2, delta_ - delta_ // 2


def get_padding_width(o_shape, d_shape):
    if is_num(o_shape):
        o_shape, d_shape = [o_shape], [d_shape]
    assert len(o_shape) == len(d_shape), 'Length mismatched!'
    borders = []
    for o, d in zip(o_shape, d_shape):
        borders.extend(delta(o, d))
    return borders


def get_crop_width(o_shape, d_shape):
    return get_padding_width(d_shape, o_shape)


def get_padding_shape_with_stride(o_shape, stride):
    assert isinstance(o_shape, list) or isinstance(o_shape, tuple) or isinstance(o_shape, np.ndarray)
    o_shape = np.array(o_shape)
    d_shape = np.ceil(o_shape / stride) * stride
    return d_shape.astype(np.int32)


def pad(arr, d_shape, mode='constant', value=0, strict=True):
    """
    pad numpy array, tested!
    :param arr: numpy array
    :param d_shape: array shape after padding or minimum shape
    :param mode: padding mode,
    :param value: padding value
    :param strict: if True, d_shape must be greater than arr shape and output shape is d_shape. if False, d_shape is minimum shape and output shape is np.maximum(arr.shape, d_shape)
    :return: padded arr with expected shape
    """
    assert arr.ndim == len(d_shape), 'Dimension mismatched!'
    if not strict:
        d_shape = np.maximum(arr.shape, d_shape)
    else:
        assert np.all(np.array(d_shape) >= np.array(arr.shape)), 'Padding shape must be greater than arr shape'
    borders = np.array(get_padding_width(arr.shape, d_shape))
    before = borders[list(range(0, len(borders), 2))]
    after = borders[list(range(1, len(borders), 2))]
    padding_borders = tuple(zip([int(x) for x in before], [int(x) for x in after]))
    # print(padding_borders)
    if mode == 'constant':
        return np.pad(arr, padding_borders, mode=mode, constant_values=value)
    else:
        return np.pad(arr, padding_borders, mode=mode)


def crop(arr, d_shape, strict=True):
    """
    central  crop numpy array, tested!
    :param arr: numpy array
    :param d_shape: expected shape
    :return: cropped array with expected array
    """
    assert arr.ndim == len(d_shape), 'Dimension mismatched!'
    if not strict:
        d_shape = np.minimum(arr.shape, d_shape)
    else:
        assert np.all(np.array(d_shape) <= np.array(arr.shape)), 'Crop shape must be smaller than arr shape'
    borders = np.array(get_crop_width(arr.shape, d_shape))
    start = borders[list(range(0, len(borders), 2))]
    # end = - borders[list(range(1, len(borders), 2))]
    end = map(operator.add, start, d_shape)
    slices = tuple(map(slice, start, end))
    return arr[slices]


def pad_crop(arr, d_shape, mode='constant', value=0):
    """
    pad or crop numpy array to expected shape, tested!
    :param arr: numpy array
    :param d_shape: expected shape
    :param mode: padding mode,
    :param value: padding value
    :return: padded and cropped array
    """
    assert arr.ndim == len(d_shape), 'Dimension mismatched!'
    arr = pad(arr, d_shape, mode, value, strict=False)
    return crop(arr, d_shape)

# def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
#     a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
#     a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

#     _, h, w = a_src.shape
#     b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
#     c_h = np.floor(h/2.0).astype(int)
#     c_w = np.floor(w/2.0).astype(int)
#     # print (b)
#     h1 = c_h-b
#     h2 = c_h+b+1
#     w1 = c_w-b
#     w2 = c_w+b+1

#     ratio = random.randint(1,10)/10

#     a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
#     # a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
#     # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
#     a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
#     # a_trg[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2]
#     # a_trg = np.fft.ifftshift( a_trg, axes=(-2, -1) )
#     return a_src

class CMRDataset(Dataset):
    def __init__(self, args, mode='train', k_fold=5, k=0, seed=0):
        
        self.mode = mode
        self.args = args

        assert mode in ['train', 'test']

        with open(os.path.join(args.data_root, 'list', 'dataset.yaml'), 'r') as f:
            img_name_list = yaml.load(f, Loader=yaml.SafeLoader)


        random.Random(seed).shuffle(img_name_list)

        length = len(img_name_list)
        test_name_list = img_name_list[k*(length//k_fold) : (k+1)*(length//k_fold)]
        train_name_list = list(set(img_name_list) - set(test_name_list))

        if mode == 'train':
            img_name_list = train_name_list
        else:
            img_name_list = test_name_list


        print('Start loading %s data'%self.mode)

        path = args.data_root

        self.img_list = []
        self.lab_list = []
        self.img_amp_list = []
        self.img_pha_list = []
        self.spacing_list = []

        for name in img_name_list:
            for idx in [0, 1]:
                
                img_name = name + '_%d.nii.gz'%idx
                lab_name = name + '_%d_gt.nii.gz'%idx

                itk_img = sitk.ReadImage(os.path.join(path, img_name))
                itk_lab = sitk.ReadImage(os.path.join(path, lab_name))

                spacing = np.array(itk_lab.GetSpacing()).tolist()
                self.spacing_list.append(spacing[::-1])  # itk axis order is inverse of numpy axis order

                assert itk_img.GetSize() == itk_lab.GetSize()

                img, lab = self.preprocess(itk_img, itk_lab)
                fft=np.fft.fftn(img)
                amp_np, pha_np = np.abs(fft), np.angle(fft)

                self.img_list.append(img)
                self.lab_list.append(lab)
                #Amp and pha precalculate, wait for augmentation
                self.img_amp_list.append(amp_np)
                self.img_pha_list.append(pha_np)

        self.amps=np.load("/filer/tmp1/xh172/Derivate1.npz")["amp"]

        print('Load done, length of dataset:', len(self.img_list))
        print('Target Amp Load Done, length of the target amps',len(self.amps)) 

    def __len__(self):
        if self.mode == 'train':
            return len(self.img_list) * 100000
        else:
            return len(self.img_list)

    def preprocess(self, itk_img, itk_lab):
        
        img = sitk.GetArrayFromImage(itk_img)
        lab = sitk.GetArrayFromImage(itk_lab)

        max98 = np.percentile(img, 98)
        img = np.clip(img, 0, max98)

        z, y, x = img.shape
        
        img=pad_crop(img,self.args.training_size)
        lab=pad_crop(lab,self.args.training_size)
        # pad if the image size is smaller than trainig size
        # if z < self.args.training_size[0]:
        #     diff = (self.args.training_size[0]+2 - z) // 2
        #     img = np.pad(img, ((diff, diff), (0,0), (0,0)))
        #     lab = np.pad(lab, ((diff, diff), (0,0), (0,0)))
        # if y < self.args.training_size[1]:
        #     diff = (self.args.training_size[1]+2 - y) // 2
        #     img = np.pad(img, ((0,0), (diff,diff), (0,0)))
        #     lab = np.pad(lab, ((0,0), (diff, diff), (0,0)))
        # if x < self.args.training_size[2]:
        #     diff = (self.args.training_size[2]+2 - x) // 2
        #     img = np.pad(img, ((0,0), (0,0), (diff, diff)))
        #     lab = np.pad(lab, ((0,0), (0,0), (diff, diff)))


        img = img / max98
        # def remove_background(img, lab, size=256):
        #     z, y, x = img.shape
        #     if y > size:
        #         img = img[:, y//2-size//2:y//2+size//2, :]
        #         lab = lab[:, y//2-size//2:y//2+size//2, :]
        #     if x > size:
        #         img = img[:, :, x//2-size//2:x//2+size//2]
        #         lab = lab[:, :, x//2-size//2:x//2+size//2]

        #     return img, lab
        # img, lab = remove_background(img, lab, size=256)
        
        
        img = img.astype(np.float32)
        lab = lab.astype(np.uint8)

        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()
        
        return tensor_img, tensor_lab

    def __getitem__(self, idx):
        
        fft_index = int(np.floor((idx / len(self.img_list))) % len(self.amps))


        idx = idx % len(self.img_list)
        
        a_src=np.fft.fftshift(self.amps[fft_index])
        a_trg=np.fft.fftshift(self.img_amp_list[idx])

        h,w,d = a_src.shape
        #Here we need to change L
        b = (np.floor(np.amin((w,d))*0.1)  ).astype(int)
        h_d=np.floor(h*0.1).astype(int)

        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)
        c_d = np.floor(d/2.0).astype(int)
        
        h1 = c_h-h_d
        h2 = c_h+h_d+1
        w1 = c_w-b
        w2 = c_w+b+1
        d1 = c_d-b
        d2 = c_d+b+1
        
        # ratio = random.randint(1,10)/10
        
        a_src[h1:h2,w1:w2,d1:d2] = a_trg[h1:h2,w1:w2,d1:d2]

        a_src = np.fft.ifftshift( a_src )

        fft_src_ = a_src * np.exp( 1j * self.img_pha_list[idx])
        src_in_trg = np.fft.ifftn( fft_src_ )
        src_in_trg = np.real(src_in_trg)



        tensor_img = src_in_trg.astype(np.float32)
        tensor_img = torch.from_numpy(tensor_img).float()
        tensor_lab = self.lab_list[idx]

        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
        tensor_lab = tensor_lab.unsqueeze(0).unsqueeze(0)
        # 1, C, D, H, W


        if self.mode == 'train':
            # Gaussian Noise
            tensor_img = augmentation.gaussian_noise(tensor_img, std=self.args.gaussian_noise_std)
            # Additive brightness
            tensor_img = augmentation.brightness_additive(tensor_img, std=self.args.additive_brightness_std)
            # gamma
            tensor_img = augmentation.gamma(tensor_img, gamma_range=self.args.gamma_range, retain_stats=True)
            
            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='random')

        #else:
        #    tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab,self.args.training_size, mode='center')

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        assert tensor_img.shape == tensor_lab.shape

        if self.mode == 'train':
            return tensor_img, tensor_lab
        else:
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])

            
