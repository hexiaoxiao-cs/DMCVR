import SimpleITK as sitk
import torch
import SimpleITK as sitk
import numpy as np
import glob
import os
import tqdm
import operator
import torch
import yaml
import collections
import argparse
import pickle
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

crop_size=[16, 256, 256]
target_spacing=[0.7422,0.7422,8.0]
target_spacing.reverse() #This is because the italy dataset spacing is reversed
def ResampleXYZAxis(imImage, space=(1., 1., 1.), interp=sitk.sitkLinear):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    sp1 = imImage.GetSpacing()
    sz1 = imImage.GetSize()

    sz2 = (int(round(sz1[0]*sp1[0]*1.0/space[0])), int(round(sz1[1]*sp1[1]*1.0/space[1])), int(round(sz1[2]*sp1[2]*1.0/space[2])))

    imRefImage = sitk.Image(sz2, imImage.GetPixelIDValue())
    imRefImage.SetSpacing(space)
    imRefImage.SetOrigin(imImage.GetOrigin())
    imRefImage.SetDirection(imImage.GetDirection())

    imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

    return imOutImage

import sys
# i=int(sys.argv[1])

#source_folder=glob.glob("/filer/tmp1/xh172/ukbb/test_extracted/")
# target_folder="/filer/tmp1/xh172/Derivate1_spacing_regularized_segmented/"
os.environ['CUDA_VISIBLE_DEVICES']="0"
save_to="/filer/tmp1/xh172/ukbb_cq_latent_code"
if not os.path.exists(os.path.join(save_to,"img")):
    os.makedirs(os.path.join(save_to,"img"))

#Here we load the model
model_path="./exp/biobank/test/fold_3_best.pth"
#Here we load the configuration file
config_path="./config/biobank/medformer_2d.yaml"
# parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
# args = vars()
with open(config_path, 'r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
# args.update(config)
# for key, value in config.items():
#     setattr(args, key, value)
from model.dim2 import MedFormer
net=MedFormer(args['in_chan'], args['classes'], args['base_chan'], map_size=args['map_size'], conv_block=args['conv_block'], conv_num=args['conv_num'], trans_num=args['trans_num'], num_heads=args['num_heads'], fusion_depth=args['fusion_depth'], fusion_dim=args['fusion_dim'], fusion_heads=args['fusion_heads'], expansion=args['expansion'], attn_drop=args['attn_drop'], proj_drop=args['proj_drop'], proj_type=args['proj_type'])
net.load_state_dict(torch.load(model_path))
net.cuda()
net.eval()
activation = {}
def get_activation(name):
    def hook(model, input, output):
        output,_ = output
        activation[name] = output.detach()
    return hook
handle=net.down4.register_forward_hook(get_activation("down4"))

img_name_list=[]
img_slice_list=[]
latent_code_list=[]

for i in tqdm.tqdm(glob.glob("/research/cbim/datasets/biobank-dcm/ShortAxisHeartImages_20209/processing/ukbb/test_split/img/*.nii.gz")):
    try:
        img=sitk.ReadImage(i)
    except:
        print(i)
        continue
    img_np=sitk.GetArrayFromImage(img)
    img_slice_list=[t for t in range(len(img_np)) if img_np[t].max()!=0] #get non-zero slices
    assert len(img_slice_list)>1
    def preprocessing(img_np):
        max98=np.percentile(img_np,98.0)
        img_clipped=np.clip(img_np,0,max98)
        img_normalized=img_clipped/max98
        # print(img_np.shape)
        img_normalized=pad_crop(img_normalized,(img_np.shape[0],256,256))
        return img_normalized
    img_np_processed=preprocessing(img_np[img_slice_list])
    # img_np.shape
    img_input=torch.from_numpy(img_np_processed).unsqueeze(1)
    output=net(img_input.float().cuda())
    arr=activation["down4"].cpu().numpy()
    # breakpoint()
    #Save into files
    #save_name
    for j in range(img_np_processed.shape[0]):
        save_name=i.replace("/research/cbim/datasets/biobank-dcm/ShortAxisHeartImages_20209/processing/ukbb/test_split/img",save_to).replace(".nii.gz","_"+str(img_slice_list[j])+".npy")
        np.save(open(save_name,"wb"),arr[j])

# a={"img": img_name_list,"slice": img_slice_list, "code": latent_code_list}
# pickle.dump(a, open('computed_trainset_latent_code.p', 'wb'))