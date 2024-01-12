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
import torch
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
# target_spacing.reverse() #This is because the italy dataset spacing is reversed
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

source_folder=glob.glob("/filer/tmp1/xh172/generated_ukbb/no_inter_test/T100/img/*.nii.gz")
#source_folder=glob.glob("/filer/tmp1/xh172/generated_ukbb/no_inter/img/*.nii.gz")
# target_folder="/filer/tmp1/xh172/Derivate1_spacing_regularized_segmented/"
os.environ['CUDA_VISIBLE_DEVICES']="0"

#Here we load the model
model_path="/research/cbim/vast/xh172/cardiac-segmentation-FDA/CBIM-Medical-Image-Segmentation/exp/biobank/test/fold_3_best.pth"
#Here we load the configuration file
config_path="/research/cbim/vast/xh172/cardiac-segmentation-FDA/CBIM-Medical-Image-Segmentation/config/biobank/medformer_2d.yaml"

#Here we get the gt path


# parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
# args = vars()
with open(config_path, 'r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
# args.update(config)
# for key, value in config.items():
#     setattr(args, key, value)
from model.dim2 import MedFormer
#net=MedFormer(args['in_chan'], args['classes'], args['base_chan'], map_size=args['map_size'], conv_block=args['conv_block'], conv_num=args['conv_num'], trans_num=args['trans_num'], num_heads=args['num_heads'], fusion_depth=args['fusion_depth'], fusion_dim=args['fusion_dim'], fusion_heads=args['fusion_heads'], expansion=args['expansion'], attn_drop=args['attn_drop'], proj_drop=args['proj_drop'], proj_type=args['proj_type'], norm=args['norm'], act=args['act'], kernel_size=args['kernel_size'], scale=args['down_scale'])

net=MedFormer(args['in_chan'], args['classes'], args['base_chan'], map_size=args['map_size'], conv_block=args['conv_block'], conv_num=args['conv_num'], trans_num=args['trans_num'], num_heads=args['num_heads'], fusion_depth=args['fusion_depth'], fusion_dim=args['fusion_dim'], fusion_heads=args['fusion_heads'], expansion=args['expansion'], attn_drop=args['attn_drop'], proj_drop=args['proj_drop'], proj_type=args['proj_type'])

net.load_state_dict(torch.load(model_path))
net.cuda()
net.eval()


for i in tqdm.tqdm(source_folder):
    output_name=i.replace(".nii.gz","_segmented.nii.gz")
    img=sitk.ReadImage(i)
    img_np=sitk.GetArrayFromImage(img)
    img_98=np.percentile(img_np,98)
    img_cropped_regularized=np.clip(img_np,0,img_98)
    # img_sitk=sitk.GetImageFromArray(img_cropped_regularized)
    img_tensor=torch.from_numpy(img_cropped_regularized).unsqueeze(1)
    inputs = img_tensor.float().cuda()
    pred_=net(inputs)
    pred=torch.argmax(torch.nn.functional.softmax(pred_,dim=1),dim=1)
    pred_itk = sitk.GetImageFromArray(pred.cpu())
    pred_itk.SetSpacing((1,1,1))
    sitk.WriteImage(pred_itk,output_name)
    # gt_itk=sitk.ReadImage(i.replace("img","lbl").replace("_sa","_label_sa"))
    


'''
for i in tqdm.tqdm(source_folder):
    list_file=glob.glob(os.path.join(i,"*"))
    fold_name=os.path.basename(i)
    curr_tgt_folder=os.path.join(target_folder,fold_name)
    if not os.path.exists(curr_tgt_folder):
        os.makedirs(curr_tgt_folder)
    for l in list_file:
        img=sitk.ReadImage(l)
        spacing = img.GetSpacing()
        print(spacing)
        print(img.GetSize())
        print(target_spacing)
        # img=ResampleXYZAxis(img,target_spacing)
        # re_img_xy = ResampleXYZAxis(img, space=(spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkBSpline)
        # re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)
        # re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
        # re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)
        # print(re_img_xyz.GetSpacing())
        img_np=sitk.GetArrayFromImage(img)
        print(img_np.shape)
        img_np=img_np.T
        img_cropped=pad_crop(img_np,crop_size)
        print("image_cropped",img_cropped.shape)
        img_98=np.percentile(img_cropped,98)
        img_cropped_regularized=np.clip(img_cropped,0,img_98)
        img_sitk=sitk.GetImageFromArray(img_cropped_regularized)
        img_cropped_regularized=img_cropped_regularized/img_98
        # img_cropped_regularized=img_cropped_regularized.astype(np.float32)
        img_cropped_regularized=img_cropped_regularized[np.newaxis,np.newaxis, :]
        img_tensor=torch.from_numpy(img_cropped_regularized)
        inputs = img_tensor.float().cuda()
        #predict
        pred_=net(inputs)
        pred=torch.argmax(torch.nn.functional.softmax(pred_,dim=1),dim=1)
        pred = pred[0].cpu().numpy()
        print(pred.shape)
        #write to file in corresponding place
        pred_itk = sitk.GetImageFromArray(pred)
        # pred_itk.SetOrigin(img.GetOrigin())
        pred_itk.SetSpacing((0.7422,0.7422,8.0))
        # pred_itk.SetDirection(img.GetDirection())
        # img_sitk.SetOrigin(img.GetOrigin())
        img_sitk.SetSpacing((0.7422,0.7422,8.0))
        # img_sitk.SetDirection(img.GetDirection())
        # print(pred_itk.GetSize())
        # print(img_sitk.GetSize())
        base_name=os.path.basename(l).replace(".nii.gz","") #get subject name
        sitk.WriteImage(img_sitk,os.path.join(curr_tgt_folder,base_name+"_img.nii.gz"))
        sitk.WriteImage(pred_itk,os.path.join(curr_tgt_folder,base_name+"_lbl.nii.gz"))
'''