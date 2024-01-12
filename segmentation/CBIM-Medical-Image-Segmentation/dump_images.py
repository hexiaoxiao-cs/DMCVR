import argparse
import logging
import os
import random
import socket
import sys
import datetime
import glob
from SimpleITK.extra import GetArrayFromImage
import numpy as np
import psutil
# import setproctitle
import torch
# import wandb
import pandas as pd
import tqdm
from torchinfo import summary
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "/research/cbim/vast/xh172/FedCV/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "/research/cbim/vast/xh172/FedCV/FedML")))
# from data_preprocessing.sensetime.utils import *

# add the FedML root directory to the python path
# from model.segmentation.Unet_3D import Unet_3D



# from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
# from FedML.fedml_api.distributed.fedseg.FedSegAPI import FedML_init, FedML_FedSeg_distributed
# from FedML.fedml_api.distributed.fedseg.utils import count_parameters

#from data_preprocessing.coco.segmentation.data_loader.py import load_partition_data_distributed_coco_segmentation, load_partition_data_coco_segmentation
# from data_preprocessing.pascal_voc_augmented.data_loader import load_partition_data_distributed_pascal_voc, \
#     load_partition_data_pascal_voc
# from data_preprocessing.cityscapes.data_loader import load_partition_data_distributed_cityscapes, \
#     load_partition_data_cityscapes
# from data_preprocessing.sensetime.data_loader import load_partition_data_sensetime, load_partition_data_sensetime_semi
# from data_preprocessing.OAI.data_loader import partition_data_semi_with_did_test
# # from model.segmentation.deeplabV3_plus import DeepLabV3_plus
# from model.segmentation.unet import UNet
# from model.segmentation.Vnet_3D import VNet
# from data_preprocessing.sensetime.utils import *
# #create data_loader for validation data
# # system
import os, sys, time, copy, shutil

# plasma
# import sitktools

# numpy, scipy, scikit-learn
import numpy as np
from numpy import random
# import cPickle as pickle
import pickle
import gzip
import SimpleITK as sitk
from PIL import Image
# scipy
import scipy
# from scipy.misc import imsave, imread
import scipy.io as sio
import scipy.ndimage as ndimage
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
# skimage
from skimage.morphology import label


## create dir
def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)
    return


## get each file by postfix_file
def eachFile_postfix(dir_path, postfix_file):
    pathDir = os.listdir(dir_path)
    prefix_list = []
    icount = 0
    for file_one in pathDir:
        prefix_ind = file_one.find(postfix_file)
        if prefix_ind < 0:
            continue
        prefix_list.insert(icount, file_one[:prefix_ind])
        icount += 1
    return prefix_list


## get each file by postfix_file to a file
def eachFile_postfix_toFile(dir_path, postfix_file, output_file):
    prefix_list = eachFile_postfix(dir_path, postfix_file)
    with open(output_file, 'wb') as fp:
        pickle.dump(prefix_list, fp)
    return


## create file list from desc_file
def create_fileList(desc_file):
    try:
        with open(desc_file, 'r') as f:
            file_list = []
            for line in f:
                file_list.append(line.strip('\n'))
            f.close()
    except:
        raise NameError("cannot open " + desc_file)
    #
    return file_list


"""
## dump using pickle
def dopickle(my_buff, fname, mode='wb'):
    if fname[-3:] == '.gz':
        output = gzip.open(fname, mode)
    else:
        output = open(fname, mode)
    pickle.dump(my_buff, output, protocol=pickle.HIGHEST_PROTOCOL)
    output.close()
    return

## load using pickle
def unpickle(fname, mode='rb'):
    if fname[-3:] == '.gz':
        fo = gzip.open(fname, mode)
    else:
        fo = open(fname, mode)
    my_dict = pickle.load(fo)
    fo.close()
    return my_dict
"""


## dicomSeriesReader. from: https://itk.org/SimpleITKDoxygen/html/examples.html
def dicomSeriesReader(dicoms_folder, output_file=None):
    # init
    print("Reading Dicom directory:", dicoms_folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicoms_folder)
    reader.SetFileNames(dicom_names)
    # read
    image_sitk = reader.Execute()
    # image size
    size = image_sitk.GetSize()
    print("Image size:", size[0], size[1], size[2])
    # write
    if output_file != None:
        print("Writing image:", output_file)
        sitk.WriteImage(image_sitk, output_file)
    return image_sitk


## get buffer from a sitk image
def getBufferFromItkImage(image_sitk):
    buffer_ndarray = sitk.GetArrayFromImage(image_sitk)
    return buffer_ndarray


## get sitk image from an array
def getItkImageFromBuffer(buffer_ndarray):
    print(buffer_ndarray.shape)
    image_sitk = sitk.GetImageFromArray(buffer_ndarray, isVector=False)
    return image_sitk


## save 3d buffer to a certain model
def save3DBuff_toStd(fname, buffer_ndarray):
    image_sitk = getItkImageFromBuffer(buffer_ndarray)
    sitk.WriteImage(image_sitk, fname)


## load 3d buffer from a std 3d file
def load3DBuff_fromStd(fname):
    image_sitk = sitk.ReadImage(fname)
    buffer_ndarray = getBufferFromItkImage(image_sitk)
    return buffer_ndarray


## using sitk to resample image, change buffer size and spacing. from: http://itk-users.7.n7.nabble.com/SimpleITK-Downsampling-results-in-blank-image-td34923.html
def resamplingImage(image_sitk, fScale=1.0):
    fScale = np.float32(fScale)

    ## get org info
    buff_sz1, sp1, origin = image_sitk.GetSize(), image_sitk.GetSpacing(), image_sitk.GetOrigin()
    direction = image_sitk.GetDirection()
    # change buff size, spacing
    buff_sz2, sp2 = [np.uint32(np.round(n / fScale)) for n in buff_sz1], [n * fScale for n in sp1]
    # resampled info
    print("Orig Size ", buff_sz1, "\nNew Size ", buff_sz2)
    print("Orig Sp ", sp1, "\nNew Sp ", sp2)
    print(origin)

    ## resample
    fScale = np.float(fScale)
    t = sitk.Transform(3, sitk.sitkScale)
    t.SetParameters((1.0, 1.0, 1.0))
    resampled_image_sitk = sitk.Resample(image_sitk, buff_sz2, t, sitk.sitkLinear, origin, sp2, direction, 0.0,
                                         sitk.sitkFloat32)
    print("New Image size:", resampled_image_sitk.GetSize())

    ## using sitkFloat32 as final format
    '''
    ## itk image buffer type
    temp_buff = getBufferFromItkImage(image_sitk)
    np_type = temp_buff.dtype
    del temp_buff
    ## sitk cast
    if np_type == 'uint8':
        resampled_image_sitk = sitk.Cast(sitk.RescaleIntensity(resampled_image_sitk), sitk.sitkLabelUInt8)
        print "Cast type: uint8"
    elif np_type == 'uint16':
        resampled_image_sitk = sitk.Cast(sitk.RescaleIntensity(resampled_image_sitk), sitk.sitkLabelUInt16)
        print "Cast type: uint16"
    elif np_type == 'int32':
        resampled_image_sitk = sitk.Cast(sitk.RescaleIntensity(resampled_image_sitk), sitk.sitkInt32)
        print "Cast type: int32"
    elif np_type == 'float32':
        resampled_image_sitk = sitk.Cast(sitk.RescaleIntensity(resampled_image_sitk), sitk.sitkFloat32)
        print "Cast type: float32"
    '''
    return resampled_image_sitk


## original link: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/03_Image_Details.html
def resamplingImage_Spacing(image_sitk, scale_spacing=1.0, sampling_way=sitk.sitkNearestNeighbor,
                            sampling_pixel_t=sitk.sitkUInt16):
    fScale = np.float32(scale_spacing)
    ## get org info
    buff_sz1, sp1, origin = image_sitk.GetSize(), image_sitk.GetSpacing(), image_sitk.GetOrigin()
    direction = image_sitk.GetDirection()
    # change buff size, spacing
    buff_sz2, sp2 = [np.int32(np.round(n / fScale)) for n in buff_sz1], [n * fScale for n in sp1]
    # resampled info
    print("Orig Size ", buff_sz1, "\nNew Size ", buff_sz2)
    print("Orig Sp ", sp1, "\nNew Sp ", sp2)
    print(origin)
    ## resample
    t = sitk.Transform(3, sitk.sitkScale)
    t.SetParameters((1.0, 1.0, 1.0))
    resampled_image_sitk = sitk.Resample(image_sitk, buff_sz2, t, sampling_way, origin, sp2, direction, 0.0,
                                         sampling_pixel_t)
    print("New Image size:", resampled_image_sitk.GetSize())
    return resampled_image_sitk


## original link: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/03_Image_Details.html
def resamplingImage_Sp2(image_sitk, sp2=[1.0, 1.0, 1.0], sampling_way=sitk.sitkNearestNeighbor,
                        sampling_pixel_t=sitk.sitkUInt16):
    ## get org info
    buff_sz1, sp1, origin = image_sitk.GetSize(), image_sitk.GetSpacing(), image_sitk.GetOrigin()
    direction = image_sitk.GetDirection()
    # rate
    fScales = np.zeros(len(sp1), np.float32)
    for i in range(len(sp1)):
        fScales[i] = np.float32(sp2[i]) / np.float32(sp1[i])
    # change buff size
    buff_sz2 = list()
    for i in range(len(buff_sz1)):
        buff_sz2.append(int(np.round(buff_sz1[i] / fScales[i])))
    # resampled info
    print("Orig Size ", buff_sz1, "\nNew Size ", buff_sz2)
    print("Orig Sp ", sp1, "\nNew Sp ", sp2)
    print(origin)
    ## resample
    t = sitk.Transform(3, sitk.sitkScale)
    t.SetParameters((1.0, 1.0, 1.0))
    resampled_image_sitk = sitk.Resample(image_sitk, buff_sz2, t, sampling_way, origin, sp2, direction, 0.0,
                                         sampling_pixel_t)
    print("New Image size:", resampled_image_sitk.GetSize())
    return resampled_image_sitk


## bias correction on a 3d file.
def mriN4BiasCorrection(input_file, output_file, mask_file=None, shrinkFactor=1, numOfIters=10, numOfFilltingLev=4):
    # read
    input_sitk = sitk.ReadImage(input_file)
    if mask_file != None:
        mask_sitk = sitk.ReadImage(mask_file)
    else:
        mask_sitk = None
    # correction
    output_sitk = mriN4BiasCorrection_sitkImage(input_sitk, mask_sitk, shrinkFactor, numOfIters, numOfFilltingLev)
    sitk.WriteImage(output_sitk, output_file)
    return


## bias correction on a sitk image data. from: https://itk.org/SimpleITKDoxygen/html/Python_2N4BiasFieldCorrection_8py-example.html
def mriN4BiasCorrection_sitkImage(input_sitk, mask_sitk=None, shrinkFactor=1, numOfIters=100, numOfFilltingLev=0):
    # get mask
    if mask_sitk == None:
        mask_sitk = sitk.OtsuThreshold(input_sitk, 0, 1, 200)
        # shrink, using 1
    input_sitk = sitk.Shrink(input_sitk, [int(shrinkFactor)] * input_sitk.GetDimension())
    mask_sitk = sitk.Shrink(mask_sitk, [int(shrinkFactor)] * mask_sitk.GetDimension())
    # cast to float32
    input_sitk = sitk.Cast(input_sitk, sitk.sitkFloat32)

    ## correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # set max num of iterations
    numberFilltingLevels = 4
    if numOfFilltingLev > 0:
        numberFilltingLevels = numOfFilltingLev
    corrector.SetMaximumNumberOfIterations([int(numOfIters)] * numberFilltingLevels)
    # do
    output_sitk = corrector.Execute(input_sitk, mask_sitk)

    return output_sitk


## load mat
def load_mat(matFile, variable_name):
    if not os.path.isfile(matFile):
        raise IOError()
    else:
        print("load mat...")
        mat_contents = sio.loadmat(matFile)
    return mat_contents[variable_name]


## mat to bin
def mat_toBin(matFile, variable_name, binFile):
    if not os.path.isfile(matFile):
        raise IOError()
    else:
        print("load mat...")
        mat_contents = sio.loadmat(matFile)
        temp_buff = mat_contents[variable_name]
        temp_buff = np.array(temp_buff).astype(np.float32)
        print(binFile)
        print("bin shape: ", temp_buff.shape)
        temp_buff.tofile(binFile)
    return


## 0-1 linear normalization
def normalization_0_1(allData_4d):
    # one_im = allData_4d.flatten()
    allData_4d = np.asarray(allData_4d).astype(np.float32)
    dMax = np.max(allData_4d)
    dMin = np.min(allData_4d)
    if dMax - dMin > 0.001:
        allData_4d = (allData_4d - dMin) / (dMax - dMin)
    return allData_4d


## 0-255 linear normalization
def normalization_0_255(allData_4d):
    # one_im = allData_4d.flatten()
    dMax = np.amax(allData_4d)
    dMin = np.amin(allData_4d)
    if dMax - dMin > 0.001:
        allData_4d = (allData_4d - dMin) / (dMax - dMin) * 255.0
    return allData_4d


def specialMhdNorm(data):
    allData_4d = data
    # min-max adjust linear normalization
    dMax_new = 1400.0
    dMin_new = 500.0
    allData_4d[allData_4d < dMin_new] = 0
    allData_4d[allData_4d >= dMax_new] = dMax_new
    one_im = allData_4d.flatten()
    dMax = np.amax(one_im)
    dMin = np.amin(one_im)
    allData_4d = (allData_4d - dMin) / (dMax - dMin) * (dMax_new - dMin_new) + dMin_new
    # zero normalization
    xmean = allData_4d.mean()
    xstd = allData_4d.std()
    if np.abs(xstd < 0.0001):
        allData_4d = (allData_4d - xmean) / xstd
    # 0-1 linear normalization
    one_im = allData_4d.flatten()
    dMax = np.amax(one_im)
    dMin = np.amin(one_im)
    if dMax - dMin > 0.001:
        allData_4d = (allData_4d - dMin) / (dMax - dMin)
    return allData_4d
#
#
# ## display tool
# def display_Axial_Coronal(X_4d, y_4d, iAxial_pos, iCoronal_pos, dis_prefix, out_folder):
#     # Axial
#     im = X_4d[iAxial_pos, 0, :, :]
#     haha = dis_prefix + "Axial_x.png"
#     imsave(os.path.join(out_folder, haha), im)
#     im = y_4d[iAxial_pos, 0, :, :]
#     haha = dis_prefix + "Axial_y.png"
#     imsave(os.path.join(out_folder, haha), im)
#     # Coronal
#     im = X_4d[:, 0, iCoronal_pos, :]
#     haha = dis_prefix + "Coronal_x.png"
#     imsave(os.path.join(out_folder, haha), im)
#     im = y_4d[:, 0, iCoronal_pos, :]
#     haha = dis_prefix + "Coronal_y.png"
#     imsave(os.path.join(out_folder, haha), im)
#     return
#
#
# ## display tool, three directions, three slices
# def display_Axial_Coronal_Sagittal(X_3d, y_3d, iAxial_pos, iCoronal_pos, iSagittal_pos, dis_prefix, out_folder):
#     # Axial
#     im = X_3d[iAxial_pos, :, :]
#     haha = dis_prefix + "Axial_x.png"
#     imsave(os.path.join(out_folder, haha), im)
#     im = y_3d[iAxial_pos, :, :]
#     haha = dis_prefix + "Axial_y.png"
#     imsave(os.path.join(out_folder, haha), im)
#     # Coronal
#     im = X_3d[:, iCoronal_pos, :]
#     haha = dis_prefix + "Coronal_x.png"
#     imsave(os.path.join(out_folder, haha), im)
#     im = y_3d[:, iCoronal_pos, :]
#     haha = dis_prefix + "Coronal_y.png"
#     imsave(os.path.join(out_folder, haha), im)
#     # Sagittal
#     im = X_3d[:, :, iSagittal_pos]
#     haha = dis_prefix + "Sagittal_x.png"
#     imsave(os.path.join(out_folder, haha), im)
#     im = y_3d[:, :, iSagittal_pos]
#     haha = dis_prefix + "Sagittal_y.png"
#     imsave(os.path.join(out_folder, haha), im)
#     return
#

## gray buffer to rgb buffer
def grayBuff_to_rgbBuff(gray_img_buff, lb_map_1=None, map_1_color_list=[255, 0, 0], lb_map_2=None,
                        map_2_color_list=[0, 255, 0], lb_map_3=None, map_3_color_list=[0, 0, 255]):
    ih, iw = gray_img_buff.shape
    # rgb buffer
    imgx1 = gray_img_buff.copy()
    imgx2 = gray_img_buff.copy()
    imgx3 = gray_img_buff.copy()
    # main channel
    if not (lb_map_1 is None):
        imgx1[lb_map_1 > 0] = map_1_color_list[0]
        imgx2[lb_map_1 > 0] = map_1_color_list[1]
        imgx3[lb_map_1 > 0] = map_1_color_list[2]
    # mid channel
    if not (lb_map_2 is None):
        imgx1[lb_map_2 > 0] = map_2_color_list[0]
        imgx2[lb_map_2 > 0] = map_2_color_list[1]
        imgx3[lb_map_2 > 0] = map_2_color_list[2]
    # minor channel
    if not (lb_map_3 is None):
        imgx1[lb_map_3 > 0] = map_3_color_list[0]
        imgx2[lb_map_3 > 0] = map_3_color_list[1]
        imgx3[lb_map_3 > 0] = map_3_color_list[2]
    #
    rgbx = np.zeros([ih, iw, 3], np.uint8)
    rgbx[:, :, 0] = imgx1
    rgbx[:, :, 1] = imgx2
    rgbx[:, :, 2] = imgx3
    return rgbx

#
# ## display tool, three directions, three slices, dump contour
# def display_oaiBone_Axial_Coronal_Sagittal_withContours(X_3d, y_3d, dis_prefix, out_folder, contour_label=1,
#                                                         dump_contour_color=0, dump_pos="adaptive"):
#     ## re-normalization to 0-255
#     d_max = np.amax(X_3d)
#     d_min = np.amin(X_3d)
#     X_3d = (X_3d - d_min) / (d_max - d_min) * 255.0
#     ## check pos
#     iz, iy, ix = X_3d.shape
#     if dump_pos == "center":
#         iAxial_pos = np.around(iy / 2.0)
#         iSagittal_pos = np.around(iz / 2.0)
#         iCoronal_pos = np.around(ix / 2.0)
#     elif dump_pos == "adaptive":
#         inds = np.where(y_3d == contour_label)
#         iAxial_pos = np.around(np.mean(inds[1]))
#         iSagittal_pos = np.around(np.mean(inds[0]))
#         iCoronal_pos = np.around(np.mean(inds[2]))
#     else:
#         iAxial_pos = 0.0
#         iSagittal_pos = 0.0
#         iCoronal_pos = 0.0
#     iAxial_pos = np.uint32(iAxial_pos)
#     iSagittal_pos = np.uint32(iSagittal_pos)
#     iCoronal_pos = np.uint32(iCoronal_pos)
#     ## Axial
#     im = X_3d[:, iAxial_pos, :]
#     im = np.rot90(im, k=3)
#     lb = y_3d[:, iAxial_pos, :]
#     lb = np.rot90(lb, k=3)
#     probaBx = np.uint8(lb > 0)
#     Cx = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     if dump_contour_color == 0:
#         rgbx = grayBuff_to_rgbBuff(im, lb_map_1=Cx)
#     elif dump_contour_color == 1:
#         rgbx = grayBuff_to_rgbBuff(im, lb_map_2=Cx)
#     elif dump_contour_color == 2:
#         rgbx = grayBuff_to_rgbBuff(im, lb_map_3=Cx)
#     haha = dis_prefix + "_Axial.png"
#     imsave(os.path.join(out_folder, haha), rgbx)
#     ## Sagittal
#     im = X_3d[iSagittal_pos, :, :]
#     lb = y_3d[iSagittal_pos, :, :]
#     probaBx = np.uint8(lb > 0)
#     Cx = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     if dump_contour_color == 0:
#         rgbx = grayBuff_to_rgbBuff(im, lb_map_1=Cx)
#     elif dump_contour_color == 1:
#         rgbx = grayBuff_to_rgbBuff(im, lb_map_2=Cx)
#     elif dump_contour_color == 2:
#         rgbx = grayBuff_to_rgbBuff(im, lb_map_3=Cx)
#     haha = dis_prefix + "_Sagittal.png"
#     imsave(os.path.join(out_folder, haha), rgbx)
#     ## Coronal
#     im = X_3d[:, :, iCoronal_pos]
#     im = np.rot90(im, k=3)
#     lb = y_3d[:, :, iCoronal_pos]
#     lb = np.rot90(lb, k=3)
#     probaBx = np.uint8(lb > 0)
#     Cx = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     if dump_contour_color == 0:
#         rgbx = grayBuff_to_rgbBuff(im, lb_map_1=Cx)
#     elif dump_contour_color == 1:
#         rgbx = grayBuff_to_rgbBuff(im, lb_map_2=Cx)
#     elif dump_contour_color == 2:
#         rgbx = grayBuff_to_rgbBuff(im, lb_map_3=Cx)
#     haha = dis_prefix + "_Coronal.png"
#     imsave(os.path.join(out_folder, haha), rgbx)
#     return
#
#
# ## display tool, three directions, three slices, dump contour, multi-label
# def display_oaiBone_Axial_Coronal_Sagittal_withMultiContours(X_3d, y_3d, dis_prefix, out_folder, dump_pos="adaptive"):
#     ## re-normalization to 0-255
#     d_max = np.amax(X_3d)
#     d_min = np.amin(X_3d)
#     X_3d = (X_3d - d_min) / (d_max - d_min) * 255.0
#     ## check pos
#     iz, iy, ix = X_3d.shape
#     if dump_pos == "center":
#         iAxial_pos = np.around(iy / 2.0)
#         iSagittal_pos = np.around(iz / 2.0)
#         iCoronal_pos = np.around(ix / 2.0)
#     elif dump_pos == "adaptive":
#         inds = np.where(y_3d > 0)
#         iAxial_pos = np.around(np.mean(inds[1]))
#         iSagittal_pos = np.around(np.mean(inds[0]))
#         iCoronal_pos = np.around(np.mean(inds[2]))
#     else:
#         iAxial_pos = 0.0
#         iSagittal_pos = 0.0
#         iCoronal_pos = 0.0
#     iAxial_pos = np.uint32(iAxial_pos)
#     iSagittal_pos = np.uint32(iSagittal_pos)
#     iCoronal_pos = np.uint32(iCoronal_pos)
#     ## Axial
#     im = X_3d[:, iAxial_pos, :]
#     im = np.rot90(im, k=3)
#     lb = y_3d[:, iAxial_pos, :]
#     lb = np.rot90(lb, k=3)
#     probaBx = np.uint8(lb == 1)
#     Cx_r = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     probaBx = np.uint8(lb == 2)
#     Cx_g = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     probaBx = np.uint8(lb == 3)
#     Cx_b = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     rgbx = grayBuff_to_rgbBuff(im, lb_map_1=Cx_r, lb_map_2=Cx_g, lb_map_3=Cx_b)
#     haha = dis_prefix + "_Axial.png"
#     imsave(os.path.join(out_folder, haha), rgbx)
#     ## Sagittal
#     im = X_3d[iSagittal_pos, :, :]
#     lb = y_3d[iSagittal_pos, :, :]
#     probaBx = np.uint8(lb == 1)
#     Cx_r = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     probaBx = np.uint8(lb == 2)
#     Cx_g = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     probaBx = np.uint8(lb == 3)
#     Cx_b = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     rgbx = grayBuff_to_rgbBuff(im, lb_map_1=Cx_r, lb_map_2=Cx_g, lb_map_3=Cx_b)
#     haha = dis_prefix + "_Sagittal.png"
#     imsave(os.path.join(out_folder, haha), rgbx)
#     ## Coronal
#     im = X_3d[:, :, iCoronal_pos]
#     im = np.rot90(im, k=3)
#     lb = y_3d[:, :, iCoronal_pos]
#     lb = np.rot90(lb, k=3)
#     probaBx = np.uint8(lb == 1)
#     Cx_r = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     probaBx = np.uint8(lb == 2)
#     Cx_g = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     probaBx = np.uint8(lb == 3)
#     Cx_b = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     rgbx = grayBuff_to_rgbBuff(im, lb_map_1=Cx_r, lb_map_2=Cx_g, lb_map_3=Cx_b)
#     haha = dis_prefix + "_Coronal.png"
#     imsave(os.path.join(out_folder, haha), rgbx)
#     return
#
#
# ## display tool, three directions, three slices, dump seg and gt contours (binary segmentation)
# def display_oaiBone_Axial_Coronal_Sagittal_withSegGTContours(X_3d, y_3d, gt_3d, dis_prefix, out_folder, contour_label=1,
#                                                              dump_pos="adaptive"):
#     ## re-normalization to 0-255
#     d_max = np.amax(X_3d)
#     d_min = np.amin(X_3d)
#     X_3d = (X_3d - d_min) / (d_max - d_min) * 255.0
#     ## check pos
#     iz, iy, ix = X_3d.shape
#     if dump_pos == "center":
#         iAxial_pos = np.around(iy / 2.0)
#         iSagittal_pos = np.around(iz / 2.0)
#         iCoronal_pos = np.around(ix / 2.0)
#     elif dump_pos == "adaptive":
#         inds = np.where(y_3d == contour_label)
#         iAxial_pos = np.around(np.mean(inds[1]))
#         iSagittal_pos = np.around(np.mean(inds[0]))
#         iCoronal_pos = np.around(np.mean(inds[2]))
#     else:
#         iAxial_pos = 0.0
#         iSagittal_pos = 0.0
#         iCoronal_pos = 0.0
#     iAxial_pos = np.uint32(iAxial_pos)
#     iSagittal_pos = np.uint32(iSagittal_pos)
#     iCoronal_pos = np.uint32(iCoronal_pos)
#     ## Axial
#     im = X_3d[:, iAxial_pos, :]
#     im = np.rot90(im, k=3)
#     lb = y_3d[:, iAxial_pos, :]
#     lb = np.rot90(lb, k=3)
#     probaBx = np.uint8(lb > 0)
#     Cx = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     gt = gt_3d[:, iAxial_pos, :]
#     gt = np.rot90(gt, k=3)
#     probaBx_gt = np.uint8(gt > 0)
#     Cx_gt = probaBx_gt - ndimage.binary_erosion(probaBx_gt)  # scipy.ndimage
#     rgbx = grayBuff_to_rgbBuff(im, lb_map_1=Cx_gt, map_1_color_list=[0, 255, 0], lb_map_2=Cx,
#                                map_2_color_list=[255, 0, 0])
#     haha = dis_prefix + "Axial.png"
#     imsave(os.path.join(out_folder, haha), rgbx)
#     ## Sagittal
#     im = X_3d[iSagittal_pos, :, :]
#     lb = y_3d[iSagittal_pos, :, :]
#     probaBx = np.uint8(lb > 0)
#     Cx = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     gt = gt_3d[iSagittal_pos, :, :]
#     probaBx_gt = np.uint8(gt > 0)
#     Cx_gt = probaBx_gt - ndimage.binary_erosion(probaBx_gt)  # scipy.ndimage
#     rgbx = grayBuff_to_rgbBuff(im, lb_map_1=Cx_gt, map_1_color_list=[0, 255, 0], lb_map_2=Cx,
#                                map_2_color_list=[255, 0, 0])
#     haha = dis_prefix + "Sagittal.png"
#     imsave(os.path.join(out_folder, haha), rgbx)
#     ## Coronal
#     im = X_3d[:, :, iCoronal_pos]
#     im = np.rot90(im, k=3)
#     lb = y_3d[:, :, iCoronal_pos]
#     lb = np.rot90(lb, k=3)
#     probaBx = np.uint8(lb > 0)
#     Cx = probaBx - ndimage.binary_erosion(probaBx)  # scipy.ndimage
#     gt = gt_3d[:, :, iCoronal_pos]
#     gt = np.rot90(gt, k=3)
#     probaBx_gt = np.uint8(gt > 0)
#     Cx_gt = probaBx_gt - ndimage.binary_erosion(probaBx_gt)  # scipy.ndimage
#     rgbx = grayBuff_to_rgbBuff(im, lb_map_1=Cx_gt, map_1_color_list=[0, 255, 0], lb_map_2=Cx,
#                                map_2_color_list=[255, 0, 0])
#     haha = dis_prefix + "Coronal.png"
#     imsave(os.path.join(out_folder, haha), rgbx)
#     return
#

## display all slices with GT
def dumpContours_withGT(image_Buffer4D, proba_Buffer4D, gt_Buffer4D,
                        proba_thres=0.5, subject_num=-1, axi=2,
                        result_folder='C:\Temp'):
    imgB = image_Buffer4D
    if imgB.dtype == 'float64':
        if np.max(imgB) <= 1.0 and np.min(imgB) >= 0.0:
            imgB *= 255
        imgB = np.asarray(imgB).astype(np.uint8)

    probaB = proba_Buffer4D
    gtB = gt_Buffer4D
    # print "imgB : ", imgB.shape
    # print "probaB : ", probaB.shape
    # print "gtB : ", gtB.shape
    iS, iW, iH = imgB.shape
    ## liang's visualization
    imageSave = result_folder
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    for mz in range(iS):
        slice_pos = mz
        #########
        if axi == 0:
            imgx = np.array(Image.fromarray(imgB[::-1, :, mz]))
            probaBx = probaB[::-1, :, mz]
            gtBx = gtB[::-1, :, mz]
        elif axi == 1:
            imgx = np.array(Image.fromarray(imgB[::-1, mz, :]))
            probaBx = probaB[::-1, mz, :]
            gtBx = gtB[::-1, mz, :]
        elif axi == 2:
            imgx = np.array(Image.fromarray(imgB[mz, :, :]))
            probaBx = probaB[mz, :, :]
            gtBx = gtB[mz, :, :]
        # print "imgx: ", imgx.shape
        # print "probaBx: ", probaBx.shape
        # print "gtBx: ", gtBx.shape
        #########
        Cx1=np.bitwise_xor(gtBx,scipy.ndimage.binary_erosion(gtBx))
        #Cx1 = gtBx - scipy.ndimage.binary_erosion(gtBx)
        tmp0 = probaBx > proba_thres
        Cx2=np.bitwise_xor(tmp0,scipy.ndimage.binary_erosion(tmp0))
        #Cx2 = tmp0 - scipy.ndimage.binary_erosion(tmp0)
        
        rgbX = np.zeros([imgx.shape[0], imgx.shape[1], 3], np.uint8)
        # print "Cx1: ", Cx1.shape
        # print "Cx2: ", Cx2.shape
        #
        color=[(246, 16, 103),(94, 35, 157),(193, 174, 124),(0, 240, 181),(255, 237, 102),(255, 181, 99)]
        imgx2 = imgx.copy()
        imgx3 = imgx.copy()
        imgx4 = imgx.copy()
        imgx2[Cx1 > 0] = 0
        imgx3[Cx1 > 0] = 255
        imgx4[Cx1 > 0] = 0
        imgx2[Cx2 > 0] = 255
        imgx3[Cx2 > 0] = 0
        imgx4[Cx2 > 0] = 0
        rgbx = np.zeros([imgx2.shape[0], imgx2.shape[1], 3], np.uint8)
        rgbx[:, :, 0] = imgx2
        rgbx[:, :, 1] = imgx3
        rgbx[:, :, 2] = imgx4
        tmp1 = Image.fromarray(rgbx)
        if axi == 0:
            tmp1.save(imageSave + r'/%s_%.4d_z.bmp' % (subject_num, slice_pos))
            Image.fromarray(imgx).convert(mode="RGB").save(imageSave + r'/%s_%.4d_zImage.bmp' % (subject_num, slice_pos))
            Image.fromarray(probaBx).convert(mode="RGB").save(
                imageSave + r'/%s_%.4d_zR.bmp' % (subject_num, slice_pos))
        elif axi == 1:
            tmp1.save(imageSave + r'/%s_%.4d_y.bmp' % (subject_num, slice_pos))
            Image.fromarray(imgx).save(imageSave + r'/%s_%.4d_yImage.bmp' % (subject_num, slice_pos))
            Image.fromarray(probaBx, cmin=0, cmax=1).save(
                imageSave + r'/%s_%.4d_yR.bmp' % (subject_num, slice_pos))
        else:
            tmp1.save(imageSave + r'/%s_%.4d_x.bmp' % (subject_num, slice_pos))
            Image.fromarray(imgx).convert(mode="RGB").save(imageSave + r'/%s_%.4d_xImage.bmp' % (subject_num, slice_pos))
            Image.fromarray(probaBx).convert(mode="RGB").save(
                imageSave + r'/%s_%.4d_xR.bmp' % (subject_num, slice_pos))
    return

def dumpimages(image_Buffer4D,
                        proba_thres=0.5, subject_num=-1, axi=2,
                        result_folder='C:\Temp'):
    imgB = image_Buffer4D
    if imgB.dtype == 'float64':
        if np.max(imgB) <= 1.0 and np.min(imgB) >= 0.0:
            imgB *= 255
        imgB = np.asarray(imgB).astype(np.uint8)

    # print "imgB : ", imgB.shape
    # print "probaB : ", probaB.shape
    # print "gtB : ", gtB.shape
    iS, iW, iH = imgB.shape
    ## liang's visualization
    imageSave = result_folder
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    for mz in range(iS):
        slice_pos = mz
        #########
        if axi == 0:
            imgx = np.array(Image.fromarray(imgB[::-1, :, mz]))
        elif axi == 1:
            imgx = np.array(Image.fromarray(imgB[::-1, mz, :]))
        elif axi == 2:
            imgx = np.array(Image.fromarray(imgB[mz, :, :]))
        # print(imgx.shape)
        # print(imgx.min())
        # print(imgx.max())
        # breakpoint()
        imgx=imgx-0.5 #Ours and Orig
        imgx=imgx*2   #Ours and Orig  
        imgx=np.clip(imgx,0,1) #Ours and Orig
        # imgx_98=np.percentile(imgx,98) #CQ
        # imgx=np.clip(imgx,0,imgx_98) #CQ
        if imgx.max()==0:
            continue
        # imgx=imgx/imgx_98 #CQ
        # imgx=imgx/imgx.max()
        imgx=imgx*255
        # print(imgx.min())
        # print(imgx.max())
        rgbx = np.zeros([imgx.shape[0], imgx.shape[1], 3], np.uint8)
        rgbx[:, :, 0] = imgx.copy()
        rgbx[:, :, 1] = imgx.copy()
        rgbx[:, :, 2] = imgx.copy()
        tmp1 = Image.fromarray(rgbx.astype(np.uint8))
        if axi == 0:
            tmp1.save(imageSave + r'/%s_%.4d_z.bmp' % (subject_num, slice_pos))
            # Image.fromarray(imgx).convert(mode="RGB").save(imageSave + r'/%s_%.4d_zImage.bmp' % (subject_num, slice_pos))
            # Image.fromarray(probaBx).convert(mode="RGB").save(
            #     imageSave + r'/%s_%.4d_zR.bmp' % (subject_num, slice_pos))
        elif axi == 1:
            tmp1.save(imageSave + r'/%s_%.4d_y.bmp' % (subject_num, slice_pos))
            # Image.fromarray(imgx).save(imageSave + r'/%s_%.4d_yImage.bmp' % (subject_num, slice_pos))
            # Image.fromarray(probaBx, cmin=0, cmax=1).save(
            #     imageSave + r'/%s_%.4d_yR.bmp' % (subject_num, slice_pos))
        else:
            tmp1.save(imageSave + r'/%s_%.4d_x.bmp' % (subject_num, slice_pos))
            # Image.fromarray(imgx).convert(mode="RGB").save(imageSave + r'/%s_%.4d_xImage.bmp' % (subject_num, slice_pos))
            # Image.fromarray(probaBx).convert(mode="RGB").save(
            #     imageSave + r'/%s_%.4d_xR.bmp' % (subject_num, slice_pos))
    return


## display all slices without GT
def dumpContours_withoutGT(image_Buffer3D, proba_Buffer3D,
                           proba_thres=0.5, subject_num=-1, axi=0,
                           result_folder='C:\Temp'):
    imgB = image_Buffer3D
    probaB = proba_Buffer3D
    # print "imgB : ", imgB.shape
    # print "probaB : ", probaB.shape
    iS, iW, iH = imgB.shape
    ## liang's visualization
    imageSave = result_folder
    for mz in range(iS):
        slice_pos = mz
        #########
        if axi == 0:
            imgx = np.array(scipy.misc.toimage(imgB[::-1, :, mz]))
            probaBx = probaB[::-1, :, mz]
        elif axi == 1:
            imgx = np.array(scipy.misc.toimage(imgB[::-1, mz, :]))
            probaBx = probaB[::-1, mz, :]
        elif axi == 2:
            imgx = np.array(scipy.misc.toimage(imgB[mz, :, :]))
            probaBx = probaB[mz, :, :]
        # print "imgx: ", imgx.shape
        # print "probaBx: ", probaBx.shape
        #########
        tmp0 = probaBx > proba_thres
        Cx2 = tmp0 - scipy.ndimage.binary_erosion(tmp0)
        rgbX = np.zeros([imgx.shape[0], imgx.shape[1], 3], np.uint8)
        # print "Cx2: ", Cx2.shape
        #
        imgx2 = imgx.copy()
        imgx3 = imgx.copy()
        imgx4 = imgx.copy()
        imgx2[Cx2 > 0] = 255
        imgx4[Cx2 > 0] = 0
        imgx3[Cx2 > 0] = 0
        rgbx = np.zeros([imgx2.shape[0], imgx2.shape[1], 3], np.uint8)
        rgbx[:, :, 0] = imgx2
        rgbx[:, :, 1] = imgx3
        rgbx[:, :, 2] = imgx4
        tmp1 = Image.fromarray(rgbx)
        if axi == 0:
            tmp1.save(imageSave + r'\%.4d_%.4d_z.bmp' % (subject_num, slice_pos))
            scipy.misc.toimage(imgx).save(imageSave + r'\%.4d_%.4d_zImage.bmp' % (subject_num, slice_pos))
            scipy.misc.toimage(probaBx, cmin=0, cmax=1).save(
                imageSave + r'\%.4d_%.4d_zR.bmp' % (subject_num, slice_pos))
        elif axi == 1:
            tmp1.save(imageSave + r'\%.4d_%.4d_y.bmp' % (subject_num, slice_pos))
            scipy.misc.toimage(imgx).save(imageSave + r'\%.4d_%.4d_yImage.bmp' % (subject_num, slice_pos))
            scipy.misc.toimage(probaBx, cmin=0, cmax=1).save(
                imageSave + r'\%.4d_%.4d_yR.bmp' % (subject_num, slice_pos))
        else:
            tmp1.save(imageSave + r'\%.4d_%.4d_x.bmp' % (subject_num, slice_pos))
            scipy.misc.toimage(imgx).save(imageSave + r'\%.4d_%.4d_xImage.bmp' % (subject_num, slice_pos))
            scipy.misc.toimage(probaBx, cmin=0, cmax=1).save(
                imageSave + r'\%.4d_%.4d_xR.bmp' % (subject_num, slice_pos))
    return


## display the mid slices of a volume with GT
def dumpContours_groups_withGT(image_Buffer4D, proba_Buffer4D, gt_Buffer4D,
                               proba_thres=0.5, subject_num=-1, slice_pos=64, axi=0,
                               result_folder='C:\Temp'):
    imgB = image_Buffer4D
    if imgB.dtype == 'float64':
        if np.max(imgB) <= 1.0 and np.min(imgB) >= 0.0:
            imgB *= 255
        imgB = np.asarray(imgB).astype(np.uint8)

    probaB = proba_Buffer4D
    gtB = gt_Buffer4D
    # print "imgB : ", imgB.shape
    # print "probaB : ", probaB.shape
    # print "gtB : ", gtB.shape
    ## liang's visualization
    mz = slice_pos
    imageSave = result_folder
    # print mz
    #########
    if axi == 0:
        imgx = imgB[::-1, :, mz]
        probaBx = probaB[::-1, :, mz]
        gtBx = gtB[::-1, :, mz]
    elif axi == 1:
        imgx = imgB[::-1, mz, :]
        probaBx = probaB[::-1, mz, :]
        gtBx = gtB[::-1, mz, :]
    elif axi == 2:
        imgx = imgB[mz, :, :]
        probaBx = probaB[mz, :, :]
        gtBx = gtB[mz, :, :]
    # print "imgx: ", imgx.shape
    # print "probaBx: ", probaBx.shape
    # print "gtBx: ", gtBx.shape
    #########
    Cx1 = gtBx - scipy.ndimage.binary_erosion(gtBx)
    tmp0 = probaBx > proba_thres
    Cx2 = tmp0 - scipy.ndimage.binary_erosion(tmp0)
    rgbX = np.zeros([imgx.shape[0], imgx.shape[1], 3], np.uint8)
    # print "Cx2: ", Cx2.shape
    #
    imgx2 = imgx.copy()
    imgx3 = imgx.copy()
    imgx4 = imgx.copy()
    imgx2[Cx1 > 0] = 0
    imgx4[Cx1 > 0] = 0
    imgx3[Cx1 > 0] = 255
    imgx2[Cx2 > 0] = 255
    imgx4[Cx2 > 0] = 0
    imgx3[Cx2 > 0] = 0
    rgbx = np.zeros([imgx2.shape[0], imgx2.shape[1], 3], np.uint8)
    rgbx[:, :, 0] = imgx2
    rgbx[:, :, 1] = imgx3
    rgbx[:, :, 2] = imgx4
    tmp1 = Image.fromarray(rgbx)
    if axi == 0:
        tmp1.save(imageSave + r'\%.4d_%.4d_z.bmp' % (subject_num, slice_pos))
        # scipy.misc.toimage(imgx).save(imageSave+r'\%.4d_%.4d_zImage.bmp'%(subject_num, slice_pos))
        scipy.misc.toimage(probaBx, cmin=0, cmax=1).save(imageSave + r'\%.4d_%.4d_zR.bmp' % (subject_num, slice_pos))
    elif axi == 1:
        tmp1.save(imageSave + r'\%.4d_%.4d_y.bmp' % (subject_num, slice_pos))
        # scipy.misc.toimage(imgx).save(imageSave+r'\%.4d_%.4d_yImage.bmp'%(subject_num, slice_pos))
        scipy.misc.toimage(probaBx, cmin=0, cmax=1).save(imageSave + r'\%.4d_%.4d_yR.bmp' % (subject_num, slice_pos))
    else:
        tmp1.save(imageSave + r'\%.4d_%.4d_x.bmp' % (subject_num, slice_pos))
        # scipy.misc.toimage(imgx).save(imageSave+r'\%.4d_%.4d_xImage.bmp'%(subject_num, slice_pos))
        scipy.misc.toimage(probaBx, cmin=0, cmax=1).save(imageSave + r'\%.4d_%.4d_xR.bmp' % (subject_num, slice_pos))
    return


## display the mid slices of a volume without GT
def dumpContours_groups_withoutGT(image_Buffer4D, proba_Buffer4D,
                                  proba_thres=0.5, subject_num=-1, slice_pos=64, axi=0,
                                  result_folder='C:\Temp'):
    imgB = image_Buffer4D
    probaB = proba_Buffer4D
    # print "imgB : ", imgB.shape
    # print "probaB : ", probaB.shape
    ## liang's visualization
    mz = slice_pos
    imageSave = result_folder
    print(mz)
    #########
    if axi == 0:
        imgx = imgB[::-1, :, mz]
        probaBx = probaB[::-1, :, mz]
    elif axi == 1:
        imgx = imgB[::-1, mz, :]
        probaBx = probaB[::-1, mz, :]
    elif axi == 2:
        imgx = imgB[mz, :, :]
        probaBx = probaB[mz, :, :]
    # print "imgx: ", imgx.shape
    # print "probaBx: ", probaBx.shape
    #########
    tmp0 = probaBx > proba_thres
    Cx2 = tmp0 - scipy.ndimage.binary_erosion(tmp0)
    rgbX = np.zeros([imgx.shape[0], imgx.shape[1], 3], np.uint8)
    # print "Cx2: ", Cx2.shape
    #
    imgx2 = imgx.copy()
    imgx3 = imgx.copy()
    imgx4 = imgx.copy()
    imgx2[Cx2 > 0] = 255
    imgx4[Cx2 > 0] = 0
    imgx3[Cx2 > 0] = 0
    rgbx = np.zeros([imgx2.shape[0], imgx2.shape[1], 3], np.uint8)
    rgbx[:, :, 0] = imgx2
    rgbx[:, :, 1] = imgx3
    rgbx[:, :, 2] = imgx4
    tmp1 = Image.fromarray(rgbx)
    if axi == 0:
        tmp1.save(imageSave + r'\%.4d_%.4d_z.bmp' % (subject_num, slice_pos))
        scipy.misc.toimage(imgx).save(imageSave + r'\%.4d_%.4d_zImage.bmp' % (subject_num, slice_pos))
        scipy.misc.toimage(probaBx, cmin=0, cmax=1).save(imageSave + r'\%.4d_%.4d_zR.bmp' % (subject_num, slice_pos))
    elif axi == 1:
        tmp1.save(imageSave + r'\%.4d_%.4d_y.bmp' % (subject_num, slice_pos))
        scipy.misc.toimage(imgx).save(imageSave + r'\%.4d_%.4d_yImage.bmp' % (subject_num, slice_pos))
        scipy.misc.toimage(probaBx, cmin=0, cmax=1).save(imageSave + r'\%.4d_%.4d_yR.bmp' % (subject_num, slice_pos))
    else:
        tmp1.save(imageSave + r'\%.4d_%.4d_x.bmp' % (subject_num, slice_pos))
        scipy.misc.toimage(imgx).save(imageSave + r'\%.4d_%.4d_xImage.bmp' % (subject_num, slice_pos))
        scipy.misc.toimage(probaBx, cmin=0, cmax=1).save(imageSave + r'\%.4d_%.4d_xR.bmp' % (subject_num, slice_pos))
    return


# read data by simple itk
# def getItkImage(fimg):
#     image_sitk = sitktools.loadImage(fimg)
#     return image_sitk


# convert buffer as a sitk image
def convBufferToItkImage(buffer, orgImage):
    newImage = sitk.GetImageFromArray(buffer)
    newImage.SetOrigin(orgImage.GetOrigin())
    newImage.SetDirection(orgImage.GetDirection())
    newImage.SetSpacing(orgImage.GetSpacing())
    return newImage


# # read itk data from plasma
# def getItkImageFromPlasma(plasmaImage, outFolder=r'c:\temp'):
#     plasmatools.SaveBoxToMHD(plasmaImage, os.path.join(outFolder, 'image'))
#     image_sitk = sitktools.loadImage(os.path.join(outFolder, 'image.mhd'))
#     return image_sitk
#
#
# # read itk data buffer from plasma
# def getImageBufferFromPlasma(plasmaImage, outFolder=r'c:\temp'):
#     plasmatools.SaveBoxToMHD(plasmaImage, os.path.join(outFolder, 'image'))
#     image = sitktools.loadImage(os.path.join(outFolder, 'image.mhd'))
#     return sitk.GetArrayFromImage(image)
#
#
# # load itk data to plasma
# def loadItkImageToPlasma(itkImage, imTag='m'):
#     sitktools.viewImage(itkImage, imTag)
#     return plasmatools.allids()[-1]
#

# dice: single label
def dice(seg, gt, val_lb=1):
    """
    ## init
    """
    if seg.shape != gt.shape:
        raise ValueError("Shape mismatch: seg and gt must have the same shape.")
    #
    if val_lb < 0:
        seg = seg > 0
        gt = gt > 0
    else:
        seg = seg == val_lb
        gt = gt == val_lb
    # Compute Dice coefficient
    intersection = np.logical_and(seg, gt)
    return 2. * intersection.sum() / (seg.sum() + gt.sum())


# voe: single label
def voe(seg, gt, val_lb=1):
    """
    ## init
    """
    if seg.shape != gt.shape:
        raise ValueError("Shape mismatch: seg and gt must have the same shape.")
    #
    if val_lb < 0:
        seg = seg > 0
        gt = gt > 0
    else:
        seg = seg == val_lb
        gt = gt == val_lb
    # Compute voe coefficient
    intersection = np.logical_and(seg, gt)
    union = np.logical_or(seg, gt)
    return 100.0 * (1.0 - np.float32(intersection.sum()) / np.float32(union.sum()))


# vd: single label
def vd(seg, gt, val_lb=1):
    """
    ## init
    """
    if seg.shape != gt.shape:
        raise ValueError("Shape mismatch: seg and gt must have the same shape.")
    #
    if val_lb < 0:
        seg = seg > 0
        gt = gt > 0
    else:
        seg = seg == val_lb
        gt = gt == val_lb
    # Compute vd coefficient
    gt = np.int8(gt)
    wori = np.int8(seg - gt)
    return 100.0 * (wori.sum() / gt.sum())


"""
## medpy for dists
"""


## basic: surface errors/distances
def surface_distances(result, reference, voxelspacing=None, connectivity=1, iterations=1, ret_all=False):
    """
    # The distances between the surface voxel of binary objects in result and their
    # nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_3d(result.astype(np.bool))
    reference = np.atleast_3d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')
    # extract only 1-pixel border line of objects
    result_border = np.logical_xor(result, binary_erosion(result, structure=footprint, iterations=iterations))
    reference_border = np.logical_xor(reference, binary_erosion(reference, structure=footprint, iterations=iterations))
    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    ##
    if ret_all:
        return sds, dt, result_border, reference_border
    else:
        return sds


## ausdorff Distance.
def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    ## Hausdorff Distance.
    # Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    # images. It is defined as the maximum surface distance between the objects.
    ## Parameters
    # ----------
    # result : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # reference : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # voxelspacing : float or sequence of floats, optional
    #     The voxelspacing in a distance unit i.e. spacing of elements
    #     along each dimension. If a sequence, must be of length equal to
    #     the input rank; if a single number, this is used for all axes. If
    #     not specified, a grid spacing of unity is implied.
    # connectivity : int
    #     The neighbourhood/connectivity considered when determining the surface
    #     of the binary objects. This value is passed to
    #     `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
    #     Note that the connectivity influences the result in the case of the Hausdorff distance.
    # Returns
    # -------
    # hd : float
    #     The symmetric Hausdorff Distance between the object(s) in ```result``` and the
    #     object(s) in ```reference```. The distance unit is the same as for the spacing of
    #     elements along each dimension, which is usually given in mm.
    #
    # See also
    # --------
    # :func:`assd`
    # :func:`asd`
    # Notes
    # -----
    # This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


## Average surface distance metric.
def asd(result, reference, voxelspacing=None, connectivity=1):
    """
    # Average surface distance metric.
    # Computes the average surface distance (ASD) between the binary objects in two images.
    # Parameters
    # ----------
    # result : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # reference : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # voxelspacing : float or sequence of floats, optional
    #     The voxelspacing in a distance unit i.e. spacing of elements
    #     along each dimension. If a sequence, must be of length equal to
    #     the input rank; if a single number, this is used for all axes. If
    #     not specified, a grid spacing of unity is implied.
    # connectivity : int
    #     The neighbourhood/connectivity considered when determining the surface
    #     of the binary objects. This value is passed to
    #     `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
    #     The decision on the connectivity is important, as it can influence the results
    #     strongly. If in doubt, leave it as it is.
    # Returns
    # -------
    # asd : float
    #     The average surface distance between the object(s) in ``result`` and the
    #     object(s) in ``reference``. The distance unit is the same as for the spacing
    #     of elements along each dimension, which is usually given in mm.
    # See also
    # --------
    # :func:`assd`
    # :func:`hd`
    # Notes
    # -----
    # This is not a real metric, as it is directed. See `assd` for a real metric of this.
    # The method is implemented making use of distance images and simple binary morphology
    # to achieve high computational speed.
    # Examples
    # --------
    # The `connectivity` determines what pixels/voxels are considered the surface of a
    # binary object. Take the following binary image showing a cross
    #
    # from scipy.ndimage.morphology import generate_binary_structure
    # cross = generate_binary_structure(2, 1)
    # array([[0, 1, 0],
    #        [1, 1, 1],
    #        [0, 1, 0]])
    # With `connectivity` set to `1` a 4-neighbourhood is considered when determining the
    # object surface, resulting in the surface
    # .. code-block:: python
    #
    #     array([[0, 1, 0],
    #            [1, 0, 1],
    #            [0, 1, 0]])
    # Changing `connectivity` to `2`, a 8-neighbourhood is considered and we get:
    # .. code-block:: python
    #
    #     array([[0, 1, 0],
    #            [1, 1, 1],
    #            [0, 1, 0]])
    #
    # , as a diagonal connection does no longer qualifies as valid object surface.
    #
    # This influences the  results `asd` returns. Imagine we want to compute the surface
    # distance of our cross to a cube-like object:
    #
    # cube = generate_binary_structure(2, 1)
    # array([[1, 1, 1],
    #        [1, 1, 1],
    #        [1, 1, 1]])
    #
    # , which surface is, independent of the `connectivity` value set, always
    #
    # .. code-block:: python
    #
    #     array([[1, 1, 1],
    #            [1, 0, 1],
    #            [1, 1, 1]])
    #
    # Using a `connectivity` of `1` we get
    #
    # asd(cross, cube, connectivity=1)
    # 0.0
    #
    # while a value of `2` returns us
    #
    # asd(cross, cube, connectivity=2)
    # 0.20000000000000001
    #
    # due to the center of the cross being considered surface as well.
    """
    sds = surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd


## Average symmetric surface distance.
def assd(result, reference, voxelspacing=None, connectivity=1):
    """
    # Average symmetric surface distance.
    #
    # Computes the average symmetric surface distance (ASD) between the binary objects in
    # two images.
    #
    # Parameters
    # ----------
    # result : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # reference : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # voxelspacing : float or sequence of floats, optional
    #     The voxelspacing in a distance unit i.e. spacing of elements
    #     along each dimension. If a sequence, must be of length equal to
    #     the input rank; if a single number, this is used for all axes. If
    #     not specified, a grid spacing of unity is implied.
    # connectivity : int
    #     The neighbourhood/connectivity considered when determining the surface
    #     of the binary objects. This value is passed to
    #     `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
    #     The decision on the connectivity is important, as it can influence the results
    #     strongly. If in doubt, leave it as it is.
    #
    # Returns
    # -------
    # assd : float
    #     The average symmetric surface distance between the object(s) in ``result`` and the
    #     object(s) in ``reference``. The distance unit is the same as for the spacing of
    #     elements along each dimension, which is usually given in mm.
    #
    # See also
    # --------
    # :func:`asd`
    # :func:`hd`
    #
    # Notes
    # -----
    # This is a real metric, obtained by calling and averaging
    #
    # >>> asd(result, reference)
    #
    # and
    #
    # >>> asd(reference, result)
    #
    # The binary images can therefore be supplied in any order.
    """
    assd = np.mean(
        (asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)))
    return assd


"""
## connected comp
"""


## max connected
def max_connected_comp(tmp_buff, lb_num=-1, neighbors=4):
    if lb_num == -1:
        binary_buff = np.uint8(tmp_buff > 0)
    else:
        binary_buff = np.uint8(tmp_buff == lb_num)
    # connected_comp
    connected_group, connected_num = label(binary_buff, neighbors=neighbors, return_num=True)
    comp_sum = []
    for i in range(connected_num + 1):
        if i == 0:
            comp_sum.insert(i, 0)
            continue
        comp_sum.insert(i, np.sum(connected_group == i))
    max_comp_ind = np.argmax(comp_sum)
    if lb_num == -1:
        max_comp = np.uint8(connected_group == max_comp_ind)
    else:
        max_comp = np.uint8(connected_group == max_comp_ind) * np.uint8(lb_num)
    #
    return max_comp


## max two connected
def max_two_connected_comp(tmp_buff, lb_num=-1, neighbors=4):
    if lb_num == -1:
        binary_buff = np.uint8(tmp_buff > 0)
    else:
        binary_buff = np.uint8(tmp_buff == lb_num)
    # connected_comp
    connected_group, connected_num = label(binary_buff, neighbors=neighbors, return_num=True)
    comp_sum = []
    for i in range(connected_num + 1):
        if i == 0:
            comp_sum.insert(i, 0)
            continue
        comp_sum.insert(i, np.sum(connected_group == i))
    small_to_big_inds = np.argsort(comp_sum)
    first_max = small_to_big_inds[-1]
    second_max = small_to_big_inds[-2]
    if lb_num == -1:
        max_two_comp = np.uint8(connected_group == first_max) + np.uint8(connected_group == second_max)
    else:
        max_two_comp = (np.uint8(connected_group == first_max) + np.uint8(connected_group == second_max)) * np.uint8(
            lb_num)
    #
    return max_two_comp


"""
## half_expect_size to real_size: boundary loss
"""


def half_expect_to_real_boundaryLoss(expect_size_sagittal_axial_coronal, buff_l, kval=3):
    ## real size
    if kval < 0:
        a, b, c = np.where(buff_l > 0)
    else:
        a, b, c = np.where(buff_l == kval)
    min_real_size_sagittal_axial_coronal = [np.min(a), np.min(b), np.min(c)]
    max_real_size_sagittal_axial_coronal = [np.max(a), np.max(b), np.max(c)]
    center_real_size_sagittal_axial_coronal = getTargetCenter_fromBuffL_by2(buff_l, kval=kval)
    delta_center2min = np.asarray(center_real_size_sagittal_axial_coronal) - np.asarray(
        min_real_size_sagittal_axial_coronal)
    delta_max2center = np.asarray(max_real_size_sagittal_axial_coronal) - np.asarray(
        center_real_size_sagittal_axial_coronal)
    ## expected size
    expect_size_S, expect_size_A, expect_size_C = expect_size_sagittal_axial_coronal
    half_z_1 = expect_size_S // 2
    half_z_2 = expect_size_S - half_z_1
    half_y_1 = expect_size_A // 2
    half_y_2 = expect_size_A - half_y_1
    half_x_1 = expect_size_C // 2
    half_x_2 = expect_size_C - half_x_1
    half_expect_1 = np.asarray([half_z_1, half_y_1, half_x_1])
    half_expect_2 = np.asarray([half_z_2, half_y_2, half_x_2])
    ## find
    low_bound = half_expect_1 - delta_center2min
    real_largerThan_expect_low = False
    if np.sum(np.int8(low_bound > 0)) < len(delta_center2min):
        real_largerThan_expect_low = True
    high_bound = half_expect_2 - delta_max2center
    real_largerThan_expect_max = False
    if np.sum(np.int8(high_bound > 0)) < len(delta_max2center):
        real_largerThan_expect_max = True
    #
    return real_largerThan_expect_low, low_bound, real_largerThan_expect_max, high_bound


"""
## get center
"""


def getTargetCenter_fromBuffL_by2(buff_l, kval=3):
    """
    ## inputs: buff_l from an itk image of label, kval is value of target
    ## outputs: target center in voxel coordinates
    """
    if kval < 0:
        a, b, c = np.where(buff_l > 0)
    else:
        a, b, c = np.where(buff_l == kval)
    # center in voxel coordinates, z, x, and y are in voxel coordinates
    # len_axial = [1] --> b
    # len_coronal = [0] --> c
    # len_sagittal = [2] --> a
    center_sagittal_axial_coronal = [(np.max(a) + np.min(a)) / 2.0, (np.max(b) + np.min(b)) / 2.0,
                                     (np.max(c) + np.min(c)) / 2.0]
    return center_sagittal_axial_coronal


"""
## get crop buff
"""


def getCropBuff_byCenter(buff_zyx, center_sagittal_axial_coronal, expect_size_sagittal_axial_coronal, file_name=None,
                         record_file=None):
    ## init
    expect_size_S, expect_size_A, expect_size_C = expect_size_sagittal_axial_coronal
    iS, iH, iW = buff_zyx.shape
    cropbuff_sagittal_axial_coronal = np.zeros((expect_size_S, expect_size_A, expect_size_C), dtype=buff_zyx.dtype)
    c_z_sagittal, c_y_axial, c_x_coronal = center_sagittal_axial_coronal
    ## real crop range
    half_z_1 = expect_size_S // 2
    half_z_2 = expect_size_S - half_z_1
    half_y_1 = expect_size_A // 2
    half_y_2 = expect_size_A - half_y_1
    half_x_1 = expect_size_C // 2
    half_x_2 = expect_size_C - half_x_1
    # print("buff_zyx shape: ", buff_zyx.shape)
    # print("expect_size_sagittal_axial_coronal: ", expect_size_sagittal_axial_coronal)
    # print("center_sagittal_axial_coronal: ", center_sagittal_axial_coronal)
    # print("half_z_1=%d, half_z_2=%d, half_y_1=%d, half_y_2=%d, half_x_1=%d, half_x_2=%d."%(half_z_1, half_z_2, half_y_1, half_y_2, half_x_1, half_x_2))
    ## z, sagittal
    low_z = c_z_sagittal - half_z_1 + 1
    if low_z < 0:
        print("%s has low sagittal!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has low sagittal!!!\n" % (file_name))
        local_l_z = -low_z
        low_z = 0
    else:
        local_l_z = 0
    up_z = c_z_sagittal + half_z_2 + 1
    if up_z >= iS:
        print("%s has up sagittal!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has up sagittal!!!\n" % (file_name))
        local_u_z = expect_size_S - (up_z - iS)  # (up_z - iS + 1)
        up_z = iS
    else:
        local_u_z = expect_size_S
    ## y, axial
    low_y = c_y_axial - half_y_1 + 1
    if low_y < 0:
        print("%s has low axial!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has low axial!!!\n" % (file_name))
        local_l_y = -low_y
        low_y = 0
    else:
        local_l_y = 0
    up_y = c_y_axial + half_y_2 + 1
    if up_y >= iH:
        print("%s has up axial!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has up axial!!!\n" % (file_name))
        local_u_y = expect_size_A - (up_y - iH)  # (up_y - iH + 1)
        up_y = iH
    else:
        local_u_y = expect_size_A
    ## x, coronal
    low_x = c_x_coronal - half_x_1 + 1
    if low_x < 0:
        print("%s has low coronal!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has low coronal!!!\n" % (file_name))
        local_l_x = -low_x
        low_x = 0
    else:
        local_l_x = 0
    up_x = c_x_coronal + half_x_2 + 1
    if up_x >= iW:
        print("%s has up coronal!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has up coronal!!!\n" % (file_name))
        local_u_x = expect_size_C - (up_x - iW)  # (up_x - iW + 1)
        up_x = iW
    else:
        local_u_x = expect_size_C
    ## temp buff
    # print("low_sagittal=%d, up_sagittal=%d, low_axial=%d, up_axial=%d, low_coronal=%d, up_coronal=%d."%
    #      (low_z, up_z, low_y, up_y, low_x, up_x))
    # print("local_l_sagittal=%d, local_u_sagittal=%d, local_l_axial=%d, local_u_axial=%d, local_l_coronal=%d, local_u_coronal=%d."%
    #      (local_l_z, local_u_z, local_l_y, local_u_y, local_l_x, local_u_x))
    cropbuff_sagittal_axial_coronal[local_l_z:local_u_z, local_l_y:local_u_y, local_l_x:local_u_x] = \
        buff_zyx[low_z:up_z, low_y:up_y, low_x:up_x]
    ##
    return cropbuff_sagittal_axial_coronal




def process(img_path):
    spacing_c_a_s = [0.3125, 0.3125, 3.3]
    basename=os.path.basename(img_path)
    bn=basename.split("_")
    subject_name=bn[0]+"_"+bn[5].split(".")[0]
    
    # print(subject_name)
    # truth_path=img_path.replace("img","lbl")
    # lbl_path=img_path.replace("img","pred")
    img_sitk=sitk.ReadImage(img_path)
    # pred_sitk=sitk.ReadImage(lbl_path)
    # truth_sitk=sitk.ReadImage(truth_path)
    img=sitk.GetArrayFromImage(img_sitk)
    # pred=sitk.GetArrayFromImage(pred_sitk)
    # truth=sitk.GetArrayFromImage(truth_sitk)
    # truth[truth == 1] = 0
    # truth[truth == 2] = 1
    # truth[truth == 2] = 0
    # truth[truth == 6] = 2
    # truth[truth == 3] = 0
    # truth[truth == 8] = 3
    # truth[truth > 3] = 0
    # pic_res = list()
    # resampled_img_itk, resampled_label_itk = itk_resample(img_sitk, truth_sitk, out_spacing=spacing_c_a_s,
    #                                                   interpolation=sitk.sitkLinear,
    #                                                   dtype=sitk.sitkFloat32)
    # resampled_label = sitk.GetArrayFromImage(resampled_label_itk)
    # sagital_num_slices=32
    # resampled_pred_itk = itk_resample_only_label(pred_sitk, out_spacing=spacing_c_a_s)
    # resampled_pred = sitk.GetArrayFromImage(resampled_pred_itk)
    # fixed_size_label = pad_crop(resampled_label, (sagital_num_slices, 512, 512), mode='constant', value=0)
    # fixed_size_pred = pad_crop(resampled_pred, (sagital_num_slices, 512, 512), mode='constant', value=0)
    # spacing_c_a_s = list(truth_sitk.GetSpacing())
    # prediction_truth = prediction > 0
    # prediction_1_truth = prediction_1 == 1
    # prediction_2_truth = prediction_2 == 2
    # truth_1 = truth == 1
    # truth_2 = truth == 2
    # truth_truth = truth > 0
    # voxelspacing = spacing_c_a_s.reverse()
    # pred=fixed_size_pred
    # truth=fixed_size_label
    # line=list()
    # line.append(subject_name)
    # line.append(dice(pred,truth,-1))
    # line.append(voe(pred,truth,-1))
    # line.append(vd(pred,truth,-1))
    # line.append(asd(pred,truth,voxelspacing=voxelspacing))
    # line.append(hd(pred, truth, voxelspacing=voxelspacing))
    # line.append(assd(pred, truth, voxelspacing=voxelspacing))
    # print(line)
    # pic_res = list()
    # pic_res.append(line)
    # for j in range(3):
    #     line=list()
    #     line.append(subject_name)
    #     # pred_temp=pred== j+1
    #     # truth_temp=truth==j+1
    #     line.append(dice(pred,truth,j+1))
    #     line.append(voe(pred, truth, j + 1))
    #     line.append(vd(pred, truth, j + 1))
    #     pred_temp=pred== j+1
    #     truth_temp=truth==j+1
    #     # print(pred_temp)
    #     # print(truth_temp)
    #     if (~pred_temp).all()==True:
    #         pic_res.append(line)
    #         print("Empty")
    #         continue
    #     line.append(asd(pred_temp, truth_temp,voxelspacing=voxelspacing))
    #     line.append(hd(pred_temp, truth_temp, voxelspacing=voxelspacing))
    #     line.append(assd(pred_temp, truth_temp, voxelspacing=voxelspacing))
    #     pic_res.append(line)
    #     print(line)
    #start visualization
    # img=sitk.GetArrayFromImage(img_sitk)
    # pred=sitk.GetArrayFromImage(pred_sitk)
    # truth=sitk.GetArrayFromImage(truth_sitk)
    #dumpContours_withGT(img,pred,truth,subject_num=subject_name,result_folder=os.path.join(save_dir,subject_name))
    dumpimages(img,subject_num=subject_name,result_folder=os.path.join(save_dir,subject_name))
    return None

#save_dir="./generated_ukbb_2nd_lbled_testing"
# save_dir="./generated_ukbb_cq"
# save_dir="./orig_images"
save_dir="./generated_ukbb_diffae"
save_dir="./generated_ukbb_2nd_lbled_testing_LAX"
save_dir="./generated_ukbb_2nd_lbled_testing_LAX_case_1"
from multiprocessing import Pool
spacing_c_a_s = [0.3125, 0.3125, 3.3]
if __name__=="__main__":
    scores=None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    process("/research/cbim/vast/xh172/diffusion/diffae/lin_slerp_slerp/img/1001904_20209_2_0_sa_0.nii.gz")
    # process("/filer/tmp1/xh172/generated_ukbb_2nd_to_LAX/T100/lin_slerp_slerp/img/1911473_20209_2_0_sa_0.nii.gz")
    # for i in tqdm.tqdm(glob.glob("/filer/tmp1/xh172/generated_ukbb/no_inter_test/T100/img/*.nii.gz")):
    #     if "segmented" in i:
    #         continue
    #     process(i)
    # for i in tqdm.tqdm(glob.glob("/filer/tmp1/xh172/generated_ukbb_2nd_lbled_testing/T100/img/*.nii.gz")):
    #     if "segmented" in i:
    #         continue
    #     process(i)
    # for i in tqdm.tqdm(glob.glob("/filer/tmp1/xh172/generated_cq_opt/img/*.nii.gz")):
    #     if "segmented" in i:
    #         continue
    #     process(i)
    # for i in tqdm.tqdm(glob.glob("/filer/tmp1/xh172/ukbb/test_extracted/img/*.nii.gz")):
    #     if "segmented" in i:
    #         continue
    #     if "1911473" in i:
    #         process(i)
    
    # with Pool(25) as p:
    #     scores=p.map(process,glob.glob("/filer/tmp1/xh172/generated_ukbb_2nd_lbled_testing/T100/img/*.nii.gz"))
    # df=pd.DataFrame(scores)
    # df.to_csv(os.path.join(save_dir,"summary.csv"))