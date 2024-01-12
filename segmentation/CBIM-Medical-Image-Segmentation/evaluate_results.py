
# numpy, scipy, scikit-learn
import warnings
# with warnings.catch_warnings():
#     # filter sklearn\externals\joblib\parallel.py:268:
#     # DeprecationWarning: check_pickle is deprecated
#     warnings.simplefilter("ignore", category=DeprecationWarning)
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
import glob
import os
import pandas as pd
import tqdm
from ipdb import iex
from multiprocessing import Pool,Manager
from itertools import repeat

def binary_find_largest_k_region(arr, top_k=1):
    """
    binary find the largest k region(s) of numpy array
    :param arr:
    :param top_k:
    :return:
    """
    arr = label(arr, connectivity=1)
    labels, counts = np.unique(arr, return_counts=True)
    counts = counts[labels > 0]
    labels = labels[labels > 0]
    top_k_label = labels[np.argsort(counts)[-top_k:]]
    return np.isin(arr, top_k_label).astype(np.uint8)


def multi_find_largest_k_region(arr, top_ks=None):
    """
    multiple class find the largest k region(s) of numpy array
    :param arr:
    :param top_ks:
    :return:
    """
    labels = np.unique(arr)
    labels = np.sort(labels[labels > 0])
    if top_ks is None:
        top_ks = [1] * len(labels)
    else:
        # if len(top_ks) != len(labels), just return the origin image
        # return arr
        assert len(top_ks) == len(labels), 'got %d labels and %d top_k(s)' % (len(labels), len(top_ks))
    multi_largest_k = np.zeros_like(arr)
    for cls, k in zip(labels, top_ks):
        cls_mask = arr == cls
        multi_largest_k += binary_find_largest_k_region(cls_mask, k) * cls
    return multi_largest_k


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
    result = np.atleast_3d(result.astype(bool))
    reference = np.atleast_3d(reference.astype(bool))
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
#Config for Segmentation Performance
save_path="/filer/tmp1/xh172/test_extracted/"
path_to_gt="/filer/tmp1/xh172/test_extracted/lbl"
path_to_pred="/filer/tmp1/xh172/test_extracted/img"


#Config for E2E generated dataset testing
# save_path="/filer/tmp1/xh172/generated_ukbb_E2E_lbled_testing/T100/"
# path_to_gt="/filer/tmp1/xh172/generated_ukbb_E2E_lbled_testing/T100/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_ukbb_E2E_lbled_testing/T100/img"


#Config for cq generated dataset testing (not optimized):
# save_path="/filer/tmp1/xh172/generated_cq_no_opt/"
# path_to_gt="/filer/tmp1/xh172/generated_cq_no_opt/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_cq_no_opt/img"


#config for cq generated dataset testing (optimized):
# save_path="/filer/tmp1/xh172/generated_cq_opt/"
# path_to_gt="/filer/tmp1/xh172/generated_cq_opt/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_cq_opt/img"


#Config for med_256 generated dataset testing:
# save_path="/filer/tmp1/xh172/generated_ukbb/no_inter_test/T100/"
# path_to_gt="/filer/tmp1/xh172/generated_ukbb/no_inter_test/T100/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_ukbb/no_inter_test/T100/img"

#Config for 2nd generated dataset testing:
# save_path="/filer/tmp1/xh172/generated_ukbb_2nd_lbled_testing/T100/"
# path_to_gt="/filer/tmp1/xh172/generated_ukbb_2nd_lbled_testing/T100/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_ukbb_2nd_lbled_testing/T100/img"

#Config for our generated dataset#
# save_path="/filer/tmp1/xh172/generated_ukbb_2nd_lbled/T100/"
# path_to_gt="/filer/tmp1/xh172/generated_ukbb_2nd_lbled/T100/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_ukbb_2nd_lbled/T100/img"

#Config for diffae

# save_path="/filer/tmp1/xh172/generated_ukbb/no_inter/"
# path_to_gt="/filer/tmp1/xh172/generated_ukbb/no_inter/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_ukbb/no_inter/img"

#This does not need to be changed
# path_to_orig_img="/filer/tmp1/xh172/ukbb/train_extracted/img"
path_to_orig_img="/filer/tmp1/xh172/ukbb/test_extracted/img"
error_path=[]
def processing(error_path,p_path):
    if "segmented_segmented" in p_path:
        error_path.append(p_path)
        return  None
    #Now we have prediciton path, we need the to find gt path
    gt_path=p_path.replace(path_to_pred,path_to_gt).replace("_segmented","").replace("_sa","_label_sa")
    # print(gt_path)
    #print(gt_path)
    #check if gt exists
    # breakpoint()
    if not os.path.exists(gt_path):
        return None
    pred_itk=sitk.ReadImage(p_path)
    pred=sitk.GetArrayFromImage(pred_itk)
    truth_itk=sitk.ReadImage(gt_path)
    truth=sitk.GetArrayFromImage(truth_itk)
    img_path=p_path.replace("_segmented","")
    img_bn=os.path.basename(img_path)
    # print(img_bn)
    
    img_orig_path=os.path.join(path_to_orig_img,img_bn)
    img_sitk=sitk.ReadImage(img_orig_path)
    img=sitk.GetArrayFromImage(img_sitk)
    
    filtered_indexes=[i for i in range(len(img)) if not img[i].max()==0]
    # print(filtered_indexes)
    truth=truth[filtered_indexes]
    pred=pred[filtered_indexes]
    # print(truth.shape)
    # print(pred.shape)
    # breakpoint()
    #Whether get the max connected component
    region_num = [1,1, 1]
    # pred = multi_find_largest_k_region(pred.astype(np.uint8), region_num)
    #voxelspacing= list(truth_itk.GetSpacing()).reverse()
    voxelspacing=list((1.8269230127334595, 1.8269230127334595, 10.0)).reverse()
    # breakpoint()
    try:
        pic_res = list()
        line = list()
        line.append("0")
        line.append(dice(pred, truth, -1))
        line.append(voe(pred, truth, -1))
        line.append(vd(pred, truth, -1))
        line.append(asd(pred, truth, voxelspacing=voxelspacing))
        line.append(hd(pred, truth, voxelspacing=voxelspacing))
        line.append(assd(pred, truth, voxelspacing=voxelspacing))
        # print(line)
        pic_res.append(line)
        for j in range(3):
            line = list()
            # pred_temp=pred== j+1
            # truth_temp=truth==j+1
            line.append(str(j+1))
            line.append(dice(pred, truth, j + 1))
            line.append(voe(pred, truth, j + 1))
            line.append(vd(pred, truth, j + 1))
            pred_temp = pred == j + 1
            truth_temp = truth == j + 1
            line.append(asd(pred_temp, truth_temp, voxelspacing=voxelspacing))
            line.append(hd(pred_temp, truth_temp, voxelspacing=voxelspacing))
            line.append(assd(pred_temp, truth_temp, voxelspacing=voxelspacing))
            pic_res.append(line)
            # print(line)
    except:
        error_path.append(p_path)
        return None
    return pic_res

# @iex
def main():
    #pred:prediction
    #truth:gt

    # scores=pd.DataFrame()
    scores_l=None
    error_list=None
    clean_list=[]
    scores=pd.DataFrame()
    error_list=[]
    with Pool(20) as p,Manager() as manager:
        L = manager.list() 
        # args=list(itertools.product([L],glob.glob(os.path.join(path_to_pred,"*_segmented.nii.gz"))))
        # print(args)
        scores_l=p.starmap(processing,zip(repeat(L), glob.glob(os.path.join(path_to_pred,"*_segmented.nii.gz"))))
        for i in L:
            print(i)
        for i in scores_l:
            if i is not None:
                scores=scores.append(i)
    # for i in glob.glob(os.path.join(path_to_pred,"*_segmented.nii.gz")):
    #     processing(error_list,i)

    # scores=pd.DataFrame(clean_list)
    # breakpoint()
    # for i in scores_l:
    #     if i is not None:
    #         # for j in i:
    #         scores=scores(i)
    scores.to_csv(os.path.join(save_path,"summary.csv"))

if __name__=="__main__":
    main()