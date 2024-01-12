import SimpleITK as sitk
import numpy as np  
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os
import pandas as pd
import glob
import tqdm
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms


#Config for E2E generated dataset testing
# save_path="/filer/tmp1/xh172/generated_ukbb_E2E_lbled_testing/T100/"
# path_to_gt="/filer/tmp1/xh172/generated_ukbb_E2E_lbled_testing/T100/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_ukbb_E2E_lbled_testing/T100/img"


#Config for cq generated dataset testing (not optimized):
# save_path="/filer/tmp1/xh172/generated_cq_no_opt/"
# path_to_gt="/filer/tmp1/xh172/generated_cq_no_opt/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_cq_no_opt/img"


#config for cq generated dataset testing (optimized):
save_path="/filer/tmp1/xh172/generated_cq_opt/"
path_to_gt="/filer/tmp1/xh172/generated_cq_opt/lbl"
path_to_pred="/filer/tmp1/xh172/generated_cq_opt/img"


#Config for med_256 generated dataset testing:
# save_path="/filer/tmp1/xh172/generated_ukbb/no_inter_test/T100/"
# path_to_gt="/filer/tmp1/xh172/generated_ukbb/no_inter_test/T100/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_ukbb/no_inter_test/T100/img"

#Config for 2nd generated dataset testing:
save_path="/filer/tmp1/xh172/generated_ukbb_2nd_lbled_testing/T100/"
path_to_gt="/filer/tmp1/xh172/generated_ukbb_2nd_lbled_testing/T100/lbl"
path_to_pred="/filer/tmp1/xh172/generated_ukbb_2nd_lbled_testing/T100/img"

#Config for our generated dataset#
# save_path="/filer/tmp1/xh172/generated_ukbb_2nd_lbled/T100/"
# path_to_gt="/filer/tmp1/xh172/generated_ukbb_2nd_lbled/T100/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_ukbb_2nd_lbled/T100/img"

#Config for diffae

# save_path="/filer/tmp1/xh172/generated_ukbb/no_inter/"
# path_to_gt="/filer/tmp1/xh172/generated_ukbb/no_inter/lbl"
# path_to_pred="/filer/tmp1/xh172/generated_ukbb/no_inter/img"

path_to_orig_img="/filer/tmp1/xh172/ukbb/test_extracted/img"
ret_list=[]


for i in glob.glob(os.path.join(path_to_pred,"*_segmented.nii.gz")):
    test_path=i.replace("_segmented","")
    bn=os.path.basename(test_path)
    ref_path=os.path.join(path_to_orig_img,bn)
    ref_img_sitk=sitk.ReadImage(ref_path)
    test_img_sitk=sitk.ReadImage(test_path)
    #Here we need to regularize ref image 
    ref_img=sitk.GetArrayFromImage(ref_img_sitk)
    # breakpoint()
    filtered_indexes=[i for i in range(len(ref_img)) if not ref_img[i].max()==0]
    filtered_ref=ref_img[filtered_indexes]
    max98 = np.percentile(filtered_ref, 98)
    filtered_ref = np.clip(filtered_ref, 0, max98)
    filtered_ref = filtered_ref / max98

    #here we regularize the other image
    test_img=sitk.GetArrayFromImage(test_img_sitk)
    test_img=test_img-0.5
    test_img=test_img*2
    test_img= np.clip(test_img,0,1)
    # max98 = np.percentile(test_img, 98)
    # test_img = np.clip(test_img, 0, max98)
    # test_img = test_img / max98
    filtered_indexes=[i for i in range(len(test_img)) if not test_img[i].max()==0]
    filtered_test=test_img[filtered_indexes]

    # print(filtered_test.shape,filtered_ref.shape)
    # breakpoint()
    # for i in range(len(filtered_ref)):
    #     #try to match histogram
    #     # matched = match_histograms(filtered_test[i], filtered_ref[i])
    #     matched=filtered_test[i]
    #     print(ssim(filtered_ref[i],matched,data_range=1.0),psnr(filtered_ref[i],matched,data_range=1.0))
    #here we get both images now we calculate the ssim and psnr for each slice
    ret_ssim=ssim(filtered_ref,filtered_test,data_range=1.0)
    ret_psnr=psnr(filtered_ref,filtered_test,data_range=1.0)
    ret_list.append(str(ret_ssim)+","+str(ret_psnr)+"\n")
    # psnr_list.append(ret_psnr)
    # print(ret_ssim,ret_psnr)

with open(os.path.join(save_path,"image_quality_metric.txt"),"w") as f:
    f.writelines(ret_list)


# error_path=[]
# def processing(error_path,p_path):
#     if "segmented_segmented" in p_path:
#         error_path.append(p_path)
#         return  None
#     #Now we have prediciton path, we need the to find gt path
#     gt_path=p_path.replace(path_to_pred,path_to_gt).replace("_segmented","").replace("_sa","_label_sa")
#     #print(gt_path)
#     #check if gt exists
#     if not os.path.exists(gt_path):
#         return None
#     pred_itk=sitk.ReadImage(p_path)
#     pred=sitk.GetArrayFromImage(pred_itk)
#     truth_itk=sitk.ReadImage(gt_path)
#     truth=sitk.GetArrayFromImage(truth_itk)
#     img_path=p_path.replace("_segmented","")
#     img_bn=os.path.basename(img_path)
#     # print(img_bn)
    
#     img_orig_path=os.path.join(path_to_orig_img,img_bn)
#     img_sitk=sitk.ReadImage(img_orig_path)
#     img=sitk.GetArrayFromImage(img_sitk)
    
#     filtered_indexes=[i for i in range(len(img)) if not img[i].max()==0]
#     # print(filtered_indexes)
#     truth=truth[filtered_indexes]
    
#     # print(truth.shape)
#     # print(pred.shape)
#     # breakpoint()
#     #Whether get the max connected component
#     region_num = [1,1, 1]
#     # pred = multi_find_largest_k_region(pred.astype(np.uint8), region_num)
#     #voxelspacing= list(truth_itk.GetSpacing()).reverse()
#     voxelspacing=list((1.8269230127334595, 1.8269230127334595, 10.0)).reverse()
#     try:
#         pic_res = list()
#         line = list()
#         line.append("0")
#         line.append(dice(pred, truth, -1))
#         line.append(voe(pred, truth, -1))
#         line.append(vd(pred, truth, -1))
#         line.append(asd(pred, truth, voxelspacing=voxelspacing))
#         line.append(hd(pred, truth, voxelspacing=voxelspacing))
#         line.append(assd(pred, truth, voxelspacing=voxelspacing))
#         # print(line)
#         pic_res.append(line)
#         for j in range(3):
#             line = list()
#             # pred_temp=pred== j+1
#             # truth_temp=truth==j+1
#             line.append(str(j+1))
#             line.append(dice(pred, truth, j + 1))
#             line.append(voe(pred, truth, j + 1))
#             line.append(vd(pred, truth, j + 1))
#             pred_temp = pred == j + 1
#             truth_temp = truth == j + 1
#             line.append(asd(pred_temp, truth_temp, voxelspacing=voxelspacing))
#             line.append(hd(pred_temp, truth_temp, voxelspacing=voxelspacing))
#             line.append(assd(pred_temp, truth_temp, voxelspacing=voxelspacing))
#             pic_res.append(line)
#             # print(line)
#     except:
#         error_path.append(p_path)
#         return None
#     return pic_res

# # @iex
# def main():
#     #pred:prediction
#     #truth:gt

#     # scores=pd.DataFrame()
#     scores_l=None
#     error_list=None
#     clean_list=[]
#     scores=pd.DataFrame()
#     # error_list=[]
#     with Pool(20) as p,Manager() as manager:
#         L = manager.list() 
#         # args=list(itertools.product([L],glob.glob(os.path.join(path_to_pred,"*_segmented.nii.gz"))))
#         # print(args)
#         scores_l=p.starmap(processing,zip(repeat(L), glob.glob(os.path.join(path_to_pred,"*_segmented.nii.gz"))))
#         for i in L:
#             print(i)
#         for i in scores_l:
#             if i is not None:
#                 scores=scores.append(i)

#     # scores=pd.DataFrame(clean_list)
#     # breakpoint()
#     # for i in scores_l:
#     #     if i is not None:
#     #         # for j in i:
#     #         scores=scores(i)
#     scores.to_csv(os.path.join(save_path,"summary.csv"))
