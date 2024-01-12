import SimpleITK as sitk
import os
import glob



path_to_orig_img="/filer/tmp1/xh172/ukbb/train_extracted/img"

save_path="/filer/tmp1/xh172/generated_ukbb_2nd_lbled/T100/lbl_clean"
path_to_gt="/filer/tmp1/xh172/generated_ukbb_2nd_lbled/T100/lbl"
path_to_pred="/filer/tmp1/xh172/generated_ukbb_2nd_lbled/T100/img"

if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in glob.glob(os.path.join(path_to_gt,"*.nii.gz")):
    bn=os.path.basename(i)
    bn_img=bn.replace("_label","")
    if not os.path.exists(os.path.join(path_to_orig_img,bn_img)):
        print(i)
        print(bn_img)
        print(os.path.join(path_to_orig_img,bn_img))
        continue
    #CHeck if file exists
    
    orig_img_sitk=sitk.ReadImage(os.path.join(path_to_orig_img,bn_img))
    orig_img=sitk.GetArrayFromImage(orig_img_sitk)
    lbl_sitk=sitk.ReadImage(i)
    lbl=sitk.GetArrayFromImage(lbl_sitk)
    filtered_indexes=[i for i in range(len(orig_img)) if not orig_img[i].max()==0]
    lbl_filtered=lbl[filtered_indexes]
    new_lbl_sitk=sitk.GetImageFromArray(lbl_filtered)
    new_lbl_sitk.SetSpacing((1,1,1))
    output_name=os.path.join(save_path,bn)
    sitk.WriteImage(new_lbl_sitk,output_name)