import SimpleITK as sitk
import glob
import os
import numpy as np
import tqdm 

base_path="/filer/tmp1/xh172/generated_ukbb_2nd_lbled_testing/T100/"

save_path="img_contrasted/"

if not os.path.exists(os.path.join(base_path,save_path)):
    os.makedirs(os.path.join(base_path,save_path))

for i in tqdm.tqdm(glob.glob(os.path.join(base_path,"img","*.nii.gz"))):
    if "segmented" in i:
        continue
    img_sitk=sitk.ReadImage(i)
    fname=os.path.basename(i)
    img_np=sitk.GetArrayFromImage(img_sitk)
    img_np=img_np-0.5
    img_np=img_np*2.0
    img_np=np.clip(img_np,0,1)
    new_sitk=sitk.GetImageFromArray(img_np)
    new_sitk.SetOrigin(img_sitk.GetOrigin())
    new_sitk.SetDirection(img_sitk.GetDirection())
    new_sitk.SetSpacing(img_sitk.GetSpacing())
    sitk.WriteImage(new_sitk,os.path.join(base_path,save_path,fname))
