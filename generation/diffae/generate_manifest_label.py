import glob
import os
import json
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
path="/filer/tmp1/xh172/ukbb/train_extracted/"
# path_to_latent="/filer/tmp1/xh172/ukbb_latent_code/"
files=glob.glob(os.path.join(path,"img","*.nii.gz"))

files_l=[]
index_ref=[]
conds_p=[]

for i in tqdm(files):
    img_itk=sitk.ReadImage(i)
    img=sitk.GetArrayFromImage(img_itk)
    num_of_images=img.shape[0]
    files_l.extend([i]*num_of_images)
    index_ref.extend(range(num_of_images))
    
    # self.img_tensors.extend(img_np.)
    # for j in range(num_of_images):
    #     conds_p.append(i.replace(os.path.join(path,"img"),path_to_latent).replace(".nii.gz","_"+str(j)+".npy"))

dictionary={"files_l":files_l,"index_ref":index_ref,"conds_p":conds_p}

json_object = json.dumps(dictionary, indent=4)
 
# Writing to sample.json
with open("medical_dataset_manifest.json", "w") as outfile:
    outfile.write(json_object)