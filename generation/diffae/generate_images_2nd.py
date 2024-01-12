from templates import *
import matplotlib.pyplot as plt
import numpy as np
from math import acos, sin
import SimpleITK as sitk
import tqdm
import sys
from templates import *
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import shutil
#Config of the generating parameters

skipping=1 #1 is interpolating each slice (filling the blanks) n,n+1,n+2
filling=1 #1 is filling one slice between each skipping 



t=100

k=int(sys.argv[1])

'''
For example, skipping=0,filling=1 means the following:
I will use the latent code for each slice and interpolate one slice between slices (basically make the images size*2)
'''
device = 'cuda:'+str(k)
nodes = 1
conf = autoenc_base()
conf.data_name = 'medical_dataset_latent'
conf.scale_up_gpus(1)
conf.model_name = ModelName.beatgans_autoenc_2nd
# conf.img_size = 128
# conf.net_ch = 128
# # final resolution = 8x8
# conf.net_ch_mult = (1, 1, 2, 3, 4)
# final resolution = 4x4
# conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
# conf.eval_ema_every_samples = 10_000_000
# conf.eval_every_samples = 10_000_000
# conf.make_model_conf()
conf.img_size = 256
conf.net_ch = 128
conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
conf.eval_every_samples = 10_000_000
conf.eval_ema_every_samples = 10_000_000
# conf.eval_every_samples=10_000
# conf.eval_ema_every_samples=10_000
conf.total_samples = 200_000_000
#HERE IS FOR 8GPUS
conf.batch_size = 32
conf.batch_size_eval=32
#HERE IS FOR 4GPUS
# conf.batch_size = 32
# conf.batch_size_eval=32
# conf.eval_num_images=
conf.net_beatgans_embed_channels = 512
conf.num_workers=16
# conf.num_workers=1
conf.make_model_conf()
conf.name = 'med256_autoenc_2nd'
# return conf
from experiment_2nd import *

# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=True)
model.ema_model.eval()
model.ema_model.to(device)
# data=MedicalDataset(path="/filer/tmp1/xh172/ukbb/train_extracted/",image_size=conf.img_size)

#NEED TO CHECK THE IMAGE HAS CONTENT INSIDE OTHERWISE IDK WHAT GOING TO HAPPEN

# for i in range()


base_path="/filer/tmp1/xh172/ukbb/train_extracted/"
path_to_latent="/filer/tmp1/xh172/ukbb_latent_code/"

save_to_path="/filer/tmp1/xh172/generated_ukbb_2nd_lbled/T"+str(t)+"/"

if not os.path.exists(os.path.join(save_to_path,"img")):
    os.makedirs(os.path.join(save_to_path,"img"))

if not os.path.exists(os.path.join(save_to_path,"lbl")):
    os.makedirs(os.path.join(save_to_path,"lbl"))

files=glob.glob(os.path.join(base_path,"lbl","*.nii.gz"))


for i in tqdm(files):
    shutil.copyfile(i, i.replace(base_path,save_to_path))
    i=i.replace("_label","").replace("lbl","img")
    img_itk=sitk.ReadImage(i)
    img=sitk.GetArrayFromImage(img_itk)
    # self.img_tensors.extend(img_np.)
    max98 = np.percentile(img, 98)
    img = np.clip(img, 0, max98)
    img = img / max98
    #Excluding zero matrices here:
    
    #here we extracting the key frames that need to be computed with the neural nets
    # filtered=np.asarray([filtered[i] for i in range(len(filtered)) if i%(skipping+1)==0])
    # print(filtered.shape)
    try:
        conds=[]
        for j in range(len(img)):
            conds.append(np.load(i.replace(os.path.join(base_path,"img"),path_to_latent).replace(".nii.gz","_"+str(j)+".npy")))
        latents=np.asarray(conds)
    except:
        continue  
    
    #filtered=np.asarray([i for i in img if not i.max()==0])
    filtered_indexes=[i for i in range(len(img)) if not img[i].max()==0]
    #here we sort something
    filtered=img[filtered_indexes]
    latents=latents[filtered_indexes]
    latents_tensor=torch.from_numpy(latents).float()
    tensor_img = torch.from_numpy(filtered).float()
    input_tensor=tensor_img.unsqueeze(1).to(device)
    latents_tensor=latents_tensor.to(device)
    cond_tensor=model.encode(input_tensor)
    # conds = model.encode(latents_tensor)
    # def encode_stochastic(self, x, cond, seg_style,T=None):
    xTs=model.encode_stochastic(input_tensor,cond_tensor,latents_tensor,T=t)
    pred = model.render(xTs,cond=cond_tensor,seg_style=latents_tensor ,T=t)
    tosave_path=i.replace(base_path,save_to_path)
    img=sitk.GetImageFromArray(pred.squeeze(1).cpu().numpy())
    # breakpoint()
    img.SetSpacing((1.0,1.0,1.0))
    sitk.WriteImage(img,i.replace(base_path,save_to_path).replace("_xTs.npz",".nii.gz")) #Change to better support
    # np.savez_compressed(tosave_path.replace(".nii.gz","_xTs.npz"),xTs)

def lin_interpolate(slice_1,slice_2,num=filling):
    #num is how many slices need to be inserted
    alpha=1.0/(num-1.0)
    out=[]
    # out.append(slice_1)
    for i in range(num-1):
        out.append((i)*alpha*slice_2+(1.0-(i)*alpha)*slice_1)
    # out.append(slice_2)
    return out

def slerp(x0: torch.Tensor, x1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical Linear intERPolation
    Args:
        x0 (`torch.Tensor`): first tensor to interpolate between
        x1 (`torch.Tensor`): seconds tensor to interpolate between
        alpha (`float`): interpolation between 0 and 1
    Returns:
        `torch.Tensor`: interpolated tensor
    """

    theta = acos(torch.dot(torch.flatten(x0), torch.flatten(x1)) / torch.norm(x0) / torch.norm(x1))
    return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(alpha * theta) * x1 / sin(theta)

def slerp_np(x0: np.ndarray, x1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical Linear intERPolation
    Args:
        x0 (`torch.Tensor`): first tensor to interpolate between
        x1 (`torch.Tensor`): seconds tensor to interpolate between
        alpha (`float`): interpolation between 0 and 1
    Returns:
        `torch.Tensor`: interpolated tensor
    """

    theta = acos(np.dot(x0.flatten(), x1.flatten()) / np.linalg.norm(x0) / np.linalg.norm(x1))
    return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(alpha * theta) * x1 / sin(theta)

def slerp_interpolate(slice_1,slice_2,num=filling):
    #num is how many slices need to be inserted
    alpha=1.0/(num-1)
    out=[]
    # out.append(slice_1)
    for i in range(num-1):
        out.append(slerp_np(slice_1,slice_2,alpha*i))
    # out.append(slice_2)
    return out

# for i in tqdm.tqdm(files[0+10000*k:10000*(k+1)]):
#     xTs=np.load(i)["arr_0"]
#     conds=np.load(i.replace("xTs","conds"))["arr_0"]
#     #here we extracting the key frames that need to be computed with the neural nets
#     # filtered_xTs=np.asarray([xTs[j] for j in range(len(xTs)) if j%(skipping+1)==0])
#     # filtered_conds=np.asarray([conds[j] for j in range(len(conds)) if j%(skipping+1)==0])
#     # breakpoint()
#     preds=[]
#     # for j in range(len(xTs)):
#         #interpolated_conds.extend(lin_interpolate(filtered_conds[j],filtered_conds[j+1])) ###Changing LIN to SLERP
#     pred = model.render(torch.tensor(xTs).float().to(device),torch.tensor(conds).float().to(device) ,T=100)
#     #We predict the images,then we save the images
#     #Need to preserve the direction information for extracting the SAX/LAX informaiton (not in source,so don't matter right now)
#     img=sitk.GetImageFromArray(pred.squeeze(1).cpu().numpy())
#     # breakpoint()
#     img.SetSpacing((1.0,1.0,1.0))
#     sitk.WriteImage(img,i.replace(base_path,save_to_path).replace("_xTs.npz",".nii.gz"))


# for i in files:
#     img_itk=sitk.ReadImage(i)
#     img=sitk.GetArrayFromImage(img_itk)
#     # self.img_tensors.extend(img_np.)
#     max98 = np.percentile(img, 98)
#     img = np.clip(img, 0, max98)
#     img = img / max98
#     #Excluding zero matrices here:
#     filtered=[i for i in img if not i.max()==0]
#     #here we extracting the key frames that need to be computed with the neural nets
#     filtered=np.asarray([filtered[i] for i in range(len(filtered)) if i%(skipping+1)==0])
#     print(filtered.shape)
#     tensor_img = torch.from_numpy(filtered).float()
#     input_tensor=tensor_img.unsqueeze(1).to(device)
#     conds = model.encode(input_tensor)
#     xTs=model.encode_stochastic(input_tensor,conds,T=100)
