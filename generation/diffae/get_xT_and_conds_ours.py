from templates import *
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
#Config of the generating parameters

skipping=1 #1 is interpolating each slice (filling the blanks) n,n+1,n+2
filling=1 #1 is filling one slice between each skipping 



t=20



'''
For example, skipping=0,filling=1 means the following:
I will use the latent code for each slice and interpolate one slice between slices (basically make the images size*2)
'''
device = 'cuda:0'
nodes = 1
conf = autoenc_base()
conf.data_name = 'medical_dataset_latent'
conf.scale_up_gpus(1)
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
conf.net_enc_channel_mult = (1, 1, 1, 1, 1) #16->16->8->4->2->1
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
conf.name = 'med256_autoenc_our_code_finalized'
# return conf
from experiment_ours import *

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

save_to_path="/filer/tmp1/xh172/generated_ukbb_xT_cond_ours/T"+str(t)+"/"

if not os.path.exists(os.path.join(save_to_path,"img")):
    os.makedirs(os.path.join(save_to_path,"img"))

files=glob.glob(os.path.join(base_path,"img","*.nii.gz"))


for i in tqdm(files):
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
    conds=[]
    for j in range(len(img)):
        conds.append(np.load(i.replace(os.path.join(base_path,"img"),path_to_latent).replace(".nii.gz","_"+str(j)+".npy")))
    latents=np.asarray(conds)
    
    #filtered=np.asarray([i for i in img if not i.max()==0])
    filtered_indexes=[i for i in range(len(img)) if not img[i].max()==0]
    #here we sort something
    filtered=img[filtered_indexes]
    latents=latents[filtered_indexes]
    latents_tensor=torch.from_numpy(latents).float()
    tensor_img = torch.from_numpy(filtered).float()
    input_tensor=tensor_img.unsqueeze(1).to(device)
    latents_tensor=latents_tensor.to(device)
    conds = model.encode(latents_tensor)
    xTs=model.encode_stochastic(input_tensor,latents_tensor,T=t)
    conds=conds.cpu().numpy()
    xTs=xTs.cpu().numpy()
    tosave_path=i.replace(base_path,save_to_path)
    np.savez(tosave_path.replace(".nii.gz",".npz"),conds=conds,xTs=xTs) #Change to better support
    # np.savez_compressed(tosave_path.replace(".nii.gz","_xTs.npz"),xTs)
