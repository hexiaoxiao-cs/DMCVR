from templates import *
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
#Config of the generating parameters

skipping=1 #1 is interpolating each slice (filling the blanks) n,n+1,n+2
filling=1 #1 is filling one slice between each skipping 

'''
For example, skipping=0,filling=1 means the following:
I will use the latent code for each slice and interpolate one slice between slices (basically make the images size*2)
'''
t=7
device = 'cuda:'+str(t)
conf = autoenc_base()
conf.img_size = 256
conf.net_ch = 128
conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
conf.eval_every_samples = 10_000_000
conf.eval_ema_every_samples = 10_000_000
conf.total_samples = 200_000_000
# conf.batch_size = 1
conf.make_model_conf()
conf.name = 'med256_autoenc'
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=True)
model.ema_model.eval()
model.ema_model.to(device)
# data=MedicalDataset(path="/filer/tmp1/xh172/ukbb/train_extracted/",image_size=conf.img_size)

#NEED TO CHECK THE IMAGE HAS CONTENT INSIDE OTHERWISE IDK WHAT GOING TO HAPPEN

# for i in range()

print(t)
base_path="/filer/tmp1/xh172/ukbb/train_extracted/"
save_to_path="/filer/tmp1/xh172/generated_ukbb_xT_cond/"

if not os.path.exists(os.path.join(save_to_path,"img")):
    os.makedirs(os.path.join(save_to_path,"img"))

files=glob.glob(os.path.join(base_path,"img","*.nii.gz"))

for i in tqdm.tqdm(files[t*5000:(t+1)*5000]):
    img_itk=sitk.ReadImage(i)
    img=sitk.GetArrayFromImage(img_itk)
    # self.img_tensors.extend(img_np.)
    max98 = np.percentile(img, 98)
    img = np.clip(img, 0, max98)
    img = img / max98
    #Excluding zero matrices here:
    filtered=np.asarray([i for i in img if not i.max()==0])
    #here we extracting the key frames that need to be computed with the neural nets
    # filtered=np.asarray([filtered[i] for i in range(len(filtered)) if i%(skipping+1)==0])
    # print(filtered.shape)
    tensor_img = torch.from_numpy(filtered).float()
    input_tensor=tensor_img.unsqueeze(1).to(device)
    conds = model.encode(input_tensor)
    xTs=model.encode_stochastic(input_tensor,conds,T=100)
    conds=conds.cpu().numpy()
    xTs=xTs.cpu().numpy()
    tosave_path=i.replace(base_path,save_to_path)
    np.savez_compressed(tosave_path.replace(".nii.gz",".npz"),conds=conds,xTs=xTs) #Change to better support
    # np.savez_compressed(tosave_path.replace(".nii.gz","_xTs.npz"),xTs)