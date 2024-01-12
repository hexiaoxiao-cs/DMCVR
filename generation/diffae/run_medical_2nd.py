from templates import *
from templates_latent import *
from torch import autograd
#2nd basically concat seg features in the network at 5th block, and use the semantic features 
if __name__ == '__main__':
    # 256 requires 8x v100s, in our case, on two nodes.
    # do not run this directly, use `sbatch run_ffhq256.sh` to spawn the srun properly.
    gpus = [2]
    nodes = 1
    conf = autoenc_base()
    conf.model_name = ModelName.beatgans_autoenc_2nd
    conf.data_name = 'medical_dataset_latent'
    # conf.scale_up_gpus(4)
    conf.num_workers=16
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
    # conf.eval_every_samples = 20_000_000
    # conf.eval_ema_every_samples = 20_000_000
    conf.eval_every_samples = 1
    conf.eval_ema_every_samples = 1
    # conf.eval_every_samples=10_000
    # conf.eval_ema_every_samples=10_000
    conf.total_samples = 300_000_000
    # conf.batch_size = 32
    # conf.batch_size_eval=32
    conf.batch_size = 1
    conf.batch_size_eval=1
    conf.make_model_conf()
    conf.name = 'med256_autoenc_2nd_final_test'
    conf.dataset_seg_latent = True
    # return conf
    from experiment_2nd import train
    with autograd.detect_anomaly():
        train(conf, gpus=gpus, nodes=nodes)