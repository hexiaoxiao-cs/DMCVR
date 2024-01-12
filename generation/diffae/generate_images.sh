#!/bin/bash -l
#SBATCH --output=/research/cbim/vast/xh172/diffusion/diffae/generate_images_med256_autoenc_testing.log
conda deactivate
conda activate diffu
python /research/cbim/vast/xh172/diffusion/diffae/generate_images.py 0