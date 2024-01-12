from typing import Union
from .unet import BeatGANsUNetModel, BeatGANsUNetConfig
from .unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel
from . import unet_autoenc_2nd
from . import unet_autoenc_seg_e2e

Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel,unet_autoenc_2nd.BeatGANsAutoencModel,unet_autoenc_seg_e2e.BeatGANsAutoencModel]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig,unet_autoenc_2nd.BeatGANsAutoencConfig,unet_autoenc_seg_e2e.BeatGANsAutoencModel]
