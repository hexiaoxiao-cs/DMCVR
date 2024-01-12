from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *
from .unet import *
from choices import *


@dataclass
class BeatGANsE2EConfig(BaseConfig):
    image_size: int
    in_channels: int
    model_channels: int
    out_hid_channels: int
    out_channels: int
    embed_channels: int
    num_res_blocks: int
    attention_resolutions: Tuple[int]
    dropout: float = 0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    use_time_condition: bool = True
    conv_resample: bool = True
    dims: int = 2
    use_checkpoint: bool = False
    attn_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    pool: str = 'adaptivenonzero'
    resnet_two_cond : bool = False
    resnet_use_zero_module : bool = False
    resnet_cond_channels: int = None
    resnet_use_zero_module: bool = False
    def make_model(self):
        return BeatGANsE2EModel(self)


class BeatGANsE2EModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """
    def __init__(self, conf: BeatGANsE2EConfig):
        super().__init__()
        self.conf = conf
        print(self.conf)
        print("BEATS_GAN_E2E_CONF")
        self.dtype = th.float32
        #UnCHANGED ARIST
        # self.conf.in_channels=512 ####HARCODED
        # self.conf.model_channels=512 ####HARDCODED
        if conf.num_heads_upsample == -1:
            self.num_heads_upsample = conf.num_heads
        if conf.use_time_condition:
            time_embed_dim = conf.model_channels * 4
            self.time_embed = nn.Sequential(
                linear(conf.model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        input_ch = ch = int(conf.channel_mult[0] * conf.model_channels) #HARD CODED conf.model_channels
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))#CHANGED ARIST
        ])
        self._feature_size = ch
        input_block_chans = [[] for _ in range(len(conf.channel_mult))]
        input_block_chans[0].append(ch)
        kwargs = dict(
            use_condition=False,
            two_cond=conf.resnet_two_cond,
            use_zero_module=conf.resnet_use_zero_module,
            # style channels for the resnet block
            cond_emb_channels=conf.resnet_cond_channels,
        )
        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        ds = 1
        resolution = conf.image_size
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        time_embed_dim,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_condition=conf.use_time_condition,
                        use_checkpoint=conf.use_checkpoint,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                self.input_num_blocks[level] += 1
                # input_block_chans.append(ch)
                input_block_chans[level].append(ch)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            time_embed_dim,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_condition=conf.use_time_condition,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Downsample(ch,
                                          conf.conv_resample,
                                          dims=conf.dims,
                                          out_channels=out_ch)))
                ch = out_ch
                # input_block_chans.append(ch)
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
        )
        self._feature_size += ch
        if conf.pool == "adaptivenonzero":
            self.out_latent = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(conf.dims, ch, conf.out_channels, 1),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"Unexpected {conf.pool} pooling")
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(conf.channel_mult))[::-1]:
            for i in range(conf.num_res_blocks + 1):
                # print(input_block_chans)
                # ich = input_block_chans.pop()
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connecions for all decoder blocks
                    ich = 0
                # print('pop:', ich)
                layers = [
                    ResBlockConfig(
                        # only direct channels when gated
                        channels=ch + ich,
                        emb_channels=conf.embed_channels,
                        dropout=conf.dropout,
                        out_channels=int(conf.model_channels * mult),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        # lateral channels are described here when gated
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(conf.model_channels * mult)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                if level and i == conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            up=True,
                            **kwargs,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Upsample(ch,
                                        conf.conv_resample,
                                        dims=conf.dims,
                                        out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch
        # if conf.resnet_use_zero_module:
        #     self.out = nn.Sequential(
        #         normalization(ch),
        #         nn.SiLU(),
        #         zero_module(
        #             conv_nd(conf.dims,
        #                     input_ch,
        #                     conf.out_channels,
        #                     3,
        #                     padding=1)),
        #     )
        # else:
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(conf.dims, input_ch, 4 , 3, padding=1), #HARD CODED ARIST FOR 3 CATEGORIES + 1 background 
        )



    def forward(self, x, t=None, return_2d_feature=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        # print(x.shape)
        if self.conf.use_time_condition:
            emb = self.time_embed(timestep_embedding(t, self.model_channels))
        else:
            emb = None

        results = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        h = x.type(self.dtype)

        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                # print(i, j, h.shape)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)
        # for module in self.input_blocks:
        #     h = module(h, emb=emb)
        #     hs[i].append(h)
        #     if self.conf.pool.startswith("spatial"):
        #         results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb=emb)
        k = 0
        if self.conf.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            middle = th.cat(results, axis=-1)
        else:
            middle = h.type(x.dtype)

        h_2d = middle
        latent_code = self.out_latent(middle)
        
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        h = h.type(x.dtype)
        pred = self.out(h)
        return pred, latent_code
        # if return_2d_feature:
        #     return h, h_2d
        # else:
        #     return h

    def forward_flatten(self, x):
        """
        transform the last 2d feature into a flatten vector
        """
        h = self.out(x)
        return h




@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None
    print("!!!!!USING SEG E2E MODEL!!!!!!")

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf
        
        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )
        # self.seg_style_conv = conv_nd(2, 512, 512, 3, padding=1) ##ARIST CHANGED 

        self.encoder = BeatGANsE2EConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            embed_channels=conf.embed_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()

        # self.decoder=None

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):
        # breakpoint()
        y,cond = self.encoder.forward(x)
        return {'cond': cond,"y":y}

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S

    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                seg_style=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

        if t_cond is None:
            t_cond = t

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)


        # if cond is not None:
        #     ##CHANGED ARIST
        #     # print("!!!!!!!!!!RECEIVED CONDS WITH DIM!!!!!!!!!!!!",str(cond.shape))
        #     tmp=self.encode(cond)
        #     cond=tmp['cond']
        # elif "cond" in kwargs.keys():
        #     # print("!!!!!!!!!!RECEIVED CONDS WITH DIM!!!!!!!!!!!!",str(cond.shape))
        #     tmp=self.encode(kwargs.get("cond"))
        #     cond=tmp['cond']
        y=None
        y_=None
        if cond is None:
            #print("!!!!!!!COND NONE!!!!!!!!!!")
            if x is not None:
                # breakpoint()
                assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'
            
            tmp= self.encode(x_start)
            cond = tmp['cond']
            # print("!!!!!!!!Got Y!!!!!!!!")
            y_= tmp['y']

        # if seg_style is None:
        #     print("!!!!!!!!!SEG STYLE NONE!!!!!!!!!!")
        #     seg_style=
        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        # breakpoint()
        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h,
                                            emb=enc_time_emb,
                                            cond=enc_cond_emb)

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1

        pred = self.out(h)
        return AutoencReturn(pred=pred, cond=cond,y=y_)


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None
    y: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)
