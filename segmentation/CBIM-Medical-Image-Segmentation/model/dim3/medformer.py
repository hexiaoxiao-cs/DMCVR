import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_block, get_norm, get_act
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion
import pdb



class MedFormer(nn.Module):
    
    def __init__(self, in_chan, num_classes, base_chan=32, map_size=[4,8,8], 
        conv_block='BasicBlock', conv_num=[2,1,0,0, 0,1,2,2], trans_num=[0,1,2,2, 2,1,0,0], chan_num=[64,128,256,320,256,128,64,32], num_heads=[1,4,8,16, 8,4,1,1], fusion_depth=2, fusion_dim=320, fusion_heads=4, expansion=4, attn_drop=0., proj_drop=0., proj_type='depthwise', norm='in', act='gelu', kernel_size=[3,3,3,3], scale=[2,2,2,2], aux_loss=False):
        super().__init__()

        if conv_block == 'BasicBlock':
            dim_head = [chan_num[i]//num_heads[i] for i in range(8)]

        
        conv_block = get_block(conv_block)
        norm = get_norm(norm)
        act = get_act(act)
        
        # self.inc and self.down1 forms the conv stem
        self.inc = inconv(in_chan, base_chan, block=conv_block, kernel_size=kernel_size[0], norm=norm, act=act)
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block=conv_block, kernel_size=kernel_size[1], down_scale=scale[0], norm=norm, act=act, map_generate=False)
        
        # down2 down3 down4 apply the B-MHA blocks
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block=conv_block, kernel_size=kernel_size[2], down_scale=scale[1], heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block=conv_block, kernel_size=kernel_size[3], down_scale=scale[2], heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)


        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)

        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block=conv_block, kernel_size=kernel_size[3], up_scale=scale[3], heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)

        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block=conv_block, kernel_size=kernel_size[2], up_scale=scale[2], heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)

        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block=conv_block, kernel_size=kernel_size[1], up_scale=scale[1], norm=norm, act=act, map_shortcut=False)

        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block=conv_block, kernel_size=kernel_size[0], up_scale=scale[0], norm=norm, act=act, map_shortcut=False)

        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_out = nn.Conv3d(chan_num[5], num_classes, kernel_size=1)

        self.outc = nn.Conv3d(chan_num[7], num_classes, kernel_size=1)

    def forward(self, x):
       
        x0 = self.inc(x)
        x1, _ = self.down1(x0)
        x2, map2 = self.down2(x1)
        x3, map3 = self.down3(x2)
        x4, map4 = self.down4(x3)
        
        
        map_list = [map2, map3, map4]
        map_list = self.map_fusion(map_list)
        

        out, semantic_map = self.up1(x4, x3, map_list[2], map_list[1])
        out, semantic_map = self.up2(out, x2, semantic_map, map_list[0])

        if self.aux_loss:
            aux_out = self.aux_out(out)
            aux_out = F.interpolate(aux_out, size=x.shape[-3:], mode='trilinear', align_corners=True)

        out, semantic_map = self.up3(out, x1, semantic_map, None)
        out, semantic_map = self.up4(out, x0, semantic_map, None)
    
        out = self.outc(out)

        if self.aux_loss:
            return [out, aux_out]
        else:
            return out

