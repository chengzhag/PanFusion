import torch
import torch.nn as nn
from ..modules.transformer import BasicTransformerBlock, SphericalPE
from .utils import get_coords, get_masks
from einops import rearrange, repeat


class WarpAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = SphericalPE(dim//4)

    def forward(self, pers_x, equi_x, cameras):
        bm, c, pers_h, pers_w = pers_x.shape
        b, c, equi_h, equi_w = equi_x.shape
        m = bm // b
        pers_masks, equi_masks = get_masks(
            pers_h, pers_w, equi_h, equi_w, cameras, pers_x.device, pers_x.dtype)
        pers_coords, equi_coords = get_coords(
            pers_h, pers_w, equi_h, equi_w, cameras, pers_x.device, pers_x.dtype)

        # # cross attention from perspective to equirectangular
        # pers_xs = rearrange(pers_xs, '(b m) c h w -> b m c h w', m=m)
        # pers_masks = rearrange(pers_masks, '(b m) ... -> b m ...', m=m)
        # equi_masks = rearrange(equi_masks, '(b m) ... -> b m ...', m=m)
        # pers_xs_out, equi_xs_out = [], []
        # for pers_x, equi_x, pers_mask, equi_mask in zip(pers_xs, equi_xs, pers_masks, equi_masks):
        #     key_value = rearrange(pers_x, 'm c h w -> (m h w) c', m=m)
        #     query = rearrange(equi_x, 'c h w -> (h w) c')
        #     pers_mask = rearrange(pers_mask, 'm h w -> b (m h w)', m=m)
        #     out = self.transformer(query, key_value, pers_masks)

        # add positional encoding
        pers_pe = self.pe(pers_coords)
        pers_pe = rearrange(pers_pe, 'b h w c -> b c h w')
        pers_x_wpe = pers_x + pers_pe
        equi_pe = self.pe(equi_coords)
        equi_pe = repeat(equi_pe, 'h w c -> b c h w', b=b)
        equi_x_wpe = equi_x + equi_pe

        # cross attention from perspective to equirectangular
        query = rearrange(equi_x, 'b c h w -> b (h w) c')
        key_value = rearrange(pers_x_wpe, '(b m) c h w -> b (m h w) c', m=m)
        pers_masks = rearrange(pers_masks, '(b m) eh ew ph pw -> b (eh ew) (m ph pw)', m=m)
        equi_pe = rearrange(equi_pe, 'b c h w -> b (h w) c')
        equi_x_out = self.transformer(query, key_value, mask=pers_masks, query_pe=equi_pe)

        # cross attention from equirectangular to perspective
        query = rearrange(pers_x, '(b m) c h w -> b (m h w) c', m=m)
        key_value = rearrange(equi_x_wpe, 'b c h w -> b (h w) c')
        equi_masks = rearrange(equi_masks, '(b m) ph pw eh ew -> b (m ph pw) (eh ew)', m=m)
        pers_pe = rearrange(pers_pe, '(b m) c h w -> b (m h w) c', m=m)
        pers_x_out = self.transformer(query, key_value, mask=equi_masks, query_pe=pers_pe)

        pers_x_out = rearrange(pers_x_out, 'b (m h w) c -> (b m) c h w', m=m, h=pers_h, w=pers_w)
        equi_x_out = rearrange(equi_x_out, 'b (h w) c -> b c h w', h=equi_h, w=equi_w)
        return pers_x_out, equi_x_out
