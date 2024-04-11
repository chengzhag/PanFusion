import torch
import torch.nn as nn
from .modules import CPAttn
from einops import rearrange
from .utils import get_correspondences


class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, pers_cn=None):
        super().__init__()

        self.unet = unet
        self.pers_cn = pers_cn

        self.cp_blocks_encoder = nn.ModuleList()
        for i in range(len(self.unet.down_blocks)):
            self.cp_blocks_encoder.append(CPAttn(
                self.unet.down_blocks[i].resnets[-1].out_channels, flag360=True))

        self.cp_blocks_mid = CPAttn(
            self.unet.mid_block.resnets[-1].out_channels, flag360=True)

        self.cp_blocks_decoder = nn.ModuleList()
        for i in range(len(self.unet.up_blocks)):
            self.cp_blocks_decoder.append(CPAttn(
                self.unet.up_blocks[i].resnets[-1].out_channels, flag360=True))

        self.trainable_parameters = [(list(self.cp_blocks_mid.parameters()) + \
            list(self.cp_blocks_decoder.parameters()) + \
            list(self.cp_blocks_encoder.parameters()), 1.0)]
    
    def forward(self, latents, timestep, prompt_embd, meta, pers_layout_cond=None):
        K = meta['K']
        R = meta['R']
        
        b, m, c, h, w = latents.shape
        img_h, img_w = h*8, w*8
        correspondences=get_correspondences(R, K, img_h, img_w)

        # bs*m, 4, 64, 64
        hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')
        prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps

        timestep = timestep.reshape(-1)
        t_emb = self.unet.time_proj(timestep)  # (bs, 320)
        emb = self.unet.time_embedding(t_emb)  # (bs, 1280)

        if self.pers_cn is None:
            pers_layout_cond = None
        if pers_layout_cond is not None:
            pers_layout_cond = rearrange(pers_layout_cond, 'b m ... -> (b m) ...')
            down_block_additional_residuals, mid_block_additional_residual = self.pers_cn(
                hidden_states,
                timestep,
                encoder_hidden_states=prompt_embd,
                controlnet_cond=pers_layout_cond,
                return_dict=False,
            )

        hidden_states = self.unet.conv_in(
            hidden_states)  # bs*m, 320, 64, 64

        # unet
        # a. downsample
        down_block_res_samples = (hidden_states,)
        for i, downsample_block in enumerate(self.unet.down_blocks):
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):
                    hidden_states = resnet(hidden_states, emb)

                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample

                    down_block_res_samples += (hidden_states,)
            else:
                for resnet in downsample_block.resnets:
                    hidden_states = resnet(hidden_states, emb)
                    down_block_res_samples += (hidden_states,)
            if m > 1:
                hidden_states = self.cp_blocks_encoder[i](
                    hidden_states, correspondences, img_h, img_w, R, K, m)

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        if pers_layout_cond is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # b. mid

        hidden_states = self.unet.mid_block.resnets[0](
            hidden_states, emb)

        if m > 1:
            hidden_states = self.cp_blocks_mid(
                hidden_states, correspondences, img_h, img_w, R, K, m)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = resnet(hidden_states, emb)

        if pers_layout_cond is not None:
            hidden_states = hidden_states + mid_block_additional_residual
        h, w = hidden_states.shape[-2:]

        # c. upsample
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                for resnet, attn in zip(upsample_block.resnets, upsample_block.attentions):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
            else:
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
            if m > 1:
                hidden_states = self.cp_blocks_decoder[i](
                    hidden_states,correspondences, img_h, img_w, R, K, m)

            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # 4.post-process
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        return sample
