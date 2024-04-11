from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch
import os
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from external.MVDiffusion.pano.MVGenModel import MultiViewBaseModel
from .PanoGenerator import PanoGenerator
from external.Perspective_and_Equirectangular import mp2e
from lightning.pytorch.utilities import rank_zero_only
from einops import rearrange
from ..modules.utils import tensor_to_image


class MvDiffusion(PanoGenerator):
    def __init__(
            self,
            pers_prompt_prefix: str = 'This is one view of a scene.',
            pers_lora: bool = True,
            train_pers_lora: bool = True,
            cam_sampler: str = 'horizon',
            copy_pano_prompt: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def instantiate_model(self):
        unet, cn = self.load_pers()
        self.mv_base_model = MultiViewBaseModel(unet, cn)
        if not self.hparams.layout_cond:
            self.trainable_params.extend(self.mv_base_model.trainable_parameters)

    def embed_prompt(self, batch, num_cameras):
        pers_prompt = self.get_pers_prompt(batch)
        prompt_embds = self.encode_text(pers_prompt)
        prompt_embds = rearrange(prompt_embds, '(b m) l c -> b m l c', m=num_cameras)
        batch['pers_prompt'] = pers_prompt
        return prompt_embds

    def training_step(self, batch, batch_idx):      
        meta = {
            'K': batch['cameras']['K'],
            'R': batch['cameras']['R']
        }

        latents = self.encode_image(batch['images'], self.vae)
        t = torch.randint(0, self.scheduler.num_train_timesteps,
                          (latents.shape[0],), device=latents.device).long()
        prompt_embds = self.embed_prompt(batch, latents.shape[1])

        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t)
        t = t[:, None].repeat(1, latents.shape[1])
        denoise = self.mv_base_model(
            noise_z, t, prompt_embds, meta, batch.get('images_layout_cond'))
        target = noise

        # eps mode
        loss = torch.nn.functional.mse_loss(denoise, target)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch, images_layout_cond=None):
        latents = torch.cat([latents]*2)
        timestep = torch.cat([timestep]*2)
        images_layout_cond = torch.cat([images_layout_cond]*2) if images_layout_cond is not None else None
        
        R = torch.cat([batch['cameras']['R']]*2)
        K = torch.cat([batch['cameras']['K']]*2)
      
        meta = {
            'K': K,
            'R': R,
        }

        return latents, timestep, prompt_embd, meta, images_layout_cond

    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
        latents, _timestep, _prompt_embd, meta, images_layout_cond = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch, batch.get('images_layout_cond'))

        noise_pred = model(
            latents, _timestep, _prompt_embd, meta, images_layout_cond)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.hparams.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def inference(self, batch):
        bs, m = batch['cameras']['height'].shape[:2]
        h, w = batch['cameras']['height'][0, 0].item(), batch['cameras']['width'][0, 0].item()
        device = self.device

        latents = torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embds = self.embed_prompt(batch, m)
        prompt_null = self.encode_text('')
        prompt_embd = torch.cat([prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.hparams.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1)

            noise_pred = self.forward_cls_free(
                latents, _timestep, prompt_embd, batch, self.mv_base_model)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
        images_pred = self.decode_latent(latents, self.vae)
        images_pred = tensor_to_image(images_pred)

        return images_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred = self.inference(batch)
        self.log_val_image(images_pred, batch)

    def inference_and_save(self, batch, output_dir, ext='png'):
        prompt_path = os.path.join(output_dir, 'prompt.txt')
        if os.path.exists(prompt_path):
            return

        images_pred = self.inference(batch)

        os.makedirs(output_dir, exist_ok=True)
        for i in range(images_pred.shape[1]):
            path = os.path.join(output_dir, f"{i}.{ext}")
            im = Image.fromarray(images_pred[0, i])
            im.save(path)

        path = os.path.join(output_dir, f"pano.{ext}")
        images_pred = images_pred[0]
        fov_degs, u_degs, v_degs = [batch['cameras'][k].cpu().numpy()[0] for k in ('FoV', 'theta', 'phi')]
        pano_pred = mp2e(images_pred, fov_degs, u_degs, v_degs,
                         (batch['height'][0].item(), batch['width'][0].item()))
        im = Image.fromarray(pano_pred)
        im.save(path)

        with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
            for p in batch['pers_prompt']:
                f.write(p+'\n')

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, images_pred, batch):
        fov_degs, u_degs, v_degs = [batch['cameras'][k].cpu().numpy()[0] for k in ('FoV', 'theta', 'phi')]
        pano_pred = mp2e(images_pred[0], fov_degs, u_degs, v_degs,
                         (batch['height'][0].item(), batch['width'][0].item()))[None, None, ...]

        images, pano, pano_prompt = batch['images'], batch['pano'], batch['pano_prompt']
        images_layout_cond, pano_layout_cond = batch.get('images_layout_cond'), batch.get('pano_layout_cond')
        log_dict = {f"val/{k}_pred": v for k, v in self.temp_wandb_images(
            images_pred, pano_pred, None, pano_prompt).items()}
        log_dict.update({f"val/{k}_gt": v for k, v in self.temp_wandb_images(
            images, pano, None, pano_prompt).items()})
        if images_layout_cond is not None and pano_layout_cond is not None:
            log_dict.update({f"val/{k}_layout_cond": v for k, v in self.temp_wandb_images(
                images_layout_cond, pano_layout_cond, None, pano_prompt).items()})
        self.logger.experiment.log(log_dict)

    def temp_wandb_images(self, images, pano, prompt=None, pano_prompt=None):
        log_dict = {}
        pers = []
        for m_i in range(images.shape[1]):
            pers.append(self.temp_wandb_image(
                images[0, m_i], prompt[m_i][0] if prompt else None))
        log_dict['pers'] = pers

        log_dict['pano'] = self.temp_wandb_image(
            pano[0, 0], pano_prompt[0] if pano_prompt else None)
        return log_dict
