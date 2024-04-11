from .PanoGenerator import PanoGenerator
from ..modules.utils import tensor_to_image
from .MVGenModel import MultiViewBaseModel
import torch
import os
from PIL import Image
from external.Perspective_and_Equirectangular import e2p
from einops import rearrange
from lightning.pytorch.utilities import rank_zero_only


class PanFusion(PanoGenerator):
    def __init__(
            self,
            use_pers_prompt: bool = True,
            use_pano_prompt: bool = True,
            copy_pano_prompt: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def instantiate_model(self):
        pano_unet, cn = self.load_pano()
        unet, pers_cn = self.load_pers()
        self.mv_base_model = MultiViewBaseModel(unet, pano_unet, pers_cn, cn, self.hparams.unet_pad)
        if not self.hparams.layout_cond:
            self.trainable_params.extend(self.mv_base_model.trainable_parameters)

    def init_noise(self, bs, equi_h, equi_w, pers_h, pers_w, cameras, device):
        cameras = {k: rearrange(v, 'b m ... -> (b m) ...') for k, v in cameras.items()}
        pano_noise = torch.randn(
            bs, 1, 4, equi_h, equi_w, device=device)
        pano_noises = pano_noise.expand(-1, len(cameras['FoV']), -1, -1, -1)
        pano_noises = rearrange(pano_noises, 'b m c h w -> (b m) c h w')
        noise = e2p(
            pano_noises,
            cameras['FoV'], cameras['theta'], cameras['phi'],
            (pers_h, pers_w), mode='nearest')
        noise = rearrange(noise, '(b m) c h w -> b m c h w', b=bs, m=len(cameras['FoV']))
        # noise_sample = noise[0, 0, :3]
        # pano_noise_sample = pano_noise[0, 0, :3]
        return pano_noise, noise

    def embed_prompt(self, batch, num_cameras):
        if self.hparams.use_pers_prompt:
            pers_prompt = self.get_pers_prompt(batch)
            pers_prompt_embd = self.encode_text(pers_prompt)
            pers_prompt_embd = rearrange(pers_prompt_embd, '(b m) l c -> b m l c', m=num_cameras)
        else:
            pers_prompt = ''
            pers_prompt_embd = self.encode_text(pers_prompt)
            pers_prompt_embd = pers_prompt_embd[:, None].repeat(1, num_cameras, 1, 1)

        if self.hparams.use_pano_prompt:
            pano_prompt = self.get_pano_prompt(batch)
        else:
            pano_prompt = ''
        pano_prompt_embd = self.encode_text(pano_prompt)
        pano_prompt_embd = pano_prompt_embd[:, None]

        return pers_prompt_embd, pano_prompt_embd

    def training_step(self, batch, batch_idx):
        device = batch['images'].device
        latents = self.encode_image(batch['images'], self.vae)
        b, m, c, h, w = latents.shape

        pano_pad = self.pad_pano(batch['pano'])
        pano_latent_pad = self.encode_image(pano_pad, self.vae)
        pano_latent = self.unpad_pano(pano_latent_pad, latent=True)
        # # test encoded pano latent
        # pano_pad = ((pano_pad[0, 0] + 1) * 127.5).cpu().numpy().astype(np.uint8)
        # pano = ((batch['pano'][0, 0] + 1) * 127.5).cpu().numpy().astype(np.uint8)
        # pano_decode = self.decode_latent(pano_latent, self.vae)[0, 0]

        t = torch.randint(0, self.scheduler.config.num_train_timesteps,
                          (b,), device=latents.device).long()
        pers_prompt_embd, pano_prompt_embd = self.embed_prompt(batch, m)
        pano_noise, noise = self.init_noise(
            b, *pano_latent.shape[-2:], h, w, batch['cameras'], device)

        noise_z = self.scheduler.add_noise(latents, noise, t)
        pano_noise_z = self.scheduler.add_noise(pano_latent, pano_noise, t)
        t = t[:, None].repeat(1, m)

        denoise, pano_denoise = self.mv_base_model(
            noise_z, pano_noise_z, t, pers_prompt_embd, pano_prompt_embd, batch['cameras'],
            batch.get('images_layout_cond'), batch.get('pano_layout_cond'))

        # eps mode
        loss_pers = torch.nn.functional.mse_loss(denoise, noise)
        loss_pano = torch.nn.functional.mse_loss(pano_denoise, pano_noise)
        loss = loss_pers + loss_pano
        self.log('train/loss', loss, prog_bar=False)
        self.log('train/loss_pers', loss_pers, prog_bar=True)
        self.log('train/loss_pano', loss_pano, prog_bar=True)
        return loss

    @torch.no_grad()
    def forward_cls_free(self, latents, pano_latent, timestep, prompt_embd, pano_prompt_embd, batch, pano_layout_cond=None):
        latents, pano_latent, timestep, cameras, images_layout_cond, pano_layout_cond = self.gen_cls_free_guide_pair(
            latents, pano_latent, timestep, batch['cameras'],
            batch.get('images_layout_cond'), pano_layout_cond)

        noise_pred, pano_noise_pred = self.mv_base_model(
            latents, pano_latent, timestep, prompt_embd, pano_prompt_embd, cameras,
            images_layout_cond, pano_layout_cond)

        noise_pred, pano_noise_pred = self.combine_cls_free_guide_pred(noise_pred, pano_noise_pred)

        return noise_pred, pano_noise_pred

    def rotate_latent(self, pano_latent, cameras, degree=None):
        if degree is None:
            degree = self.hparams.rot_diff
        if degree % 360 == 0:
            return pano_latent, cameras

        pano_latent = super().rotate_latent(pano_latent, degree)
        cameras = cameras.copy()
        cameras['theta'] = (cameras['theta'] + degree) % 360
        return pano_latent, cameras

    @torch.no_grad()
    def inference(self, batch):
        bs, m = batch['cameras']['height'].shape[:2]
        h, w = batch['cameras']['height'][0, 0].item(), batch['cameras']['width'][0, 0].item()
        device = self.device

        pano_latent, latents = self.init_noise(
            bs, batch['height']//8, batch['width']//8, h//8, h//8, batch['cameras'], device)

        pers_prompt_embd, pano_prompt_embd = self.embed_prompt(batch, m)
        prompt_null = self.encode_text('')[:, None]
        pano_prompt_embd = torch.cat([prompt_null, pano_prompt_embd])
        prompt_null = prompt_null.repeat(1, m, 1, 1)
        pers_prompt_embd = torch.cat([prompt_null, pers_prompt_embd])

        self.scheduler.set_timesteps(self.hparams.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        pano_layout_cond = batch.get('pano_layout_cond')

        curr_rot = 0
        for i, t in enumerate(timesteps):
            timestep = torch.cat([t[None, None]]*m, dim=1)

            pano_latent, batch['cameras'] = self.rotate_latent(pano_latent, batch['cameras'])
            curr_rot += self.hparams.rot_diff

            if self.hparams.layout_cond:
                pano_layout_cond = super().rotate_latent(pano_layout_cond)
            else:
                pano_layout_cond = None
            noise_pred, pano_noise_pred = self.forward_cls_free(
                latents, pano_latent, timestep, pers_prompt_embd, pano_prompt_embd, batch, pano_layout_cond)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
            pano_latent = self.scheduler.step(
                pano_noise_pred, t, pano_latent).prev_sample

        pano_latent, batch['cameras'] = self.rotate_latent(pano_latent, batch['cameras'], -curr_rot)

        images_pred = self.decode_latent(latents, self.vae)
        images_pred = tensor_to_image(images_pred)

        pano_latent_pad = self.pad_pano(pano_latent, latent=True)
        pano_pred_pad = self.decode_latent(pano_latent_pad, self.vae)
        pano_pred = self.unpad_pano(pano_pred_pad)
        pano_pred = tensor_to_image(pano_pred)

        # # test encoded pano latent
        # img1 = self.decode_latent(pano_latent, self.vae).squeeze()
        # img1 = np.roll(img1, img1.shape[0]//2, axis=0)
        # img1 = np.roll(img1, img1.shape[1]//2, axis=1)
        # img2 = pano_pred.squeeze()
        # img2 = np.roll(img2, img2.shape[0]//2, axis=0)
        # img2 = np.roll(img2, img2.shape[1]//2, axis=1)

        return images_pred, pano_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred, pano_pred = self.inference(batch)
        self.log_val_image(images_pred, batch['images'], pano_pred, batch['pano'], batch['pano_prompt'],
                           batch.get('images_layout_cond'), batch.get('pano_layout_cond'))

    def inference_and_save(self, batch, output_dir, ext='png'):
        prompt_path = os.path.join(output_dir, 'prompt.txt')
        if os.path.exists(prompt_path):
            return

        _, pano_pred = self.inference(batch)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"pano.{ext}")
        im = Image.fromarray(pano_pred[0, 0])
        im.save(path)

        with open(prompt_path, 'w') as f:
            f.write(batch['pano_prompt'][0]+'\n')

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, images_pred, images, pano_pred, pano, pano_prompt,
                      images_layout_cond=None, pano_layout_cond=None):
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
