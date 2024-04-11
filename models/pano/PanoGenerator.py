import os
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange
from abc import abstractmethod
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
import copy
from utils.pano import pad_pano, unpad_pano
from ..modules.utils import WandbLightningModule
from diffusers import ControlNetModel


class PanoBase(WandbLightningModule):
    def __init__(
            self,
            pano_prompt_prefix: str = '',
            pers_prompt_prefix: str = '',
            mv_pano_prompt: bool = False,
            copy_pano_prompt: bool = False,
            ):
        super().__init__()
        self.save_hyperparameters()

    def add_pano_prompt_prefix(self, pano_prompt):
        if isinstance(pano_prompt, str):
            if pano_prompt == '':
                return ''
            if self.hparams.pano_prompt_prefix == '':
                return pano_prompt
            return ' '.join([self.hparams.pano_prompt_prefix, pano_prompt])
        return [self.add_pano_prompt_prefix(p) for p in pano_prompt]

    def add_pers_prompt_prefix(self, pers_prompt):
        if isinstance(pers_prompt, str):
            if pers_prompt == '':
                return ''
            if self.hparams.pers_prompt_prefix == '':
                return pers_prompt
            return ' '.join([self.hparams.pers_prompt_prefix, pers_prompt])
        return [self.add_pers_prompt_prefix(p) for p in pers_prompt]

    def get_pano_prompt(self, batch):
        if self.hparams.mv_pano_prompt:
            prompts = list(map(list, zip(*batch['prompt'])))
            pano_prompt = ['. '.join(p1) if p2 else '' for p1, p2 in zip(prompts, batch['pano_prompt'])]
        else:
            pano_prompt = batch['pano_prompt']
        return self.add_pano_prompt_prefix(pano_prompt)

    def get_pers_prompt(self, batch):
        if self.hparams.copy_pano_prompt:
            prompts = sum([[p] * batch['cameras']['height'].shape[-1] for p in batch['pano_prompt']], [])
        else:
            prompts = sum(map(list, zip(*batch['prompt'])), [])
        return self.add_pers_prompt_prefix(prompts)


class PanoGenerator(PanoBase):
    def __init__(
            self,
            lr: float = 2e-4,
            guidance_scale: float = 9.0,
            model_id: str = 'stabilityai/stable-diffusion-2-base',
            diff_timestep: int = 50,
            latent_pad: int = 8,
            pano_lora: bool = True,
            train_pano_lora: bool = True,
            pers_lora: bool = True,
            train_pers_lora: bool = True,
            lora_rank: int = 4,
            ckpt_path: str = None,
            rot_diff: float = 90.0,
            layout_cond: bool = False,
            pers_layout_cond: bool = False,
            unet_pad: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.trainable_params = []
        self.save_hyperparameters()
        self.load_shared()
        self.instantiate_model()
        if ckpt_path is not None:
            print(f"Loading weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path)['state_dict']
            self.convert_state_dict(state_dict)
            try:
                self.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(e)
                self.load_state_dict(state_dict, strict=False)

    def exclude_eval_metrics(self, checkpoint):
        for key in list(checkpoint['state_dict'].keys()):
            if key.startswith('eval_metrics'):
                del checkpoint['state_dict'][key]

    def convert_state_dict(self, state_dict):
        for old_k in list(state_dict.keys()):
            new_k = old_k.replace('to_q.lora_layer', 'processor.to_q_lora')
            new_k = new_k.replace('to_k.lora_layer', 'processor.to_k_lora')
            new_k = new_k.replace('to_v.lora_layer', 'processor.to_v_lora')
            new_k = new_k.replace('to_out.0.lora_layer', 'processor.to_out_lora')
            state_dict[new_k] = state_dict.pop(old_k)

    def on_load_checkpoint(self, checkpoint):
        self.exclude_eval_metrics(checkpoint)
        self.convert_state_dict(checkpoint['state_dict'])

    def on_save_checkpoint(self, checkpoint):
        self.exclude_eval_metrics(checkpoint)

    def load_shared(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.hparams.model_id, subfolder="tokenizer", torch_dtype=torch.float16, use_safetensors=True)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.hparams.model_id, subfolder="text_encoder", torch_dtype=torch.float16)
        self.text_encoder.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(
            self.hparams.model_id, subfolder="vae", torch_dtype=torch.float16, use_safetensors=True)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae = torch.compile(self.vae)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.hparams.model_id, subfolder="scheduler", torch_dtype=torch.float16, use_safetensors=True)

    def add_lora(self, unet):
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=self.hparams.lora_rank,
            )
        unet.set_attn_processor(lora_attn_procs)
        return (AttnProcsLayers(unet.attn_processors).parameters(), 1.0)

    def get_cn(self, unet):
        cn = ControlNetModel.from_unet(unet)
        cn.enable_xformers_memory_efficient_attention()
        cn.enable_gradient_checkpointing()
        return cn, (list(cn.parameters()), 0.1)

    def load_branch(self, add_lora, train_lora, add_cn):
        unet = UNet2DConditionModel.from_pretrained(
            self.hparams.model_id, subfolder="unet", torch_dtype=torch.float32, use_safetensors=True)
        unet.enable_xformers_memory_efficient_attention()
        unet.enable_gradient_checkpointing()

        if add_cn:
            cn, params = self.get_cn(unet)
            self.trainable_params.append(params)
        else:
            cn = None

        if add_lora:
            params = self.add_lora(unet)
            if train_lora and not add_cn:
                self.trainable_params.append(params)

        unet = torch.compile(unet)
        return unet, cn

    def load_pano(self):
        return self.load_branch(
            self.hparams.pano_lora,
            self.hparams.train_pano_lora,
            self.hparams.layout_cond,
        )

    def load_pers(self):
        return self.load_branch(
            self.hparams.pers_lora,
            self.hparams.train_pers_lora,
            self.hparams.layout_cond and self.hparams.pers_layout_cond,
        )

    @abstractmethod
    def instantiate_model(self):
        pass

    @torch.no_grad()
    def encode_text(self, text):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device), attention_mask=attention_mask)

        return prompt_embeds[0].to(self.dtype)

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]

        x_input = rearrange(x_input.to(vae.dtype), 'b l c h w -> (b l) c h w')
        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()
        z = rearrange(z, '(b l) c h w -> b l c h w', b=b)

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        return z.to(self.dtype)

    def pad_pano(self, pano, latent=False):
        b, m = pano.shape[:2]
        padding = self.hparams.latent_pad
        if not latent:
            padding *= 8
        return pad_pano(pano, padding=padding)

    def unpad_pano(self, pano_pad, latent=False):
        padding = self.hparams.latent_pad
        if not latent:
            padding *= 8
        return unpad_pano(pano_pad, padding=padding)

    def gen_cls_free_guide_pair(self, *inputs):
        result = []
        for input in inputs:
            if input is None:
                result.append(None)
            elif isinstance(input, dict):
                result.append({k: torch.cat([v]*2) for k, v in input.items()})
            elif isinstance(input, list):
                result.append([torch.cat([v]*2) for v in input])
            else:
                result.append(torch.cat([input]*2))
        return result

    def combine_cls_free_guide_pred(self, *noise_pred_list):
        result = []
        for noise_pred in noise_pred_list:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.hparams.guidance_scale * \
                (noise_pred_text - noise_pred_uncond)
            result.append(noise_pred)
        if len(result) == 1:
            return result[0]
        return result

    def rotate_latent(self, pano_latent, degree=None):
        if degree is None:
            degree = self.hparams.rot_diff
        if degree % 360 == 0:
            return pano_latent
        return torch.roll(pano_latent, int(degree / 360 * pano_latent.shape[-1]), dims=-1)

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b = latents.shape[0]
        latents = (1 / vae.config.scaling_factor * latents)
        latents = rearrange(latents, 'b m c h w -> (b m) c h w')
        image = vae.decode(latents.to(vae.dtype)).sample
        image = rearrange(image, '(b m) c h w -> b m c h w', b=b)
        return image.to(self.dtype)

    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            param_groups.append({"params": params, "lr": self.hparams.lr * lr_scale})
        optimizer = torch.optim.AdamW(param_groups)
        if self.hparams.layout_cond:
            return optimizer
        else:
            scheduler = {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7),
                'interval': 'epoch',  # update the learning rate after each epoch
                'name': 'cosine_annealing_lr',
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        output_dir = os.path.join(self.logger.save_dir, 'test', batch['pano_id'][0])
        self.inference_and_save(batch, output_dir)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_dir = os.path.join(self.logger.save_dir, 'predict', batch['pano_id'][0])
        self.inference_and_save(batch, output_dir, 'jpg')

    @abstractmethod
    def inference_and_save(self, batch, output_dir, ext):
        pass
