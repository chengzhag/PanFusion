import torch
from einops import rearrange
import lightning as L
import tempfile
from PIL import Image
import wandb


def tensor_to_image(image):
    if image.dtype != torch.uint8:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image * 255).round()
    image = image.cpu().numpy().astype('uint8')
    image = rearrange(image, '... c h w -> ... h w c')
    return image


class WandbLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.TemporaryDirectory()

    def temp_wandb_image(self, image, prompt=None):
        if isinstance(image, torch.Tensor):
            image = tensor_to_image(image)
        img_path = tempfile.NamedTemporaryFile(
            dir=self.temp_dir.name, suffix=".jpg", delete=False).name
        Image.fromarray(image.squeeze()).save(img_path)
        return wandb.Image(img_path, caption=prompt if prompt else None)

    def __del__(self):
        self.temp_dir.cleanup()
