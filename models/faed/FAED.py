import torch
from torch import nn
from torch import Tensor
from lightning.pytorch.utilities import rank_zero_only
from .modules import AutoEncoder
from ..modules.utils import WandbLightningModule
from torchmetrics import Metric
import math
from torchmetrics.image.fid import _compute_fid
import os


class FAED(WandbLightningModule):
    def __init__(
            self,
            lr: float = 1e-4,
            lr_decay: float = 0.99,
            ):
        super().__init__()
        self.save_hyperparameters()
        self.net = AutoEncoder()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_decay)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        pano_pred = self.net(batch['pano'].squeeze(1)).unsqueeze(1)
        loss = torch.nn.functional.l1_loss(pano_pred, batch['pano'])
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pano_pred = self.net(batch['pano'].squeeze(1)).unsqueeze(1)
        self.log_val_image(pano_pred, batch['pano'], batch['pano_id'])

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, pano_pred, pano, pano_id):
        log_dict = {
            'val/pano_pred': self.temp_wandb_image(
                pano_pred[0, 0], pano_id[0] if pano_id else None),
            'val/pano_gt': self.temp_wandb_image(
                pano[0, 0], pano_id[0] if pano_id else None),
        }
        self.logger.experiment.log(log_dict)


class FrechetAutoEncoderDistance(Metric):
    higher_is_better = False

    def __init__(self, pano_height: int):
        super().__init__()
        ckpt_path = os.path.join('weights', 'faed.ckpt')
        faed = FAED.load_from_checkpoint(ckpt_path)
        self.encoder = faed.net.encoder

        num_features = pano_height * 4
        mx_num_feets = (num_features, num_features)
        self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(mx_num_feets).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(mx_num_feets).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def get_activation(self, imgs):
        imgs = (imgs.type(torch.float32) / 127.5) - 1
        features = self.encoder(imgs)
        mean_feature = torch.mean(features, dim=3)
        weight = torch.cos(
            torch.linspace(math.pi / 2, -math.pi / 2, mean_feature.shape[-1], device=mean_feature.device)
            ).unsqueeze(0).unsqueeze(0).expand_as(mean_feature)
        mean_feature = weight * mean_feature
        mean_vector = mean_feature.view(-1, (mean_feature.shape[-2] * mean_feature.shape[-1]))
        return mean_vector

    def update(self, imgs: Tensor, real: bool):
        features = self.get_activation(imgs)
        features = features.double()
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)
