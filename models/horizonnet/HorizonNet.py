import torch
import torch.nn.functional as F
from ..modules.utils import WandbLightningModule
from .modules import HorizonNetModule
import numpy as np
from shapely.geometry import Polygon
import sys
from external.HorizonNet.misc import post_proc
from external.HorizonNet.inference import find_N_peaks
from external.HorizonNet.dataset import visualize_a_data
from utils.layout import Layout
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics import Metric
import os
import wandb


def horizon_to_manhattan_layout(horizon_layout, H, W, force_cuboid=True, min_v=None, r=0.05, normalize=False):
    y_bon_, y_cor_ = horizon_layout['bon'], horizon_layout['cor']

    y_bon_ = (y_bon_ / np.pi + 0.5) * H - 0.5
    y_bon_[0] = np.clip(y_bon_[0], 1, H/2-1)
    y_bon_[1] = np.clip(y_bon_[1], H/2+1, H-2)
    y_cor_ = y_cor_[0]

    # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    # Detech wall-wall peaks
    if min_v is None:
        min_v = 0 if force_cuboid else 0.05
    r = int(round(W * r / 2))
    N = 4 if force_cuboid else None
    xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

    # Generate wall-walls
    cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
    if not force_cuboid:
        # Check valid (for fear self-intersection)
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i - 1]['type']] = xy_cor[i - 1]['val']
        if not Polygon(xy2d).is_valid:
            print(
                'Fail to generate valid general layout!! '
                'Generate cuboid as fallback.',
                file=sys.stderr)
            xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
            cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor) * 2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j * 2] = cor[j, 0], cor[j, 1]
        cor_id[j * 2 + 1] = cor[j, 0], cor[j, 2]

    # Normalized to [0, 1]
    if normalize:
        cor_id[:, 0] /= W
        cor_id[:, 1] /= H

    return cor_id, z0, z1


class HorizonNet(WandbLightningModule):
    def __init__(
            self,
            lr: float = 1e-4,
            ckpt_path: str = None,
            vertical_fov: float = 85,
            log_test_samples: int = 50,
            ):
        super().__init__()
        self.save_hyperparameters()
        self.net = HorizonNetModule()
        self.layout_iou = LayoutIoU()
        self.layout_iou.requires_grad_(False)

        if ckpt_path is not None:
            print(f"Loading weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path)['state_dict']
            self.net.load_state_dict(state_dict, strict=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=True, threshold=0.01)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train/loss"}

    def mask_and_normalize(self, x):
        x = x.clone()
        x = x / 2 + 0.5
        height = x.shape[-2]
        fov_pix = int((self.hparams.vertical_fov / 180) * height)
        boarder = (height - fov_pix) // 2
        x[..., :boarder, :] = 0
        x[..., -boarder:, :] = 0
        return x

    def forward(self, x, normalize=False, return_intermediate=False):
        x = self.mask_and_normalize(x)
        bon, cor = self.net(x)
        manhattan_pix = self.get_manhattan_layout(bon, cor, *x.shape[-2:], normalize=normalize)
        if return_intermediate:
            return manhattan_pix, bon, cor, x
        return manhattan_pix

    def training_step(self, batch, batch_idx):
        pano = self.mask_and_normalize(batch['pano'].squeeze(1))
        bon, cor = self.net(pano)
        bon_loss = F.l1_loss(bon, batch['horizon_layout']['bon'])
        cor_loss = F.binary_cross_entropy_with_logits(cor, batch['horizon_layout']['cor'])
        loss = bon_loss + cor_loss
        self.log('train/loss', loss, prog_bar=False)
        self.log('train/bon_loss', bon_loss, prog_bar=True)
        self.log('train/cor_loss', cor_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        manhattan_pix, bon, cor, pano = self(batch['pano'].squeeze(1), return_intermediate=True)
        layout_raw = visualize_a_data(pano[0], bon[0], torch.sigmoid(cor[0]))
        layout_pos = manhattan_pix[0].reshape(-1, 2, 2)
        layout = Layout.from_layout_pos(layout_pos, 1.6, pano.shape[:-3:-1])
        image = (pano[0] * 255).round().cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        layout_man = layout.render_wireframe(background=image, color=(255, 255, 0))
        self.log_val_image(layout_raw, layout_man, batch['pano_layout_cond'], batch['pano_id'])

    def get_manhattan_layout(self, bon, cor, height, width, normalize=False):
        # transform pixel layout estimation to pixel manhattan world layout
        manhattan_pix = []
        for b, c in zip(bon.float().cpu().numpy(), cor.float().cpu().numpy()):
            try:
                dt_cor_id, z0, z1 = horizon_to_manhattan_layout(
                    {'bon': b, 'cor': c}, height, width, force_cuboid=False, normalize=normalize)
            except:
                dt_cor_id = np.array([
                    [k // 2 * 256, 256 - ((k % 2) * 2 - 1) * 120]
                    for k in range(8)
                ])
                if normalize:
                    dt_cor_id = dt_cor_id.astype(np.float32)
                    dt_cor_id[:, 0] /= width
                    dt_cor_id[:, 1] /= height
            manhattan_pix.append(dt_cor_id)
        return manhattan_pix

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, layout_raw, layout_man, layout_gt, pano_id):
        log_dict = {
            'val/layout_raw': self.temp_wandb_image(
                layout_raw, pano_id[0] if pano_id else None),
            'val/layout_man': self.temp_wandb_image(
                layout_man, pano_id[0] if pano_id else None),
            'val/layout_gt': self.temp_wandb_image(
                layout_gt[0, 0], pano_id[0] if pano_id else None),
        }
        self.logger.experiment.log(log_dict)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        manhattan_pix, bon, cor, pano = self(batch['pano'].squeeze(1), return_intermediate=True)

        if batch_idx < self.hparams.log_test_samples:
            layout_raw = visualize_a_data(pano[0], bon[0], torch.sigmoid(cor[0]))
            layout_pos = manhattan_pix[0].reshape(-1, 2, 2)
            layout = Layout.from_layout_pos(layout_pos, 1.6, pano.shape[:-3:-1])
            image = (pano[0] * 255).round().cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            layout_man = layout.render_wireframe(background=image, color=(255, 255, 0))
            pano_id = batch['pano_id'][0]
            test_sample_row = {
                'pano_id': pano_id,
                'val/layout_raw': self.temp_wandb_image(layout_raw, pano_id),
                'val/layout_man': self.temp_wandb_image(layout_man, pano_id),
                'val/layout_gt': self.temp_wandb_image(batch['pano_layout_cond'][0, 0], pano_id),
            }

            if batch_idx == 0:
                self.test_sample_table = wandb.Table(columns=list(test_sample_row.keys()))
            self.test_sample_table.add_data(*test_sample_row.values())

        self.layout_iou.update(manhattan_pix, batch['manhattan_layout'])

    @torch.no_grad()
    @rank_zero_only
    def on_test_end(self):
        iou3d, iou2d = self.layout_iou.compute()
        test_metrics = {'3D_IoU': iou3d, '2D_IoU': iou2d}
        wandb.summary.update(test_metrics)
        metrics_table = wandb.Table(columns=list(test_metrics.keys()), data=[list(test_metrics.values())])
        self.logger.experiment.log({'test/metrics': metrics_table})
        self.logger.experiment.log({'test/sample': self.test_sample_table})


class LayoutIoU(Metric):
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state("iou3d_value", default=torch.tensor(0.0).double(), dist_reduce_fx="sum")
        self.add_state("iou2d_value", default=torch.tensor(0.0).double(), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0).long(), dist_reduce_fx="sum")

    def eval_iou(self, dt_cor_id, gt_cor_id):
        dt_floor_coor = dt_cor_id[1::2]
        dt_ceil_coor = dt_cor_id[0::2]
        gt_floor_coor = gt_cor_id[1::2]
        gt_ceil_coor = gt_cor_id[0::2]
        assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
        assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0

        # Eval 3d IoU and height error(in meter)
        ch = -1.6
        dt_floor_xy = post_proc.np_coor2xy(dt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
        gt_floor_xy = post_proc.np_coor2xy(gt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
        dt_poly = Polygon(dt_floor_xy)
        gt_poly = Polygon(gt_floor_xy)

        # 2D IoU
        try:
            area_dt = dt_poly.area
            area_gt = gt_poly.area
            area_inter = dt_poly.intersection(gt_poly).area
            iou2d = area_inter / (area_gt + area_dt - area_inter)
        except:
            iou2d = 0

        # 3D IoU
        try:
            cch_dt = post_proc.get_z1(dt_floor_coor[:, 1], dt_ceil_coor[:, 1], ch, 512)
            cch_gt = post_proc.get_z1(gt_floor_coor[:, 1], gt_ceil_coor[:, 1], ch, 512)
            h_dt = abs(cch_dt.mean() - ch)
            h_gt = abs(cch_gt.mean() - ch)
            area3d_inter = area_inter * min(h_dt, h_gt)
            area3d_pred = area_dt * h_dt
            area3d_gt = area_gt * h_gt
            iou3d = area3d_inter / (area3d_pred + area3d_gt - area3d_inter)
        except:
            iou3d = 0

        return iou3d, iou2d

    def update(self, layout_pred, layout_gt):
        for dt_cor_id, gt_cor_id in zip(layout_pred, layout_gt):
            if isinstance(dt_cor_id, torch.Tensor):
                dt_cor_id = dt_cor_id.cpu().numpy()
            if isinstance(gt_cor_id, torch.Tensor):
                gt_cor_id = gt_cor_id.cpu().numpy()
            iou3d, iou2d = self.eval_iou(dt_cor_id, gt_cor_id)
        self.iou3d_value += iou3d
        self.iou2d_value += iou2d
        self.num_samples += 1

    def compute(self):
        iou3d = self.iou3d_value / self.num_samples
        iou2d = self.iou2d_value / self.num_samples
        return iou3d, iou2d


class LayoutConsistency(LayoutIoU):
    def __init__(self):
        super().__init__()
        ckpt_path = os.path.join('weights', 'horizonnet.ckpt')
        self.horizonnet = HorizonNet.load_from_checkpoint(ckpt_path, ckpt_path=None)

    def update(self, imgs, layout_gt):
        imgs = imgs * 2 - 1
        layout_pred = self.horizonnet(imgs)
        super().update(layout_pred, layout_gt)
