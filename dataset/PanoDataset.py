import torch
import os
import numpy as np
import random
from collections import defaultdict
from utils.pano import Equirectangular, random_sample_camera, horizon_sample_camera, icosahedron_sample_camera
import lightning as L
import cv2
from glob import glob
from einops import rearrange
from abc import abstractmethod
from PIL import Image
from external.Perspective_and_Equirectangular import mp2e


def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R


class PanoDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.data_dir = config['data_dir']
        self.result_dir = config.get('result_dir', None)
        self.config = config

        self.data = self.load_split(mode)
        if mode == 'predict':
            self.data = sum([[d.copy() for i in range(self.config['repeat_predict'])] for d in self.data], [])
            if self.config['repeat_predict'] > 1:
                for i, d in enumerate(self.data):
                    d['repeat_id'] = i % self.config['repeat_predict']

        if not self.config['gt_as_result'] and self.result_dir is not None:
            results = self.scan_results(self.result_dir)
            assert results, f"No results found in {self.result_dir}, forgot to set environment variable WANDB_RUN_ID?"
            new_data = set(self.data) & set(results)
            if len(new_data) != len(self.data):
                print(f"WARNING: {len(self.data)-len(new_data)} views are missing in results folder {self.result_dir} for {self.mode} set.")
                self.data = list(new_data)
                self.data.sort()

    @abstractmethod
    def load_split(self):
        pass

    @abstractmethod
    def scan_results(self):
        pass

    def __len__(self):
        return len(self.data)

    def load_prompt(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        return prompt

    @abstractmethod
    def get_data(self, idx):
        pass

    def __getitem__(self, idx):
        data = self.get_data(idx)

        # load layout
        if 'layout' in data:
            layout = data['layout']
            image_size = (self.config['pano_height'] * 2, self.config['pano_height'])
            if self.config['manhattan_layout']:
                data['manhattan_layout'] = layout.to_layout_pos(image_size).reshape(-1, 2)
            if self.config['horizon_layout']:
                data['horizon_layout'] = layout.to_horizonnet(image_size)
            del data['layout']

        # generate camera poses
        if self.config['cam_sampler'] == 'horizon':
            theta, phi = horizon_sample_camera(8)
            if self.mode == 'train':
                cam_rot = random.random() * 360
                theta = (theta + cam_rot) % 360
                if 'prompt' in data:
                    shift_idx = round(cam_rot / 45)
                    data['prompt'] = data['prompt'][shift_idx:] + data['prompt'][:shift_idx]
        elif self.config['cam_sampler'] == 'icosahedron':
            if self.mode == 'train':
                theta, phi = random_sample_camera(20)
            else:
                theta, phi = icosahedron_sample_camera()
        else:
            raise NotImplementedError
        theta, phi = np.rad2deg(theta), np.rad2deg(phi)

        Ks, Rs = [], []
        for t, p in zip(theta, phi):
            K, R = get_K_R(self.config['fov'], t, p,
                           self.config['pers_resolution'], self.config['pers_resolution'])
            Ks.append(K)
            Rs.append(R)
        K = np.stack(Ks).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)

        cameras = {
            'height': np.full_like(theta, self.config['pers_resolution'], dtype=int),
            'width': np.full_like(theta, self.config['pers_resolution'], dtype=int),
            'FoV': np.full_like(theta, self.config['fov'], dtype=int),
            'theta': theta,
            'phi': phi,
            'R': R,
            'K': K,
        }
        data['cameras'] = cameras
        data['height'] = self.config['pano_height']
        data['width'] = self.config['pano_height'] * 2

        rotation = random.random() * 360 if self.mode == 'train' and self.config['rand_rot_img'] else 0
        flip = self.config['rand_flip'] and self.mode == 'train' and random.random() < 0.5

        def process_equi(equi, normalize):
            equi.rotate(rotation)
            equi.flip(flip)
            imgs = []
            for t, p in zip(theta, phi):
                img = equi.to_perspective((self.config['fov'], self.config['fov']), t, p, (self.config['pers_resolution'], self.config['pers_resolution']))
                imgs.append(img)
            pano = cv2.resize(equi.equirectangular, (data['width'], data['height']), interpolation=cv2.INTER_AREA)
            pano = pano.reshape(data['height'], data['width'], equi.equirectangular.shape[-1])
            images = np.stack(imgs)
            if self.result_dir is None and normalize:
                images = (images.astype(np.float32)/127.5)-1
                pano = (pano.astype(np.float32)/127.5 - 1)
            pano = rearrange(pano, 'h w c -> 1 c h w')
            images = rearrange(images, 'b h w c -> b c h w')
            return pano, images

        # load images
        if 'pano_path' in data:
            equirectangular = Equirectangular.from_file(data['pano_path'])
            data['pano'], data['images'] = process_equi(equirectangular, True)

        if 'layout_cond_path' in data:
            equirectangular = Equirectangular.from_file(data['layout_cond_path'])
            if self.config['layout_cond_type'] == 'distance_map':
                equirectangular.equirectangular = equirectangular.equirectangular.astype(np.float32)[..., None] / 1e3
                depth_min = equirectangular.equirectangular.min()
                depth_max = equirectangular.equirectangular.max()
                equirectangular.equirectangular = 2.0 * (equirectangular.equirectangular - depth_min) / (depth_max - depth_min) - 1.0
                equirectangular.equirectangular = np.repeat(equirectangular.equirectangular, 3, axis=-1)
                normalize = False
            else:
                normalize = True
            if equirectangular.equirectangular.ndim == 2:
                equirectangular.equirectangular = np.repeat(equirectangular.equirectangular[..., None], 3, axis=-1)
            data['pano_layout_cond'], data['images_layout_cond'] = process_equi(equirectangular, normalize)

        if 'horizon_layout' in data:
            rot_pix = int(rotation / 360 * data['width'])
            data['horizon_layout'] = {k: np.roll(v, rot_pix, -1) for k, v in data['horizon_layout'].items()}
            if flip:
                data['horizon_layout'] = {k: np.flip(v, -1).copy() for k, v in data['horizon_layout'].items()}

        # load pano prompt
        if 'pano_prompt' not in data:
            data['pano_prompt'] = self.load_prompt(data['pano_prompt_path'])

        # unconditioned training
        if self.mode == 'train' and self.result_dir is None and random.random() < self.config['uncond_ratio']:
            data['pano_prompt'] = ''
            if 'prompt' in data:
                data['prompt'] = [''] * len(data['prompt'])

        # load results
        if self.config['gt_as_result']:
            data['pano_pred'] = data['pano']
            data['images_pred'] = data['images']
        elif self.result_dir is not None:
            images_pred = []
            for i in range(len(data['images'])):
                image_pred_path = os.path.join(os.path.dirname(data['pano_pred_path']), f"{i}.png")
                if not os.path.exists(image_pred_path):
                    break
                image_pred = Image.open(image_pred_path).convert('RGB')
                image_pred = np.array(image_pred)
                image_pred = cv2.resize(image_pred, (self.config['pers_resolution'], self.config['pers_resolution']))
                images_pred.append(image_pred)
            if images_pred:
                images_pred = np.stack(images_pred)
                data['images_pred'] = rearrange(images_pred, 'b h w c -> b c h w')

            if os.path.exists(data['pano_pred_path']):
                equirectangular = Equirectangular.from_file(data['pano_pred_path'])
                pano = cv2.resize(equirectangular.equirectangular, (data['width'], data['height']))
                data['pano_pred'] = rearrange(pano, 'h w c -> 1 c h w')
            elif 'images_pred' in data:
                # merge images for MVDiffusion results
                pano = mp2e(
                    images_pred, cameras['FoV'], cameras['theta'], cameras['phi'],
                    (data['height'], data['width']))
                data['pano_pred'] = rearrange(pano, 'h w c -> 1 c h w')

        return data


class PanoDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = None,
            fov: int = 90,
            cam_sampler: str = 'icosahedron',  # 'horizon', 'icosahedron'
            pers_resolution: int = 256,
            pano_height: int = 512,
            uncond_ratio: float = 0.2,
            batch_size: int = 1,
            num_workers: int = 8,
            result_dir: str = None,
            rand_rot_img: bool = False,
            rand_flip: bool = True,
            gt_as_result: bool = False,
            horizon_layout: bool = False,
            manhattan_layout: bool = False,
            layout_cond_type: str = None,
            repeat_predict: int = 10,
            ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_cls = PanoDataset

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.train_dataset = self.dataset_cls(self.hparams, mode='train')

        if stage in ('fit', 'validate', None):
            self.val_dataset = self.dataset_cls(self.hparams, mode='val')

        if stage in ('test', None):
            self.test_dataset = self.dataset_cls(self.hparams, mode='test')

        if stage in ('predict', None):
            self.predict_dataset = self.dataset_cls(self.hparams, mode='predict')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size,
            shuffle=True, num_workers=self.hparams.num_workers, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size,
            shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size,
            shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.hparams.batch_size,
            shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
