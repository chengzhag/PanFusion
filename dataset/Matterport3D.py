import os
import numpy as np
from glob import glob
from .PanoDataset import PanoDataset, PanoDataModule
from utils.layout import Layout


class Mp3dDataset(PanoDataset):
    def load_split(self, mode):
        if self.config['load_layout']:
            with open(os.path.join(self.data_dir, f"lo_{mode}.txt"), 'r') as f:
                data = f.read().splitlines()
            new_data = []
            for d in data:
                scene_id, view_id = d.split('_')
                new_data.append({
                    'scene_id': scene_id,
                    'view_id': view_id,
                })
        else:
            split_file = 'train.npy' if mode == 'train' else 'test.npy'
            split_dir = os.path.join(self.data_dir, split_file)
            if os.path.exists(split_dir):
                data = np.load(split_dir)
                new_data = []
                for d in data:
                    scene_id, _, view_id = d[0].split('/')
                    view_id = view_id.split('_')[0]
                    new_data.append({
                        'scene_id': scene_id,
                        'view_id': view_id,
                    })
            elif mode == 'predict':
                new_data = []
                print(f"Scanning {self.data_dir}...")
                prompts = glob(os.path.join(self.data_dir, '*', 'blip3_stitched', '*.txt'))
                for d in prompts:
                    scene_id, _, view_id = d.split('/')[-3:]
                    view_id = view_id.split('.')[0]
                    new_data.append({
                        'scene_id': scene_id,
                        'view_id': view_id,
                    })
            else:
                raise FileNotFoundError(f"Cannot find split file: {split_dir}")
        return new_data

    def scan_results(self, result_dir):
        results = glob(os.path.join(result_dir, '*/'))
        results = [tuple(r.split('/')[-2].split('_')) for r in results]
        return results

    def get_data(self, idx):
        data = self.data[idx].copy()
        scene_id, view_id = data['scene_id'], data['view_id']
        if self.mode == 'predict' and self.config['repeat_predict'] > 1:
            data['pano_id'] = f"{scene_id}_{view_id}_{data['repeat_id']:06d}"
        else:
            data['pano_id'] = f"{scene_id}_{view_id}"

        if self.mode != 'predict':
            if self.config['load_layout']:
                pano_folder = 'matterport_aligned_images'
            else:
                pano_folder = 'matterport_stitched_images'
            data['pano_path'] = os.path.join(self.data_dir, scene_id, pano_folder, f"{view_id}.png")

        if self.config['layout_cond_type']:
            data['layout_cond_path'] = os.path.join(
                self.data_dir, scene_id, 'layout', view_id, f"layout_{self.config['layout_cond_type']}.png")

        if self.mode != 'predict':
            prompt = []
            for i in range(8):
                degree = i * 45
                prompt_path = os.path.join(self.data_dir, scene_id, 'blip3', f"{view_id}_{degree}.txt")
                prompt.append(self.load_prompt(prompt_path))
            data['prompt'] = prompt

        data['pano_prompt_path'] = os.path.join(self.data_dir, scene_id, 'blip3_stitched', f"{view_id}.txt")

        if self.config['horizon_layout'] or self.config['manhattan_layout']:
            json_dir = os.path.join(self.config['layout_anno_dir'], 'label_data', f"{scene_id}_{view_id}_label.json")
            layout = Layout.from_json(json_dir)
            data['layout'] = layout

        if self.result_dir is not None:
            data['pano_pred_path'] = os.path.join(self.result_dir, data['pano_id'], 'pano.png')
        return data


class Matterport3D(PanoDataModule):
    def __init__(
            self,
            data_dir: str = 'data/Matterport3D/mp3d_skybox',
            layout_anno_dir: str = 'data/Matterport3DLayoutAnnotation',
            *args,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.dataset_cls = Mp3dDataset
        self.hparams.load_layout = self.hparams.layout_cond_type or self.hparams.horizon_layout or self.hparams.manhattan_layout

    def scan_layout_cond(self):
        splits = ['train', 'val', 'test']
        for split in splits:
            src_split_file = os.path.join(self.hparams.layout_anno_dir, 'data_list', f"mp3d_{split}.txt")
            dst_split_file = os.path.join(self.hparams.data_dir, f"lo_{split}.txt")
            if os.path.exists(dst_split_file):
                continue

            print(f"Scanning {split} split...")
            with open(src_split_file, 'r') as f:
                data = f.read().splitlines()

            new_data = []
            for d in data:
                scene_id, view_id = d.split(' ')
                layout_cond_path = os.path.join(self.hparams.data_dir, scene_id, 'layout', view_id, f"layout_{self.hparams.layout_cond_type}.png")
                pano_path = os.path.join(self.hparams.data_dir, scene_id, 'matterport_aligned_images', f"{view_id}.png")
                if os.path.exists(layout_cond_path) and os.path.exists(pano_path):
                    new_data.append('_'.join([scene_id, view_id]))
            print(f"Found {len(new_data)}/{len(data)} valid samples in {split} split.")

            with open(dst_split_file, 'w') as f:
                f.write('\n'.join(new_data))

    def prepare_data(self):
        if os.path.isdir(self.hparams.data_dir) and self.hparams.load_layout:
            self.scan_layout_cond()
