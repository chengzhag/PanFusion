import os
import argparse

import numpy as np
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image

from utils.layout import Layout


def parse_args():
    parser = argparse.ArgumentParser(description="Render Matterport3D 2D Layout")
    parser.add_argument("--path", default="data/Matterport3D/mp3d_skybox/",
                        help="dataset path")
    parser.add_argument("--scene", default=None,
                        help="scene id", type=str)
    parser.add_argument("--processes", default=16,
                        help="number of processes", type=int)
    parser.add_argument("--mp3d_anno_dir", type=str, default='data/Matterport3DLayoutAnnotation/label_data',
                        help="Path to Matterport3DLayoutAnnotation directory")
    return parser.parse_args()


def run(args):
    try:
        anno_path = os.path.join(args.mp3d_anno_dir, f"{args.scene}_label.json")
        layout = Layout.from_json(anno_path)
        scene_id, view_id = args.scene.split("_")
        layout_dir = os.path.join(args.path, scene_id, "layout", view_id)
        results = layout.render_layout(output_dir=layout_dir, output_prefix="layout_", size=(1024, 2048))
        distance_map = (results['distance_map'] * 1e3).astype(np.uint16)
        Image.fromarray(distance_map).save(os.path.join(layout_dir, "layout_distance_map.png"))
        return [True]
    except Exception as e:
        tqdm.write(f"scene {args.scene} failed: {e}")
        return [False]


def main():
    args = parse_args()

    if args.scene is not None:
        args.path = 'debug'
        run(args)
        return

    # scan all scenes
    anno_paths = glob(os.path.join(args.mp3d_anno_dir, "*.json"))
    scene_ids = [os.path.splitext(os.path.basename(anno_path))[0] for anno_path in anno_paths]
    scene_ids.sort()

    # create args for each scene
    args_list = []
    for scene_id in scene_ids:
        new_args = argparse.Namespace(**vars(args))
        new_args.scene = scene_id.removesuffix("_label")
        args_list.append(new_args)

    with Pool(args.processes) as p:
        r = list(tqdm(p.imap(run, args_list), total=len(args_list)))
        r = [item for sublist in r for item in sublist]
        print(f"total: {len(r)}, success: {sum(r)}")


if __name__ == '__main__':
    main()
