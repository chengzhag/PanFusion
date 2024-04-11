import os
import argparse
from utils.pano import Cubemap
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool


def stitch_mp3d_skybox(args):
    cubemap = Cubemap.from_mp3d_skybox(args.mp3d_skybox_path, args.scene, args.view)
    equirectangular = cubemap.to_equirectangular(1024, 2048)
    equirectangular.save(os.path.join(args.mp3d_skybox_path, args.scene, 'matterport_stitched_images', f"{args.view}.png"))


def parse_args():
    parser = argparse.ArgumentParser(description="Stitch Matterport3D Skybox")
    parser.add_argument("--mp3d_skybox_path", type=str, default='data/Matterport3D/mp3d_skybox',
                        help="Matterport3D mp3d_skybox path")
    parser.add_argument("--processes", type=int, default=16,
                        help="Number of processes to use")
    parser.add_argument("--scene", default=None,
                        help="scene id", type=str)
    parser.add_argument("--view", default=None,
                        help="view id", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    # Scan all views
    if args.scene is not None and args.view is not None:
        args_list = [args]
    else:
        scenes = [x for x in os.listdir(args.mp3d_skybox_path) if os.path.isdir(os.path.join(args.mp3d_skybox_path, x))]
        args_list = []
        for scene in tqdm(scenes, desc="Scanning scenes"):
            views = [os.path.basename(x).split('_')[0] for x in glob(
                os.path.join(args.mp3d_skybox_path, scene, 'matterport_skybox_images', '*.jpg'))]
            for view in set(views):
                new_args = argparse.Namespace(**vars(args))
                new_args.scene = scene
                new_args.view = view
                args_list.append(new_args)

    # Stitch all views
    if args.processes == 0:
        for args in tqdm(args_list, desc="Stitching"):
            stitch_mp3d_skybox(args)
    else:
        with Pool(args.processes) as p:
            list(tqdm(p.imap_unordered(stitch_mp3d_skybox, args_list), total=len(args_list), desc="Stitching"))


if __name__ == '__main__':
    main()
