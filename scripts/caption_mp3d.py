from PIL import Image
import os
import argparse
from lavis.models import load_model_and_preprocess
from tqdm import tqdm
import torch
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Caption Matterport3D Images")
    parser.add_argument("--mp3d_skybox_path", default="data/Matterport3D/mp3d_skybox",
                        help="Matterport3D mp3d_skybox path", type=str)
    parser.add_argument("--scene", default=None,
                        help="scene id", type=str)
    parser.add_argument("--view", default=None,
                        help="view id", type=str)
    parser.add_argument
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
            views = [os.path.basename(x).split('.')[0] for x in glob(
                os.path.join(args.mp3d_skybox_path, scene, 'matterport_stitched_images', '*.png'))]
            for view in set(views):
                new_args = argparse.Namespace(**vars(args))
                new_args.scene = scene
                new_args.view = view
                args_list.append(new_args)

    # Caption all views
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
    for args in tqdm(args_list, desc="Captioning"):
        caption_folder = os.path.join(args.mp3d_skybox_path, args.scene, 'blip3_stitched')
        caption_path = os.path.join(caption_folder, f"{args.view}.txt")
        if os.path.exists(caption_path):
            tqdm.write(f"Skipping {args.scene} {args.view}")
            continue

        raw_image = Image.open(os.path.join(args.mp3d_skybox_path, args.scene, 'matterport_stitched_images', f"{args.view}.png"))
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption = model.generate({"image": image, "prompt": "a 360 - degree view of"})

        os.makedirs(caption_folder, exist_ok=True)
        with open(caption_path, 'w') as f:
            f.write(caption[0])


if __name__ == '__main__':
    main()
