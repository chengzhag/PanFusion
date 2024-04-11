import numpy as np
from PIL import Image
import argparse
from .e2p import e2p
from .p2e import p2e
import torch
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Test Panorama Projection")
    parser.add_argument("--panorama_dir", type=str, required=True, help="Path to panorama image")
    return parser.parse_args()


def main():
    args = parse_args()
    pano = np.array(Image.open(args.panorama_dir))

    # test numpy
    pers = e2p(pano, 90, 0, 0, (512, 512))
    Image.fromarray(pers).save('debug/pers.png')
    equi = p2e(pers, 90, 0, 0, (512, 1024))[0]
    Image.fromarray(equi).save('debug/equi.png')

    # test torch
    pano = torch.from_numpy(pano).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    pano = pano.repeat(4, 1, 1, 1)
    pers = e2p(pano, 90, (0, 90, 180, 270), 0, (512, 512))
    os.makedirs('debug', exist_ok=True)
    for i, p in enumerate(pers):
        Image.fromarray((p.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(f'debug/pers_{i}.png')

    equi = p2e(pers, 90, (0, 90, 180, 270), 0, (512, 1024))[0]
    for i, e in enumerate(equi):
        Image.fromarray((e.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(f'debug/equi_{i}.png')

if __name__ == '__main__':
    main()
