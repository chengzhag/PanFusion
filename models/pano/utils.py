import torch
from einops import rearrange
from external.Perspective_and_Equirectangular import e2p, p2e, map_pers_coords_to_equi
from kornia.utils import create_meshgrid
from kornia.filters import gaussian_blur2d
import numpy as np
from utils.pano import pad_pano, unpad_pano


def get_masks(pers_h, pers_w, equi_h, equi_w, cameras, device, dtype=torch.float32):
    # pers_masks: attention mask in perspective images of a pixel in equirectangular image
    # (m, equi_h, equi_w, pers_h, pers_w)
    # equi_masks: attention mask in equirectangular images of a pixel in perspective image
    # (m, pers_h, pers_w, equi_h, equi_w)
    # pers_masks = torch.zeros((m, equi_h, equi_w, pers_h, pers_w), device=device, dtype=torch.bool)
    # equi_masks = torch.zeros((m, pers_h, pers_w, equi_h, equi_w), device=device, dtype=torch.bool)
    m = len(cameras['FoV'])

    # *_pixels iterates over all pixels in the image
    pers_pixels = torch.zeros((m, pers_h, pers_w, pers_h, pers_w), device=device, dtype=dtype)
    equi_pixels = torch.zeros((m, equi_h, equi_w, equi_h, equi_w), device=device, dtype=dtype)
    pers_grid = create_meshgrid(pers_h, pers_w, normalized_coordinates=False, device=device, dtype=torch.long)
    equi_grid = create_meshgrid(equi_h, equi_w, normalized_coordinates=False, device=device, dtype=torch.long)
    pers_indices = rearrange(pers_grid, '1 h w c -> c (h w)')
    equi_indices = rearrange(equi_grid, '1 h w c -> c (h w)')
    # pers_indices = pers_grid.reshape(-1, 2)
    # equi_indices = equi_grid.reshape(-1, 2)
    pers_pixels[:, pers_indices[1], pers_indices[0], pers_indices[1], pers_indices[0]] = 1.0
    equi_pixels[:, equi_indices[1], equi_indices[0], equi_indices[1], equi_indices[0]] = 1.0

    # warp pixel masks
    pers_pixels = rearrange(pers_pixels, 'm h w hh ww -> m (h w) hh ww')
    equi_pixels = rearrange(equi_pixels, 'm h w hh ww -> m (h w) hh ww')
    equi_masks = p2e(
        pers_pixels,
        cameras['FoV'], cameras['theta'], cameras['phi'],
        (equi_h, equi_w))[0]
    pers_masks = e2p(
        equi_pixels,
        cameras['FoV'], cameras['theta'], cameras['phi'],
        (pers_h, pers_w))
    # *_masks represents the correspondence pixels in * image
    pers_masks = rearrange(pers_masks, 'm (h w) hh ww -> m h w hh ww', h=equi_h, w=equi_w)
    equi_masks = rearrange(equi_masks, 'm (h w) hh ww -> m h w hh ww', h=pers_h, w=pers_w)

    # fix missing pixels in masks resulted from interpolation
    # pers_masks1 = pers_masks.clone()
    # for equi_idx in equi_indices:
    #     # for each pixel in equirectangular image, find the corresponding pixels in perspective images
    #     matches = equi_masks[:, :, :, equi_idx[1], equi_idx[0]]
    #     pers_masks1[:, equi_idx[1], equi_idx[0], :, :] |= matches
    pers_masks_pixels = equi_masks[:, :, :, equi_indices[1], equi_indices[0]]
    pers_masks_pixels = rearrange(pers_masks_pixels, 'm h w l -> m l h w')
    pers_masks[:, equi_indices[1], equi_indices[0], :, :] += pers_masks_pixels
    pers_masks = torch.clamp(pers_masks, 0, 1)
    # print((pers_masks1 != pers_masks).sum())
    equi_masks_pixels = pers_masks[:, :, :, pers_indices[1], pers_indices[0]]
    equi_masks_pixels = rearrange(equi_masks_pixels, 'm h w l -> m l h w')
    equi_masks[:, pers_indices[1], pers_indices[0], :, :] += equi_masks_pixels
    equi_masks = torch.clamp(equi_masks, 0, 1)

    # gaussian blur
    pers_masks = rearrange(pers_masks, 'm h w hh ww -> (m h w) 1 hh ww')
    equi_masks = rearrange(equi_masks, 'm h w hh ww -> (m h w) 1 hh ww')
    pers_masks = gaussian_blur2d(pers_masks, (5, 5), (1.0, 1.0), border_type='replicate')
    equi_masks = pad_pano(equi_masks, 2)
    equi_masks = gaussian_blur2d(equi_masks, (5, 5), (1.0, 1.0), border_type='replicate')
    equi_masks = unpad_pano(equi_masks, 2)
    pers_masks_max = torch.amax(pers_masks, dim=(1, 2, 3), keepdim=True)
    pers_masks_max[pers_masks_max == 0] = 1.0
    pers_masks = pers_masks / pers_masks_max
    pers_masks = pers_masks * 2 - 1
    equi_masks_max = torch.amax(equi_masks, dim=(1, 2, 3), keepdim=True)
    equi_masks_max[equi_masks_max == 0] = 1.0
    equi_masks = equi_masks / equi_masks_max
    equi_masks = equi_masks * 2 - 1
    pers_masks = rearrange(pers_masks, '(m h w) 1 hh ww -> m h w hh ww', h=equi_h, w=equi_w)
    equi_masks = rearrange(equi_masks, '(m h w) 1 hh ww -> m h w hh ww', h=pers_h, w=pers_w)
    # for i in range(8):
    #     sample_mask = (equi_masks[19, 7, i] / 2 + 0.5).cpu().numpy()
    #     from PIL import Image
    #     Image.fromarray((sample_mask * 255).astype('uint8')).save(f'debug/sample_mask-{i}.png')

    return pers_masks, equi_masks


def get_coords(pers_h, pers_w, equi_h, equi_w, cameras, device, dtype=torch.float32):
    # pers_coords: polar coordinates of a pixel in perspective images
    # (m, pers_h, pers_w, 2)
    # equi_coords: polar coordinates of a pixel in equirectangular images
    # (equi_h, equi_w, 2)
    x, y = np.meshgrid(np.linspace(-np.pi, np.pi, equi_w), np.linspace(np.pi/2, -np.pi/2, equi_h))
    xy = np.stack([x, y])
    equi_coords = torch.tensor(xy, device=device, dtype=dtype)
    equi_coords = rearrange(equi_coords, 'c h w -> h w c')

    pers_coords = []
    for fov, theta, phi in zip(cameras['FoV'], cameras['theta'], cameras['phi']):
        fov, theta, phi = fov.item(), theta.item(), phi.item()
        lon, lat = map_pers_coords_to_equi(fov, theta, phi, pers_h, pers_w)
        lonlat = np.stack([lon, lat])
        pers_coords.append(torch.tensor(lonlat, device=device, dtype=dtype))
    pers_coords = torch.stack(pers_coords, dim=0)
    pers_coords = rearrange(pers_coords, 'm c h w -> m h w c')

    return pers_coords, equi_coords
