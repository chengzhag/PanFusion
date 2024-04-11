import cv2
import numpy as np
from kornia.geometry.transform import remap
from torch import Tensor
import torch
from .utils import choose_mode, index_list_or_scalar


def map_equi_pix_to_pers(ph, pw, wfov, theta, phi, h, w):
    hfov = float(ph) / pw * wfov

    w_len = np.tan(np.radians(wfov / 2.0))
    h_len = np.tan(np.radians(hfov / 2.0))

    x, y = np.meshgrid(np.linspace(-180, 180, w), np.linspace(90, -90, h))

    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map, y_map, z_map), axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(theta))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    xyz = xyz.reshape([h * w, 3]).T
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T

    xyz = xyz.reshape([h, w, 3])
    inverse_mask = np.where(xyz[:, :, 0] > 0, 1, 0)

    xyz[:, :] = xyz[:, :] / \
        np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

    lon_map = np.where((-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) & (-h_len < xyz[:, :, 2])
                        & (xyz[:, :, 2] < h_len), (xyz[:, :, 1]+w_len)/2/w_len*pw, 0)
    lat_map = np.where((-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) & (-h_len < xyz[:, :, 2])
                        & (xyz[:, :, 2] < h_len), (-xyz[:, :, 2]+h_len)/2/h_len*ph, 0)
    mask = np.where((-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) & (-h_len < xyz[:, :, 2])
                    & (xyz[:, :, 2] < h_len), 1, 0)
    mask = (mask * inverse_mask) > 0

    return lon_map, lat_map, mask


def p2e(p_img, fov_deg, u_deg, v_deg, out_hw, mode=None):
    mode = choose_mode(p_img, mode)
    if isinstance(p_img, Tensor):
        b, c, hp, wp = p_img.shape
        lons, lats, masks = [], [], []
        if all([not hasattr(v, '__len__') for v in [fov_deg, u_deg, v_deg]]):
            b = 1  # if all scalars, then use broadcasting
        for i in range(b):
            fov = index_list_or_scalar(fov_deg, i)
            u = index_list_or_scalar(u_deg, i)
            v = index_list_or_scalar(v_deg, i)
            lon, lat, mask = map_equi_pix_to_pers(hp, wp, fov, u, v, out_hw[0], out_hw[1])
            lons.append(lon)
            lats.append(lat)
            masks.append(mask[np.newaxis, :, :])
        lons = torch.from_numpy(np.stack(lons)).to(p_img.device).type(p_img.dtype)
        lats = torch.from_numpy(np.stack(lats)).to(p_img.device).type(p_img.dtype)
        mask = torch.from_numpy(np.stack(masks)).to(p_img.device)
        equi = remap(p_img, lons, lats, align_corners=True, mode=mode)
        equi = equi * mask
    else:
        hp, wp = p_img.shape[:2]
        lon, lat, mask = map_equi_pix_to_pers(hp, wp, fov_deg, u_deg, v_deg, out_hw[0], out_hw[1])
        equi = cv2.remap(p_img, lon.astype(np.float32), lat.astype(np.float32), mode, borderMode=cv2.BORDER_WRAP)
        equi = equi * mask[:, :, np.newaxis]
    return equi, mask
