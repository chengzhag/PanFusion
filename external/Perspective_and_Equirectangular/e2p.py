import cv2
import numpy as np
import torch
from kornia.geometry.transform import remap
from torch import Tensor
from .utils import choose_mode, index_list_or_scalar


def map_pers_coords_to_equi(wfov, theta, phi, h, w):
    hfov = float(h) / w * wfov

    w_len = np.tan(np.radians(wfov / 2.0))
    h_len = np.tan(np.radians(hfov / 2.0))

    x_map = np.ones([h, w], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len, w), [h, 1])
    z_map = -np.tile(np.linspace(-h_len, h_len, h), [w, 1]).T

    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.stack((x_map, y_map, z_map), axis=2) / \
        np.repeat(D[:, :, np.newaxis], 3, axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(theta))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi))

    xyz = xyz.reshape([h * w, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2])
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])

    lon = lon.reshape([h, w])
    lat = -lat.reshape([h, w])
    return lon, lat


def map_pers_pix_to_equi(eh, ew, fov, theta, phi, h, w):
    lon, lat = map_pers_coords_to_equi(fov, theta, phi, h, w)

    equ_cx = (ew - 1) / 2.0
    equ_cy = (eh - 1) / 2.0

    lon = lon / np.pi * 180
    lat = lat / np.pi * 180

    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    return lon, lat


def e2p(e_img, fov_deg, u_deg, v_deg, out_hw, mode=None):
    '''
    e_img:   ndarray in shape of [H, W, *]
    fov_deg: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''
    mode = choose_mode(e_img, mode)
    if isinstance(e_img, Tensor):
        b, c, he, we = e_img.shape
        lons, lats = [], []
        if all([not hasattr(v, '__len__') for v in [fov_deg, u_deg, v_deg]]):
            b = 1  # if all scalars, then use broadcasting
        for i in range(b):
            fov = index_list_or_scalar(fov_deg, i)
            u = index_list_or_scalar(u_deg, i)
            v = index_list_or_scalar(v_deg, i)
            lon, lat = map_pers_pix_to_equi(he, we, fov, u, v, out_hw[0], out_hw[1])
            lons.append(lon)
            lats.append(lat)
        lons = torch.from_numpy(np.stack(lons)).to(e_img.device).type(e_img.dtype)
        lats = torch.from_numpy(np.stack(lats)).to(e_img.device).type(e_img.dtype)
        return remap(e_img, lons, lats, align_corners=True, mode=mode)
    else:
        he, we = e_img.shape[:2]
        lon, lat = map_pers_pix_to_equi(he, we, fov_deg, u_deg, v_deg, out_hw[0], out_hw[1])
        return cv2.remap(e_img, lon.astype(np.float32), lat.astype(np.float32), mode, borderMode=cv2.BORDER_WRAP)
