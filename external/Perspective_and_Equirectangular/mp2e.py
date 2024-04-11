import cv2
import numpy as np
from .p2e import p2e


def mp2e(p_imgs, fov_degs, u_degs, v_degs, out_hw, mode=None):
    merge_image = np.zeros((*out_hw, 3))
    merge_mask = np.zeros((*out_hw, 3))
    for p_img, fov_deg, u_deg, v_deg in zip(p_imgs, fov_degs, u_degs, v_degs):
        img, mask = p2e(p_img, fov_deg, u_deg, v_deg, out_hw, mode)
        mask = mask.astype(np.float32)
        img = img.astype(np.float32)

        weight_mask = np.zeros((p_img.shape[0], p_img.shape[1], 3))
        w = p_img.shape[1]
        weight_mask[:, 0:w//2, :] = np.linspace(0, 1, w//2)[..., None]
        weight_mask[:, w//2:, :] = np.linspace(1, 0, w//2)[..., None]
        weight_mask, _ = p2e(weight_mask, fov_deg, u_deg, v_deg, out_hw, mode)
        blur = cv2.blur(mask, (5, 5))
        blur = blur * mask
        mask = (blur == 1) * blur + (blur != 1) * blur * 0.05
        merge_image += img * weight_mask
        merge_mask += weight_mask

    merge_image[merge_mask == 0] = 255.
    merge_mask = np.where(merge_mask == 0, 1, merge_mask)
    merge_image = (np.divide(merge_image, merge_mask)).astype(np.uint8)
    return merge_image
