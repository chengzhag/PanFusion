from torch import Tensor
import cv2


def choose_mode(img, mode):
    if isinstance(img, Tensor):
        return mode if mode else 'bilinear'
    if mode == 'bilinear':
        mode = cv2.INTER_LINEAR
    elif mode in ('bicubic', None):
        mode = cv2.INTER_CUBIC
    elif mode == 'nearest':
        mode = cv2.INTER_NEAREST
    else:
        raise ValueError('mode must be one of [bilinear, bicubic, nearest]')


def index_list_or_scalar(lst, idx):
    if hasattr(lst, '__len__'):
        lst = lst[idx]
    if isinstance(lst, Tensor):
        lst = lst.item()
    return lst
