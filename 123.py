# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2024/11/19 17:33
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : 123.py
🌟 Description  :

    基于图像和矩阵坐标点的配准
"""
import numpy as np
import cv2 as cv


def _check_border(file: np.ndarray):
    """ Check array, default (left-up, left_down, right_down, right_up)

    Args:
        file:

    Returns:

    """
    if not isinstance(file, np.ndarray): return None
    assert file.shape == (4, 2), "Array shape error."

    file = file[np.argsort(np.mean(file, axis = 1)), :]
    if file[1, 0] > file[2, 0]:
        file = file[(0, 2, 1, 3), :]

    file = file[(0, 1, 3, 2), :]

    return file


def transform_points(**kwargs):
    """
    角点翻转
    Args:
        **kwargs:
            points:
            shape:
            flip: 0 | 1, Y axis flip and X axis flip

    Returns:
    """
    points = kwargs.get("points", None)
    shape = kwargs.get("shape", None)

    if points is None or shape is None:
        return

    flip = kwargs.get("flip", None)
    if flip == 0:
        points[:, 0] = shape[1] - points[:, 0]
        points = points[[3, 2, 1, 0], :]
    elif flip == 1:
        points[:, 1] = shape[0] - points[:, 1]
        points = points[[1, 0, 3, 2], :]
    else:
        raise ValueError

    return points

def transform_image_with_points(src, points_src, points_dst, dst_shape,
                                need_image = True):
    """

    Args:
        src:
        points_src:
        points_dst:
        dst_shape:
        need_image:

    Returns:

    """
    if isinstance(src, str):
        src = cv.imread(src, -1)

    result = M = None

    if src is None: return result, M

    M = cv.getPerspectiveTransform(np.array(points_src, dtype = np.float32),
                                   np.array(points_dst, dtype = np.float32))

    if need_image:
        result = cv.warpPerspective(src, M, (dst_shape[1], dst_shape[0]))

    return result, M

wsi_border = np.array([[503,925],
                       [404,10900],
                       [10360,10982],
                       [10461,1021]])    # 影像图芯片坐标
gene_image_shape = [14700,14700]    # 矩阵图的h, w
wsi_image = cv.imread(r"C:\Users\zoujialin\Desktop\gold\demo\B04272D4\image\B04272D4_img.png", -1)    # 影像图路径

gene_border = np.array([[2752,2234],
                        [2759,12239],
                        [12757,12244],
                        [12738,2247]])

# flip=0水平翻转，flip=1垂直翻转
wsi_border = transform_points(points = wsi_border, shape = wsi_image.shape, flip = 1)
wsi_image = np.fliplr(wsi_image)    # 水平翻转
# wsi_image = np.flipud(wsi_image)    # 垂直翻转

cv.imwrite(r"C:\Users\zoujialin\Desktop\gold\demo\B04272D4\image\B04272D4_img_flip.png", wsi_image)

wsi_border = _check_border(wsi_border)

coord_index = [0, 1, 2, 3]
# [1,2,3,0]    顺时针旋转90
# [3,0,1,2]    逆时针旋转90

register_image, M = transform_image_with_points(
    wsi_image, wsi_border[coord_index, :],
    gene_border, gene_image_shape
)

# cv.imwrite(r"C:\Users\zoujialin\Desktop\gold\demo\B04372C211\B04372C211_regist.png", register_image)    # 生成的配准图需要与矩阵映射观察精度