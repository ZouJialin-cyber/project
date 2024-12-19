import copy

import cv2 as cv
import numpy as np
from enum import Enum

from numba import njit, prange
from typing import Union, List, Tuple, Any
from pydantic import BaseModel, Field

from cellbin2.image import CBImage
from cellbin2.image import cbimread
from cellbin2.contrib.param import ChipFeature
from cellbin2.utils.common import iPlaceHolder


class AlignMode(Enum):
    TemplateCentroid = 1
    Template00Pt = 2
    ChipBox = 3


class RegistrationInfo(BaseModel):
    counter_rot90: int = Field(0, description='')
    flip: bool = Field(True, description='')
    register_score: int = Field(-999, description='')
    offset: Tuple[float, float] = Field((0., 0.), description='')
    register_mat: Any = Field(None, description='')
    method: AlignMode = Field(AlignMode.TemplateCentroid, description='')
    dst_shape: Tuple[int, int] = Field((0, 0), description='')


class Alignment(object):
    """ 配准基类

    """

    def __init__(self, ):
        # input
        self._scale_x: float = 1.
        self._scale_y: float = 1.
        self._rotation: float = 0.

        # output
        self._offset: Tuple[float, float] = (0., 0.)
        self._rot90: int = 0
        self._hflip: bool = True
        self._score: int = iPlaceHolder

        # self.registration_image: CBImage
        self._fixed_image: ChipFeature = ChipFeature()
        self._moving_image: ChipFeature = ChipFeature()

    @property
    def offset(self, ) -> Tuple[float, float]:
        return self._offset

    @property
    def rot90(self, ) -> int:
        return self._rot90

    @property
    def hflip(self, ) -> bool:
        return self._hflip

    @property
    def score(self, ) -> int:
        return self._score

    def transform_image(
            self,
            file: Union[str, np.ndarray, CBImage]
    ):
        """ 对待变换的图像，调用图像处理库按照归一化参数，返回标准化后的图 """

        if not isinstance(file, CBImage):
            image = cbimread(file)
        else:
            image = file

        result = image.trans_image(
            scale=[1 / self._scale_x, 1 / self._scale_y],
            rotate=-self._rotation,
        )

        return result

    def registration_image(self,
                           file: Union[str, np.ndarray, CBImage]):
        """ 对待变换的图像，调用图像处理库按照对齐参数，返回变换后的图 """

        if not isinstance(file, CBImage):
            image = cbimread(file)
        else:
            image = file

        result = image.trans_image(
            scale=[1 / self._scale_x, 1 / self._scale_y],
            rotate=-self._rotation,
            rot90=self.rot90,
            offset=self.offset,
            dst_size=self._fixed_image.mat.shape,
            flip_lr=self.hflip
        )

        return result

    def get_coordinate_transformation_matrix(self, shape, scale, rotate):
        """
        图像变换后，点的变换前后位置的真实变换矩阵
        不同于 cv2.getRotationMatrix2D 和 cv2.warpPerspective等
        为基于原点(0, 0)的坐标系
        ** 此函数矩阵始终以图像左上为原点的坐标系

        Args:
            shape: h, w
            scale:
            rotate:

        Returns:

        """
        # 中心旋转的变换矩阵
        mat_scale_rotate = self.scale_rotate2mat(scale, rotate)
        mat_center_f = self._matrix_eye_offset(-shape[1] / 2, -shape[0] / 2)

        # 获得变换后的图像尺寸及offset
        x, y, _, _ = self.get_scale_rotate_shape(shape, scale, rotate)
        mat_offset = self._matrix_eye_offset(x / 2, y / 2)

        # 最终变换矩阵
        mat_result = mat_offset * mat_scale_rotate * mat_center_f
        return mat_result

    def get_scale_rotate_shape(self, shape, scale, rotate):
        """
        得到旋转缩放后的图像尺度大小
        Args:
            shape: h, w
            scale:
            rotate:

        Returns:
            x, y
        """
        mat = self.scale_rotate2mat(scale, rotate)
        points = np.array([[0, 0],
                           [0, shape[0]],
                           [shape[1], 0],
                           [shape[1], shape[0]]])

        result = mat[:2, :] @ np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).transpose(1, 0)
        x = result[0, :].max() - result[0, :].min()
        y = result[1, :].max() - result[1, :].min()

        x_d, y_d = np.min(np.array(result), axis=1)

        return x, y, x_d, y_d

    @staticmethod
    def scale_rotate2mat(scale: Union[int, float, List, Tuple] = 1,
                         rotate: Union[int, float] = 0,
                         offset: Union[List, Tuple] = None):
        """
        先缩放、旋转 后平移的矩阵变换
        Args:
            scale:
            rotate:
            offset: [x, y]

        Returns:

        """
        if isinstance(scale, (int, float)):
            scale_x = scale_y = scale
        else:
            scale_x, scale_y = scale

        mat_scale = np.mat([[scale_x, 0, 0],
                            [0, scale_y, 0],
                            [0, 0, 1]])

        mat_rotate = np.mat([[np.cos(np.radians(rotate)), -np.sin(np.radians(rotate)), 0],
                             [np.sin(np.radians(rotate)), np.cos(np.radians(rotate)), 0],
                             [0, 0, 1]])

        if offset is not None:
            mat_offset = np.mat([[1, 0, offset[0]],
                                 [0, 1, offset[1]],
                                 [0, 0, 1]])

            mat = mat_offset * mat_scale * mat_rotate
        else:

            mat = mat_scale * mat_rotate

        return mat

    @staticmethod
    def get_points_by_matrix(points, mat):
        """
        图像点在变换矩阵后的新坐标
        Args:
            points:
            mat:

        Returns:

        """
        if points.ndim == 1:
            _points = np.array([points])
        else:
            _points = points

        _points = _points[:, :2]

        new_points = mat[:2, :] @ np.concatenate([
            _points, np.ones((_points.shape[0], 1))],
            axis=1
        ).transpose(1, 0)

        return np.array(new_points).squeeze().transpose()

    @staticmethod
    def get_matrix_by_points(points_src, points_dst,
                             need_image = False,
                             src: Union[CBImage, np.ndarray] = None,
                             dst_shape: tuple = None
                             ):
        """

        Args:
            src:
            points_src:
            points_dst:
            dst_shape:
            need_image:

        Returns:

        """
        result = M = None

        if src is None: return result, M

        M = cv.getPerspectiveTransform(np.array(points_src, dtype=np.float32),
                                       np.array(points_dst, dtype=np.float32))

        if need_image:
            src = src if isinstance(src, np.ndarray) else src.image
            result = cv.warpPerspective(src, M, (dst_shape[1], dst_shape[0]))

        return result, M

    @staticmethod
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

    @staticmethod
    def _matrix_eye_offset(x, y, n=3):
        """

        Args:
            x:
            y:

        Returns:

        """
        mat = np.eye(n)
        mat[:2, 2] = [x, y]
        return mat

    @staticmethod
    def get_mass(image):
        """

        Args:
            image:

        Returns:

        """
        M = cv.moments(image)
        cx_cv = round(M['m10'] / M['m00'])
        cy_cv = round(M['m01'] / M['m00'])

        return cx_cv, cy_cv

    @staticmethod
    def check_border(file: np.ndarray):
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

    @staticmethod
    @njit(parallel=True)
    def multiply_sum(a, b):
        """
        计算矩阵相乘后的累加和
        """
        res = 0
        (h, w) = a.shape
        for i in prange(h):
            for j in range(w):
                res += a[i][j] * b[i][j]
        return res


def transform_points(
        points: np.ndarray,
        src_shape: Tuple[int, int],
        scale: Union[int, float, list, tuple] = 1,
        rotation: Union[float, int] = 0,
        offset: tuple = (0, 0),
        flip: int = -1
) -> [np.ndarray, np.matrix]:
    """

    Args:
        points: n * 2/4大小数组 -- (x, y)
        src_shape: 原始图像尺寸 -- (h, w)
        scale:
        rotation:
        offset: 该值定义为做完所有变换操作后再操作 -- (x, y)
        flip: -1 表示不做  0为水平翻转 1为垂直翻转

    Returns:
        new_points:
        mat: 仅为scale和rotate的变换矩阵

    """
    align = Alignment()

    mat = align.get_coordinate_transformation_matrix(shape=src_shape, scale=scale, rotate=rotation)
    if flip == 0: points[:, 0] = src_shape[1] - points[:, 0]
    elif flip == 1: points[:, 1] = src_shape[0] - points[:, 1]
    else: pass

    new_points = align.get_points_by_matrix(points, mat)

    new_points = new_points + offset

    p = copy.copy(points)
    p[:, :2] = new_points

    return p, mat

def read_points_from_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split()
            if len(values) == 4:
                x1 = float(values[0])
                y1 = float(values[1])
                x2 = float(values[2])
                y2 = float(values[3])
                points.append([x1, y1])
                points.append([x2, y2])
    return np.array(points)


def write_points_to_file(dst_points, output_file):
    with open(output_file, 'w') as file:
        # 遍历每个点
        for point in dst_points:
            # 检查 point 是否包含两个元素
            if len(point) != 2:
                # 如果 point 包含多个点，逐个处理每个点
                for i in range(0, len(point), 2):
                    # 确保 point[i] 包含两个元素
                    if len(point[i]) != 2:
                        break
                    print(point[i])
                    x = point[i][0]
                    y = point[i][1]
                    # 输出 x 和 y 的值和类型以进行调试
                    print(f"x value: {x}, x type: {type(x)}")
                    print(f"y value: {y}, y type: {type(y)}")
                    # 写入文件
                    file.write(f"{x:.6f} {y:.6f}\n")
            else:
                # 直接访问 point 的元素
                x = point[0]
                y = point[1]
                # 输出 x 和 y 的值和类型以进行调试
                print(f"x value: {x}, x type: {type(x)}")
                print(f"y value: {y}, y type: {type(y)}")
                # 写入文件
                file.write(f"{x:.6f} {y:.6f}\n")

if __name__ == "__main__":
    file_path = r"C:\Users\zoujialin\Desktop\gold\demo\B04272D4\ipr_track_merge\untransformed_point.txt"
    src_points = read_points_from_file(file_path)
    # src_points = np.array([[21047, 21015],
    #                        [1063, 20747],
    #                        [1330, 765],
    #                        [21313, 1031]])

    ScaleX = 1.00243304114324
    ScaleY = 0.9992502748125687
    Rotation = 0.7688500000000001

    dst_points = transform_points(src_points,
                                  src_shape=(22187, 24288),
                                  scale=(1 / ScaleX, 1 / ScaleY),
                                  rotation=Rotation,
                                  offset=(-1645.75, 572.5),
                                  flip=0)
    # print(dst_points)
    output_file = r"C:\Users\zoujialin\Desktop\gold\demo\B04272D4\our_track_merge\transformed_point.txt"
    write_points_to_file(dst_points, output_file)