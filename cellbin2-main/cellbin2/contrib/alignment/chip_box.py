# -*- coding: utf-8 -*-
import numpy as np
from typing import Union

from cellbin2.contrib.param import ChipFeature
from cellbin2.contrib.alignment.basic import Alignment, RegistrationInfo, AlignMode
from cellbin2.utils import clog
from cellbin2.image import CBImage


class ChipAlignment(Alignment):
    """
    满足TissueBin需求：利用2模态芯片角为特征点，计算变换参数，实现配准。误差约100pix
    """
    def __init__(self,):
        super(ChipAlignment, self).__init__()

        self.register_matrix: np.matrix = None
        self.transforms_matrix: np.matrix = None

        self.rot90_flag = True
        self.no_trans_flag = False

        self.max_length = 9996  # 图像降采样最大尺寸

    def registration_image(self,
                           file: Union[str, np.ndarray, CBImage]):
        """ 对待变换的图像，调用图像处理库按照对齐参数，返回变换后的图 """

        if not isinstance(file, CBImage):
            image = cbimread(file)
        else:
            image = file

        if self.no_trans_flag:
            # TODO
            result = None
        else:
            result = image.trans_image(
                scale=[1 / self._scale_x, 1 / self._scale_y],
                rotate=self._rotation,
                rot90=self.rot90,
                offset=self.offset,
                dst_size=self._fixed_image.mat.shape,
                flip_lr=self.hflip
            )

        return result

    def align_stitched(
            self,
            fixed_image: ChipFeature,
            moving_image: ChipFeature
    ):
        """

        Args:
            fixed_image:
            moving_image:

        Returns:

        """
        self._rotation = -moving_image.chip_box.Rotation
        self._scale_x = moving_image.chip_box.ScaleX
        self._scale_y = moving_image.chip_box.ScaleY
        self._fixed_image = fixed_image

        if self.no_trans_flag:
            self.align_transformed(fixed_image, moving_image)
        else:
            transformed_image = self.transform_image(file=moving_image.mat)

            transformed_feature = ChipFeature()
            transformed_feature.set_mat(transformed_image)

            trans_mat = self.get_coordinate_transformation_matrix(
                moving_image.mat.shape,
                [1 / self._scale_x, 1 / self._scale_y],
                self._rotation
            )

            trans_points = self.get_points_by_matrix(moving_image.chip_box.chip_box, trans_mat)
            transformed_feature.chip_box.set_chip_box(trans_points)

            self.align_transformed(fixed_image, transformed_feature)

    def align_transformed(
            self,
            fixed_image: ChipFeature,
            moving_image: ChipFeature
    ):
        """

        Args:
            fixed_image:
            moving_image:

        Returns:

        """
        self._fixed_image = fixed_image
        coord_index = [0, 1, 2, 3]
        if self.rot90_flag: range_num = 4
        else: range_num = 1

        if self.hflip:
            new_box = self.transform_points(points = moving_image.chip_box.chip_box,
                                            shape = moving_image.mat.shape, flip=0)
            new_mi = np.fliplr(moving_image.mat.image)

            if new_mi.ndim == 3: new_mi = new_mi[:, :, 0]

        down_size = max(fixed_image.mat.shape) // self.max_length

        register_info = dict()
        for index in range(range_num):
            register_image, M = self.get_matrix_by_points(
                new_box[coord_index, :] / down_size, fixed_image.chip_box.chip_box / down_size,
                True, new_mi[::down_size, ::down_size], np.array(fixed_image.mat.shape) // down_size
            )

            lu_x, lu_y = map(int, fixed_image.chip_box.chip_box[0] / down_size)
            rd_x, rd_y = map(int, fixed_image.chip_box.chip_box[2] / down_size)

            _wsi_image = register_image[lu_y: rd_y, lu_x:rd_x]
            _gene_image = fixed_image.mat.image[::down_size, ::down_size][lu_y: rd_y, lu_x:rd_x]

            ms = self.multiply_sum(_wsi_image, _gene_image)
            # _, res = self.dft_align(_gene_image, _wsi_image, method = "sim")

            clog.info(f"Rot{index * 90}, Score: {ms}")
            register_info[index] = {"score": ms, "mat": M}  # , "res": res}

            coord_index.append(coord_index.pop(0))

        best_info = sorted(register_info.items(), key = lambda x: x[1]["score"], reverse = True)[0]
        if self.rot90_flag: self._rot90 = range_num - best_info[0]
        _mat = self.get_coordinate_transformation_matrix(
            moving_image.mat.shape, [1, 1], 90 * best_info[0]
        )
        _box = self.get_points_by_matrix(new_box, _mat)
        _box = self.check_border(_box)
        # self._offset = (fixed_image.chip_box.chip_box[0, 0] - (_box[0, 0] + _box[1, 0]) / 2,
        #                 fixed_image.chip_box.chip_box[0, 1] - (_box[0, 1] + _box[3, 1]) / 2)

        self._offset = (fixed_image.chip_box.chip_box[0, 0] - _box[0, 0],
                        fixed_image.chip_box.chip_box[0, 1] - _box[0, 1])


def chip_align(
        moving_image: ChipFeature,
        fixed_image: ChipFeature,
        from_stitched: bool = True
) -> RegistrationInfo:
    """
    :param moving_image: 待配准图，通常是染色图（如ssDNA、HE）
    :param fixed_image: 固定图
    :param from_stitched: 从拼接图开始

    Returns:

    """
    ca = ChipAlignment()
    if from_stitched: ca.align_stitched(fixed_image=fixed_image, moving_image=moving_image)
    else: ca.align_transformed(fixed_image=fixed_image, moving_image=moving_image)

    info = RegistrationInfo(**{
            'offset': tuple(list(ca.offset)),
            'counter_rot90': ca.rot90,
            'flip': ca.hflip,
            'register_score': ca.score,
            'register_mat': ca.registration_image(moving_image.mat),
            'method': AlignMode.TemplateCentroid
        }
    )

    return info


def manual_chip_box_register(
        image_path: str,
        image_points: np.ndarray,
        gene_path: str,
        gene_points: np.ndarray
):
    from cellbin2.contrib.chip_detector import ChipDetector
    from cellbin2.contrib.param import ChipBoxInfo
    from cellbin2.matrix.box_detect import MatrixBoxDetector

    moving_image = ChipFeature()
    moving_image.tech_type = TechType.DAPI
    moving_mat = cbimread(image_path)
    moving_image.set_mat(moving_mat)

    cd = ChipDetector(cfg=cfg, stain_type=TechType.DAPI)
    cd.set_corner_points(image_points)
    cd.detect(image_path, actual_size=(19992 * 2, 19992 * 2))
    info = {
        'LeftTop': cd.left_top, 'LeftBottom': cd.left_bottom,
        'RightTop': cd.right_top, 'RightBottom': cd.right_bottom,
        'ScaleX': cd.scale_x, 'ScaleY': cd.scale_y, 'Rotation': cd.rotation,
        'ChipSize': cd.chip_size, 'IsAvailable': cd.is_available
    }

    cbi = ChipBoxInfo(**info)
    moving_image.set_chip_box(cbi)

    fixed_image = ChipFeature()
    fixed_image.tech_type = TechType.Transcriptomics
    fixed_image.set_mat(gene_path)

    mbd = MatrixBoxDetector()
    cbi = ChipBoxInfo(LeftTop = gene_points[0], LeftBottom = gene_points[1],
                      RightBottom = gene_points[2], RightTop = gene_points[3])

    fixed_image.set_chip_box(cbi)

    result = chip_align(moving_image, fixed_image)


if __name__ == '__main__':
    from cellbin2.image import cbimread, cbimwrite
    from cellbin2.contrib.param import TemplateInfo
    from cellbin2.utils.common import TechType
    from cellbin2.contrib.chip_detector import ChipParam, detect_chip
    from cellbin2.matrix.box_detect import detect_chip_box


    #
    # # 移动图像信息
    # moving_image = ChipFeature()
    # moving_image.tech_type = TechType.DAPI
    # moving_mat = cbimread(r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\D04911A1C2\D04911A1C2_DAPI_stitch.tif")
    # moving_image.set_mat(moving_mat)
    #
    cfg = ChipParam(
        **{"stage1_weights_path":
            r"D:\01.code\cellbin2\weights\chip_detect_obb8n_640_SD_202409_pytorch.onnx",
            "stage2_weights_path":
            r"D:\01.code\cellbin2\weights\chip_detect_yolo8x_1024_SDH_stage2_202410_pytorch.onnx"})

    imp = r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\D04911A1C2\D04911A1C2_DAPI_stitch.tif"
    gmp = r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\D04911A1C2\D04911A1C2_Transcriptomics.tif"

    imc = r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\D04911A1C2\D04911A1C2_DAPI_stitch.txt"
    gmc = r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\D04911A1C2\gene_filter.txt"

    imcp = np.loadtxt(imc)
    gmcp = np.loadtxt(gmc)
    gmcp = gmcp + np.array([[10, 10], [10, -10], [-10, -10], [-10, 10]])

    manual_chip_box_register(imp, imcp, gmp, gmcp)

    #
    # # file_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI.tif"
    # m_info = detect_chip(moving_mat.image, cfg=cfg, stain_type=TechType.DAPI, actual_size=(19992 * 2, 19992 * 2))
    # moving_image.set_chip_box(m_info)
    #
    # # 固定对象信息
    # fixed_image = ChipFeature()
    # fixed_image.tech_type = TechType.Transcriptomics
    # fixed_image.set_mat(r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\D04911A1C2\D04911A1C2_Transcriptomics.tif")
    #
    # f_info = detect_chip_box(fixed_image.mat.image, chip_size = 2)
    # fixed_image.set_chip_box(f_info)
    #
    # result = chip_align(moving_image, fixed_image)
    # print(result)
    # cbimwrite(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI_registbox.tif', result.register_mat)
