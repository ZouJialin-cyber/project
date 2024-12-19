import numpy as np

from typing import List, Tuple, Union

from cellbin2.contrib.track_align import AlignByTrack
from cellbin2.contrib.alignment.basic import Alignment
from cellbin2.contrib.alignment import AlignMode, RegistrationInfo
from cellbin2.contrib.param import ChipFeature
from cellbin2.image import CBImage, cbimread


class Template00PtAlignment(Alignment):
    """
    满足CellBin需求，利用芯片切割特性，实现左上角第一个周期起始点对齐。实现配准，误差在10pix
    """

    def __init__(self,
                 ref: Tuple[List, List] = ([], []),
                 shape: Tuple[int, int] = (0, 0)
                 ):
        super(Template00PtAlignment, self).__init__()
        self._reference = ref
        self._register_shape = shape

        self.fixed_template: np.ndarray = np.array([])
        self.fixed_box: List[float] = [0, 0, 0, 0]

    def registration_image(
            self,
            file: Union[str, np.ndarray, CBImage]
    ) -> CBImage:
        """

        Args:
            file:

        Returns:

        """
        if not isinstance(file, CBImage):
            image = cbimread(file)
        else:
            image = file

        result = image.trans_image(
            scale=[1 / self._scale_x, 1 / self._scale_y],
            rotate=self._rotation,
            rot90=self.rot90,
            offset=self.offset,
            dst_size=self._register_shape,
            flip_ud=True  # 配准前置默认是上下翻转即可对齐
        )

        return result

    def align_stitched(self, moving_image: ChipFeature, ):
        """

        Args:
            moving_image:

        Returns:

        """
        self._scale_x, self._scale_y = moving_image.template.scale_x, moving_image.template.scale_y
        self._rotation = -moving_image.template.rotation

        transformed_image = self.transform_image(file=moving_image.mat)

        transformed_feature = ChipFeature()
        transformed_feature.set_mat(transformed_image)

        trans_mat = self.get_coordinate_transformation_matrix(
            moving_image.mat.shape,
            [1 / self._scale_x, 1 / self._scale_y],
            self._rotation
        )

        trans_points = self.get_points_by_matrix(
            np.array(moving_image.template.template_points),
            trans_mat
        )

        chip_points = self.get_points_by_matrix(
            moving_image.chip_box.chip_box,
            trans_mat
        )

        transformed_feature.set_point00(moving_image.point00)
        transformed_feature.set_anchor_point(moving_image.anchor_point)
        transformed_feature.chip_box.set_chip_box(chip_points)
        transformed_feature.set_template(
            np.concatenate(
                [trans_points, np.array(moving_image.template.template_points)[:, 2:]],
                axis=1
            )
        )

        self.align_transformed(transformed_feature)

    def align_transformed(self, moving_image: ChipFeature):
        """

        Args:
            moving_image:

        Returns:

        """
        if self.hflip:
            points = AlignByTrack.flip_points(
                moving_image.template.template_points,
                moving_image.mat.shape, self._reference,
                axis=1
            )
            chip_box_points = self.transform_points(
                points=moving_image.chip_box.chip_box,
                shape=moving_image.mat.shape,
                flip=1)
        else:
            points = moving_image.template.template_points
            chip_box_points = moving_image.chip_box.chip_box

        _points = points.copy()
        _points[:, :2] = _points[:, :2] - chip_box_points[0]

        _points = self.get_lt_zero_point(_points)
        _points = _points[(_points[:, 0] > 0) & (_points[:, 1] > 0)]

        px, py = sorted(
            _points.tolist(),
            key=lambda x: np.abs(x[0] - moving_image.anchor_point[0]) + np.abs(x[1] - moving_image.anchor_point[1])
        )[0] + chip_box_points[0]

        self._offset = [moving_image.point00[0] - px, moving_image.point00[1] - py]

    @staticmethod
    def get_lt_zero_point(template_points, x_index=0, y_index=0):
        """
        Args:
            template_points: np.array, 模板点 -- shape == (*, 4)
            x_index:
            y_index:
        Returns:
            zero_template_points: np.array
        """
        zero_template_points = template_points[(template_points[:, 3] == y_index) &
                                               (template_points[:, 2] == x_index)][:, :2]
        return zero_template_points


def template_00pt_align(
        moving_image: ChipFeature,
        ref: Tuple[List, List],
        dst_shape: Tuple[int, int],
        from_stitched: bool = True
) -> RegistrationInfo:
    """
    :param moving_image: 待配准图，通常是染色图（如ssDNA、HE）
    :param ref: 模板周期，仅在模板相关配准方法下用到
    :param dst_shape: 配准图理论尺寸
    :param from_stitched
    :return: RegistrationInfo
    """
    tpa = Template00PtAlignment(ref=ref, shape=dst_shape)
    if from_stitched:
        tpa.align_stitched(moving_image=moving_image)
    else:
        tpa.align_transformed(moving_image=moving_image)

    info = RegistrationInfo(**{
        'offset': tuple(list(tpa.offset)),
        'flip': tpa.hflip,
        'register_score': tpa.score,
        'register_mat': tpa.registration_image(moving_image.mat),
        'method': AlignMode.Template00Pt,
        'dst_shape': dst_shape
    }
                            )

    return info


if __name__ == '__main__':
    import os
    from cellbin2.image import cbimread, cbimwrite
    from cellbin2.contrib.param import TemplateInfo
    from cellbin2.utils.common import TechType
    from cellbin2.utils.stereo_chip import StereoChip
    from cellbin2.contrib.chip_detector import ChipParam, detect_chip

    template_ref = ([240, 300, 330, 390, 390, 330, 300, 240, 420],
                    [240, 300, 330, 390, 390, 330, 300, 240, 420])

    # 移动图像信息
    moving_image = ChipFeature()
    moving_image.tech_type = TechType.DAPI
    moving_mat = cbimread(r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_gene.tif")
    moving_image.set_mat(moving_mat)
    img_tpl = TemplateInfo(template_recall=1.0, template_valid_area=1.0,
                           trackcross_qc_pass_flag=1, trackline_channel=0,
                           rotation=0.11493999999999997, scale_x=0.9797232231120153, scale_y=0.978182155478651,
                           template_points=np.loadtxt(
                               r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\DAPI_matrix_template.txt"))
    moving_image.set_template(img_tpl)

    cfg = ChipParam(
        **{"DAPI_stage1_weights_path":
               r"E:/03.users/liuhuanlin/01.data/cellbin2/weights\chip_detect_obb8n_640_SD_202409_pytorch.onnx",
           "DAPI_stage2_weights_path":
               r"E:/03.users/liuhuanlin/01.data/cellbin2/weights\chip_detect_yolo8m_1024_SD_202409_pytorch.onnx"})

    # file_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI.tif"
    m_info = detect_chip(moving_mat.image, cfg=cfg, stain_type=TechType.DAPI, actual_size=(19992, 19992))
    moving_image.set_chip_box(m_info)

    # 矩阵理论原点位置
    chip_mask_file = os.path.join(r'E:\03.users\liuhuanlin\02.code\cellbin2\cellbin\config\chip_mask.json')
    sc = StereoChip(chip_mask_file)
    sc.parse_info('A03599D1')
    moving_image.set_point00(sc.zero_zero_point)

    info = template_00pt_align(moving_image=moving_image, ref=template_ref, dst_shape=(sc.height, sc.width))
    print(info)
    cbimwrite(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI_regist00.tif', info.register_mat)
