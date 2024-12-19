from cellbin2.contrib.param import ChipFeature
import numpy as np
from typing import List, Tuple
from cellbin2.contrib.alignment.basic import RegistrationInfo, AlignMode
from cellbin2.contrib.alignment.chip_box import chip_align
from cellbin2.contrib.alignment.template_centroid import centroid
from cellbin2.contrib.alignment.template_00pt import template_00pt_align
from cellbin2.utils import clog


def registration(moving_image: ChipFeature,
                 fixed_image: ChipFeature,
                 ref: Tuple[List, List],
                 from_stitched: bool = True) -> (RegistrationInfo, RegistrationInfo):
    """
    :param moving_image: 待配准图，通常是染色图（如ssDNA、HE）
    :param fixed_image: 固定图，通常是矩阵，支持TIF/GEM/GEF及数组
    :param ref: 模板周期，仅在模板相关配准方法下用到
    :param from_stitched: 从拼接图配准
    :return: RegistrationInfo
    """
    # if moving_image.template.trackcross_qc_pass_flag:
    #     res = centroid(moving_image=moving_image, fixed_image=fixed_image, ref=ref, from_stitched=from_stitched)
    # elif moving_image.chip_box.is_available:
    #     res = chip_align(moving_image=moving_image, fixed_image=fixed_image, from_stitched=from_stitched)
    # else:
    #     clog.info('Registration with no feature, cannot go on')
    #
    # return res

    # TODO 临时兼容性改动
    #  11/22 by lizepeng
    res_template = centroid(moving_image=moving_image, fixed_image=fixed_image, ref=ref, from_stitched=from_stitched)

    res_chip_box = chip_align(moving_image=moving_image, fixed_image=fixed_image, from_stitched=from_stitched)

    return res_template, res_chip_box


if __name__ == '__main__':
    from cellbin2.image import cbimread
    from cellbin2.contrib.param import TemplateInfo, ChipBoxInfo
    from cellbin2.utils.common import TechType

    template_ref = ([240, 300, 330, 390, 390, 330, 300, 240, 420],
                    [240, 300, 330, 390, 390, 330, 300, 240, 420])

    # 移动图像信息
    moving_image = ChipFeature()
    moving_image.tech_type = TechType.DAPI
    moving_mat = cbimread(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI.tif')
    moving_image.set_mat(moving_mat)
    img_tpl = TemplateInfo(template_recall=1.0, template_valid_area=1.0,
                           trackcross_qc_pass_flag=1, trackline_channel=0,
                           rotation=-0.53893, scale_x=1.0000665084, scale_y=1.00253792,
                           template_points=np.loadtxt(
                               r"E:/03.users/liuhuanlin/01.data/cellbin2/stitch/DAPI_matrix_template.txt"))
    moving_image.set_template(np.array(img_tpl))
    img_box = ChipBoxInfo(left_top=[162.28519045689168, 499.231306034147],
                          left_bottom=[377.99806165682605, 20502.069199051202],
                          right_top=[20210.76317636481, 314.47198219153387],
                          right_bottom=[20393.560877706364, 20277.53345880944],
                          scale_x=1.0006002898773978, scale_y=1.0028676122685343,
                          chip_size=(20004.000995228937, 20049.329304472536),
                          rotation=-0.5280016679897553,
                          is_available=True)
    moving_image.set_chip_box(img_box)

    # 固定对象信息
    fixed_image = ChipFeature()
    fixed_image.tech_type = TechType.Transcriptomics
    fixed_image.set_mat(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_gene.tif')
    matrix_tpl = TemplateInfo(template_recall=1.0, template_valid_area=1.0,
                              trackcross_qc_pass_flag=1, trackline_channel=0,
                              rotation=0., scale_x=1., scale_y=1.,
                              template_points=np.loadtxt(
                                  r"E:/03.users/liuhuanlin/01.data/cellbin2/stitch/A03599D1_gene.txt"))
    fixed_image.set_template(np.array(matrix_tpl))
    matrix_box = ChipBoxInfo(left_top=[124.,  1604.], left_bottom=[124., 21596.],
                             right_bottom=[20116., 21596.], right_top=[20116.,  1604.])

    # 多种方案的测试
    methods = [AlignMode.TemplateCentroid, AlignMode.Template00Pt, AlignMode.ChipBox, AlignMode.Voting]
    for m in methods[:1]:
        info = registration(moving_image=moving_image, fixed_image=fixed_image,
                            ref=template_ref, mode=AlignMode.TemplateCentroid)
        print(m, info)

