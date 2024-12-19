#  新的流程，位于imageQC后（充分利用各个独立模块计算所得的结果）
import sys
import os
import shutil

from cellbin2.utils.config import Config
from cellbin2.utils.common import TechType
from cellbin2.utils import clog
import numpy as np
from cellbin2.image import cbimread, CBImage
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.utils import ipr, rpi
from cellbin2.contrib.alignment import AlignMode
from cellbin2.utils.weights_manager import WeightDownloader
from typing import List, Dict, Any, Tuple, Union
from cellbin2.image import cbimwrite
from cellbin2.modules import naming, run_state
from cellbin2.modules.metadata import ProcParam, ProcFile, default_image, read_param_file
from cellbin2.modules.extract.image_extract import ImageFeatureExtract
from cellbin2.modules.extract.matrix_extract import MatrixFeatureExtract
from cellbin2.contrib.mask_manager import merge_cell_mask, BestTissueCellMask
from cellbin2.contrib.fast_correct import run_fast_correct
from cellbin2.utils.pro_monitor import process_decorator
from cellbin2.utils.common import ErrorCode


class Scheduler(object):
    """
        配准、分割及校准、矩阵提取
    """

    def __init__(self, config_file: str, chip_mask_file: str, weights_root: str):
        self.weights_root = weights_root
        self.param_chip = StereoChip(chip_mask_file)
        self.config = Config(config_file, weights_root)
        self._files: Dict[int, ProcFile] = {}
        self._ipr = ipr.ImageProcessRecord()
        self._channel_images: Dict[str, Union[ipr.IFChannel, ipr.ImageChannel]] = {}  # ipr.ImageChannel
        self._output_path: str = ''
        # self._image_naming: naming.DumpImageFileNaming
        # self._matrix_naming: naming.DumpMatrixFileNaming

    @process_decorator('GiB')
    def _dump_ipr(self, output_path: str):
        ipr.write(file_path=output_path, ipr=self._ipr, extra_images=self._channel_images)
        clog.info('Dump ipr to {}'.format(output_path))

    @process_decorator('GiB')
    def _dump_rpi(self, rpi_path: str):
        data = {}
        for idx, f in self._files.items():
            if f.is_image:
                g_name = f.get_group_name(sn=self.param_chip.chip_name)
                n = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name, stain_type=g_name, save_dir=self._output_path)
                c_mask_path = n.cell_mask
                c_raw_mask_path = n.cell_mask_raw
                register_path = n.registration_image
                t_mask_path = n.tissue_mask
                t_raw_mask_path = n.tissue_mask_raw
                data[g_name] = {}
                if os.path.exists(c_mask_path):
                    data[g_name]['CellMask'] = c_mask_path
                if os.path.exists(register_path):
                    data[g_name]['Image'] = register_path
                if os.path.exists(t_mask_path):
                    data[g_name]['TissueMask'] = t_mask_path
                if os.path.exists(t_raw_mask_path):
                    data[g_name]['TissueMaskRaw'] = t_raw_mask_path
                if os.path.exists(c_raw_mask_path):
                    data[g_name]['CellMaskRaw'] = c_raw_mask_path
                # if os.path.exists(c_correct_mask_path):
                #     data[g_name]['CellMaskCorrect'] = c_correct_mask_path
        data['final'] = {}
        data['final']['CellMask'] = self.p_naming.final_nuclear_mask
        data['final']['TissueMask'] = self.p_naming.final_tissue_mask
        data['final']['CellMaskCorrect'] = self.p_naming.final_cell_mask
        rpi.write(h5_path=rpi_path, extra_images=data)
        clog.info('Dump rpi to {}'.format(rpi_path))

    def _weights_check(self, ):
        weights = []
        for idx, f in self._files.items():
            if f.cell_segmentation:
                wp = self.config.cell_segmentation.get_weights_path(f.tech)
                if wp is None: return 1
                weights.append(os.path.basename(wp))

            if f.tissue_segmentation:
                wp = self.config.tissue_segmentation.get_weights_path(f.tech)
                if wp is None: return 1
                weights.append(os.path.basename(wp))

        weights = list(set(weights))
        wd = WeightDownloader(save_dir=self.weights_root)
        flag = wd.download_weight_by_names(weight_names=weights)
        if flag != 0: clog.warning('Failed to retrieve the weights file from local or server')

        return flag

    def _data_check(self, ):
        if len(self._files) < 1:
            clog.warning('No data was found that needed to be analyzed')
            return 3
        else:
            wh = []
            for idx, f in self._files.items():
                if not f.is_image: continue
                # if not os.path.exists(f.file_path):
                #     clog.warning('Missing file, {}'.format(f.file_path))
                #     sys.exit(1)  # 缺失文件，非正常退出
                image = cbimread(f.file_path)
                wh.append([image.width, image.height])

            s = np.unique(wh, axis=0)
            if s.shape[0] > 1:
                clog.warning('The sizes of the images are inconsistent')
                return 1
            elif s.shape[0] == 1:
                clog.info(
                    'Images info as (size, channel, depth) == ({}, {}, {})'.format(s[0], image.channel, image.depth))
            else:
                clog.info('No image data need deal')
                return 2
        return 0

    def _mask_merge(self, im_naming1: naming.DumpImageFileNaming, im_naming2: naming.DumpImageFileNaming):
        from cellbin2.contrib.mask_manager import merge_cell_mask
        mask1 = cbimread(im_naming1.cell_mask, only_np=True)
        mask2 = cbimread(im_naming2.cell_mask, only_np=True)
        merged_mask = final_mask = merge_cell_mask(mask1, mask2)
        # TODO: 这里暂时就默认im_naming1是核分割结果，im_naming2是膜分割结果
        return merged_mask

    def _read_transform(self, image_file: ProcFile):
        """ 标准化完成对图像位置、尺度及角度的标准放置：位置针对的是固定染色图，尺度及角度针对的是矩阵
        :param image_file: 待标准化的图文件信息
        :return:
        """
        c_name = image_file.get_group_name(sn=self.param_chip.chip_name)
        if image_file.channel_align != -1:  # 如果该图是校准图，则先切换到其对齐图上
            if self._channel_images[c_name].Calibration.CalibrationQCPassFlag:  # 校准通过
                # 先平移
                offset = (self._channel_images[c_name].Calibration.Scope.OffsetX,
                          self._channel_images[c_name].Calibration.Scope.OffsetY)
            else:  # 校准不通过，不贸然操作
                offset = (0, 0)
        else:
            offset = (0, 0)
        s = (1., 1.)  # default
        r = 0.  # default
        if image_file.tech != TechType.IF:
            if self.qc_flag == 1:  # 现有条件下无法标准化，保持原样
                # 获取配准参数
                if self._channel_images[c_name].QCInfo.TrackCrossQCPassFlag:
                    s = (1 / self._channel_images[c_name].Register.ScaleX,
                         1 / self._channel_images[c_name].Register.ScaleY)
                    r = self._channel_images[c_name].Register.Rotation
                else:
                    s = (1 / self._channel_images[c_name].QCInfo.ChipBBox.ScaleX,
                         1 / self._channel_images[c_name].QCInfo.ChipBBox.ScaleY)
                    r = self._channel_images[c_name].QCInfo.ChipBBox.Rotation
        else:
            reuse_channel = image_file.registration.reuse
            reuse_g_name = self._files[reuse_channel].get_group_name(sn=self.param_chip.chip_name)
            if reuse_channel != -1:
                s = (
                    1 / self._channel_images[reuse_g_name].Register.ScaleX,
                    1 / self._channel_images[reuse_g_name].Register.ScaleY
                )
                r = self._channel_images[reuse_g_name].Register.Rotation

        return s, r, offset

    @process_decorator('GiB')
    def _registration(self, image_file: ProcFile):
        """
        完成对校准图及非校准图的配准: transform图是起点
        :param image_file:
        :return:
        """
        from cellbin2.contrib.alignment import registration
        from cellbin2.contrib.param import ChipFeature

        fixed = self._files[image_file.registration.fixed_image]
        # 动图参数构建
        g_name = image_file.get_group_name(sn=self.param_chip.chip_name)
        param1 = self._channel_images[g_name]

        moving_image = ChipFeature()
        moving_image.tech_type = image_file.tech
        moving_image.set_mat(self._image_naming.transformed_image)
        # 这里建议不要去ipr读，而是
        if param1.QCInfo.TrackCrossQCPassFlag:
            moving_image.set_template(np.loadtxt(self._image_naming.transformed_template))  # param1.transform_template_info
        if param1.QCInfo.ChipDetectQCPassFlag:
            moving_image.set_chip_box(param1.Stitch.TransformChipBBox.get())

        # 静图参数构建
        if fixed.tech in (TechType.Protein, TechType.Transcriptomics):
            # 场景1：静图是矩阵
            mfe = MatrixFeatureExtract(
                output_path=self._output_path,
                image_file=fixed,
                m_naming=naming.DumpMatrixFileNaming(
                    sn=self.param_chip.chip_name,
                    m_type=fixed.tech.name,
                    save_dir=self._output_path
                )
            )
            mfe.set_chip_param(self.param_chip)
            mfe.set_config(self.config)
            mfe.extract4stitched()

            fixed_image = ChipFeature()
            fixed_image.tech_type = fixed.tech
            fixed_image.set_mat(mfe.mat)
            fixed_image.set_template(mfe.template)
            fixed_image.set_chip_box(mfe.chip_box)
            self._channel_images[g_name].Register.MatrixTemplate = mfe.template.template_points

        else:
            # 场景2：静图是染色图
            fixed_image = ChipFeature()
            fixed_image.tech_type = fixed.tech
            fixed_image.set_mat(fixed.file_path)
            param2 = self._channel_images[fixed.tag]
            fixed_image.set_template(param2.transform_template_info)
            fixed_image.set_chip_box(param2.box_info)

        # TODO 临时兼容性改动
        #  11/22 by lizepeng
        info, temp_info = registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            ref=self.param_chip.fov_template,
            from_stitched=False
        )
        self._channel_images[g_name].update_registration(info)
        self._channel_images[g_name].Register.GeneChipBBox.update(fixed_image.chip_box)
        temp_info.register_mat.write(os.path.join(self._output_path, f"{self._image_naming.sn}_chip_box_register.tif"))
        np.savetxt(os.path.join(self._output_path, f"{self._image_naming.sn}_chip_box_register.txt"), temp_info.offset)

        return info

    def run(self, chip_no: str, input_image: str,
            stain_type: str, param_file: str,
            output_path: str, ipr_path: str,
            matrix_path: str, kit: str):
        self._output_path = output_path
        # 芯片信息加载
        self.param_chip.parse_info(chip_no)
        self.p_naming = naming.DumpPipelineFileNaming(chip_no=chip_no, save_dir=self._output_path)
        # 数据加载
        if os.path.exists(param_file):  # 参数文件图加入计算列表
            pp = read_param_file(
                file_path=param_file,
                cfg=self.config,
                out_path=self.p_naming.input_json
            )
        else:
            pp = ProcParam()
        # 命令行图加入计算列表
        # if input_image != '':
        #     pp.add_image(file_path=input_image, stain_type=stain_type, clarity=self.config.default_image.clarity)
        # if matrix_path != '':
        #     pp.add_matrix(file_path=matrix_path, tech_type=kit, stain_type=stain_type)

        self._files = pp.get_image_files(do_image_qc=False, do_scheduler=True, cheek_exists=True)
        pp.print_files_info(self._files, mode='Scheduler')

        # 数据校验失败则退出
        flag1 = self._data_check()
        if flag1 not in [0, 2]:
            return 1
        if flag1 == 0:
            self._ipr, self._channel_images = ipr.read(ipr_path)

        # 模型加载
        flag2 = self._weights_check()
        if flag2 != 0:
            sys.exit(1)

        # 遍历图像，执行分析过程
        for idx, f in self._files.items():
            clog.info('======>  File[{}] CellBin, {}'.format(idx, f.file_path))
            if f.is_image:
                g_name = f.get_group_name(sn=self.param_chip.chip_name)
                self._image_naming = naming.DumpImageFileNaming(
                    sn=self.param_chip.chip_name, stain_type=g_name,
                    save_dir=self._output_path
                )

                if f.channel_align != -1:  # 如果该图是校准图，则先切换到其对齐图上
                    file_tag = self._files[f.channel_align].tech.name
                else:
                    file_tag = f.tech.name

                if not f.tech == TechType.IF:
                    # TODO: deal with two versions of ipr
                    qc_ = self._channel_images[g_name].QCInfo
                    if hasattr(qc_, 'QcPassFlag'):
                        self.qc_flag = getattr(qc_, 'QcPassFlag')
                    else:
                        self.qc_flag = getattr(qc_, 'QCPassFlag')
                    if self.qc_flag != 1:  # 现有条件下无法配准
                        clog.warning('Image QC not pass, cannot deal this pipeline')
                        sys.exit(ErrorCode.qcFail.value)

                s, r, offset = self._read_transform(f)
                # Transform 操作：Transform > segmentation > mask merge & expand
                cur_f_name = naming.DumpImageFileNaming(
                        sn=self.param_chip.chip_name,
                        stain_type=g_name,
                        save_dir=self._output_path
                    )
                ife = ImageFeatureExtract(
                    output_path=output_path,
                    image_file=f,
                    sn=self.param_chip.chip_name,
                    im_naming=cur_f_name,
                )
                ife.set_chip_param(self.param_chip)
                ife.set_config(self.config)
                ife.set_channel_image(self._channel_images[g_name])
                if 1:  # TODO: 先不跳过这一步了
                    ife.extract4transform(scale=s, rotation=r, offset=offset)
                    self._channel_images[g_name] = ife.channel_image

                # if not image_state.registration_image:
                if file_tag != g_name:
                    info = self._channel_images[file_tag].get_registration()
                    clog.info('Get registration param from ipr')
                else:
                    # if not os.path.exists(self._files[f.registration.fixed_image].file_path):
                    #     clog.warning('Miss fixed image, {}'.format(self._files[f.registration.fixed_image].file_path))
                    #     continue
                    # 没有配准 & 固定图存在

                    # TODO 配准前置暂关
                    #  11/22 by lizepeng
                    # if self._channel_images[file_tag].Register.Method == AlignMode.Template00Pt.name:  # 先前做了前置配准
                    #     # 从ipr获取配准参数
                    #     info = self._channel_images[file_tag].get_registration()
                    #     clog.info('Get registration param from ipr')
                    # else:
                    info = self._registration(f)

                ife.transform2regist(info)
            else:
                mfe = MatrixFeatureExtract(
                    output_path=self._output_path,
                    image_file=f,
                    m_naming=naming.DumpMatrixFileNaming(
                        sn=self.param_chip.chip_name,
                        m_type=f.tech.name,
                        save_dir=self._output_path,
                    )
                )
                mfe.set_chip_param(self.param_chip)
                mfe.set_config(self.config)
                mfe.extract4stitched(detect_feature=False)
                if f.tissue_segmentation:
                    mfe.tissue_segmentation()
                if f.cell_segmentation:
                    mfe.cell_segmentation()
        if flag1 == 0:
            self._dump_ipr(self.p_naming.ipr)
        molecular_classify_files = pp.get_molecular_classify()
        for idx, m in molecular_classify_files.items():
            clog.info('======>  Extract[{}], {}'.format(idx, m))
            picked_mask = m.cell_mask
            final_nuclear_path = self.p_naming.final_nuclear_mask
            final_t_mask_path = self.p_naming.final_tissue_mask
            final_cell_mask_path = self.p_naming.final_cell_mask
            if 0 < len(picked_mask) < 3:
                if len(picked_mask) == 1:  # 就一个mask
                    im_naming = naming.DumpImageFileNaming(
                        sn=self.param_chip.chip_name,
                        stain_type=self._files[picked_mask[0]].tech.name,
                        save_dir=self._output_path
                    )
                    to_fast = final_nuclear_path
                else:  # 两个mask，现在默认第一个是核mask，第二个是膜mask
                    im_naming1 = naming.DumpImageFileNaming(
                        sn=self.param_chip.chip_name,
                        stain_type=self._files[picked_mask[0]].get_group_name(sn=self.param_chip.chip_name),
                        save_dir=self._output_path
                    )
                    im_naming2 = naming.DumpImageFileNaming(
                        sn=self.param_chip.chip_name,
                        stain_type=self._files[picked_mask[1]].get_group_name(sn=self.param_chip.chip_name),
                        save_dir=self._output_path
                    )
                    if im_naming1.stain_type == TechType.IF.name:
                        im_naming = im_naming2
                    else:
                        im_naming = im_naming1
                    merged_cell_mask_path = im_naming.cell_mask_merged
                    if not os.path.exists(merged_cell_mask_path):
                        merged_mask = self._mask_merge(im_naming1=im_naming1, im_naming2=im_naming2)
                        cbimwrite(merged_cell_mask_path, merged_mask)
                    to_fast = merged_cell_mask_path

                shutil.copy2(im_naming.cell_mask, final_nuclear_path)
                shutil.copy2(im_naming.tissue_mask, final_t_mask_path)
                # here, we got final cell mask and final tissue mask
                if not os.path.exists(final_cell_mask_path):
                    fast_mask = run_fast_correct(
                        mask=to_fast,
                        dis=self.config.cell_correct.expand_r,
                        process=self.config.cell_correct.process
                    )
                    cbimwrite(final_cell_mask_path, fast_mask)
        if flag1 == 0:
            self._dump_rpi(self.p_naming.rpi)
        for idx, m in molecular_classify_files.items():
            if m.exp_matrix == -1:
                continue
            matrix = self._files[m.exp_matrix]
            mfe = MatrixFeatureExtract(
                output_path=self._output_path,
                image_file=matrix,
                m_naming=naming.DumpMatrixFileNaming(
                    sn=self.param_chip.chip_name,
                    m_type=matrix.tech.name,
                    save_dir=self._output_path
                )
            )
            mfe.set_chip_param(self.param_chip)
            mfe.set_config(self.config)
            mfe.extract4matrix(m_naming=self.p_naming)


def scheduler_pipeline(weights_root: str, chip_no: str, input_image: str, stain_type: str,
                       param_file: str, output_path: str, matrix_path: str, ipr_path: str,
                       kit: str):
    """
    :param weights_root: CNN权重文件本地存储目录路径
    :param chip_no: 样本芯片号
    :param input_image: 染色图本地路径
    :param stain_type: 染色图对应的染色类型
    :param param_file: 入参文件本地路径
    :param output_path: 输出文件本地存储目录路径
    :param matrix_path: 表达矩阵本地存储路径
    :param ipr_path: 图像记录文件本地存储路径
    :param kit:
    :return: int(状态码)
    """
    curr_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(curr_path, r'../config/cellbin.yaml')
    chip_mask_file = os.path.join(curr_path, r'../config/chip_mask.json')

    iqc = Scheduler(config_file=config_file, chip_mask_file=chip_mask_file, weights_root=weights_root)
    iqc.run(chip_no=chip_no,
            input_image=input_image,
            stain_type=stain_type,
            param_file=param_file,
            output_path=output_path,
            ipr_path=ipr_path,
            matrix_path=matrix_path, kit=kit)


def main(args, para):
    scheduler_pipeline(weights_root=args.weights_root, chip_no=args.chip_no,
                       input_image=args.image_path, stain_type=args.stain_type,
                       param_file=args.param_file, output_path=args.output_path,
                       ipr_path=args.ipr_path, matrix_path=args.matrix_path, kit=args.kit)


if __name__ == '__main__':
    import argparse

    _VERSION_ = '2.0'

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action="version", version=_VERSION_)
    parser.add_argument("-c", "--chip_no", action="store", dest="chip_no", type=str, required=True,
                        help="The SN of chip.")
    parser.add_argument("-i", "--image_path", action="store", dest="image_path", type=str, required=False,
                        help="The path of input file.")
    parser.add_argument("-s", "--stain_type", action="store", dest="stain_type", type=str, required=False,
                        help="The stain type of input image.")
    parser.add_argument("-p", "--param_file", action="store", dest="param_file", type=str, required=False,
                        default='', help="The path of input param file.")
    parser.add_argument("-o", "--output_path", action="store", dest="output_path", type=str, required=True,
                        help="The results output folder.")
    parser.add_argument("-w", "--weights_root", action="store", dest="weights_root", type=str, required=True,
                        help="The weights folder.")
    parser.add_argument("-k", "--kit", action="store", dest="kit", type=str, required=False,
                        help="Kit Type.(Transcriptomics, Protein)")
    parser.add_argument("-r", "--ipr_path", action="store", dest="ipr_path", type=str, required=True,
                        help="Path of image process record file.")
    parser.add_argument("-m", "--matrix_path", action="store", dest="matrix_path", type=str, required=True,
                        help="Path of matrix file.")
