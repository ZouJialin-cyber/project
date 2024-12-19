from typing import List, Any, Tuple, Union
from pathlib import Path
import os

import numpy as np

from cellbin2.image import cbimwrite
from cellbin2.modules.metadata import ProcFile
from cellbin2.modules.extract.base import FeatureExtract
from cellbin2.modules import naming, run_state
from cellbin2.matrix.matrix import cMatrix
from cellbin2.contrib.tissue_segmentor import TissueSegInputInfo
from cellbin2.utils.pro_monitor import process_decorator


class MatrixFeatureExtract(FeatureExtract):
    def __init__(self, image_file: ProcFile, output_path: str, m_naming: naming.DumpMatrixFileNaming):
        super(MatrixFeatureExtract, self).__init__(image_file, output_path)
        self._point00: Tuple[int, int] = (0, 0)  # xy,相对于芯片而不是矩阵的位置坐标
        # self._naming = run_state.DumpMatrixFileNaming(image_file.file_path)
        self._naming = m_naming
        self._mat = None

    @process_decorator('GiB')
    def extract4stitched(self, detect_feature: bool = True):
        cm = cMatrix()
        cm.read(file_path=Path(self._image_file.file_path))
        if detect_feature:
            cm.detect_feature(ref=self._param_chip.fov_template,
                              chip_size = min(self._param_chip.chip_specif))
            self._template = cm.template
            self._chip_box = cm.chip_box
            np.savetxt(self._naming.matrix_template, self._template.template_points)
        self._mat = cm.heatmap
        cbimwrite(self._naming.heatmap, self._mat)

    @process_decorator('GiB')
    def extract4matrix(self, m_naming: naming.DumpPipelineFileNaming):
        # 细胞mask提矩阵是否考虑组织mask，需确认下
        from cellbin2.matrix.matrix import save_cell_bin_data, save_tissue_bin_data
        cell_mask_path = m_naming.final_nuclear_mask
        tissue_mask_path = m_naming.final_tissue_mask
        cell_correct_mask_path = m_naming.final_cell_mask
        c_inp = None
        if Path(tissue_mask_path).exists():
            save_tissue_bin_data(
                self._image_file.file_path,
                str(self._naming.tissue_bin_matrix),
                tissue_mask_path,
            )
            c_inp = self._naming.tissue_bin_matrix
        if c_inp is None:
            c_inp = self._image_file.file_path
        if Path(cell_mask_path).exists():
            save_cell_bin_data(
                c_inp,
                str(self._naming.cell_bin_matrix),
                cell_mask_path)
        if Path(cell_correct_mask_path).exists():
            save_cell_bin_data(
                c_inp,
                str(self._naming.cell_correct_bin_matrix),
                cell_correct_mask_path
            )

    @process_decorator('GiB')
    def tissue_segmentation(self, ):
        from cellbin2.contrib.tissue_segmentor import segment4tissue

        p = self._naming.tissue_mask
        if os.path.exists(p):
            return
        image_path = self._naming.heatmap
        tissue_input = TissueSegInputInfo(
            weight_path_cfg=self._config.tissue_segmentation,
            input_path=image_path,
            stain_type=self._image_file.tech,
        )
        tissue_mask_output = segment4tissue(tissue_input)
        tissue_mask = tissue_mask_output.tissue_mask
        cbimwrite(p, tissue_mask, compression=False)

    @process_decorator('GiB')
    def cell_segmentation(self, ):
        from cellbin2.contrib.cell_segmentor import segment4cell

        p = self._naming.cell_mask
        if os.path.exists(p): return
        image_path = self._naming.heatmap
        cell_mask = segment4cell(input_path=image_path,
                                 cfg=self._config.cell_segmentation,
                                 s_type=self._image_file.tech,
                                 fast=False, gpu=0)
        cbimwrite(p, cell_mask, compression=False)


def main():
    from cellbin2.modules.metadata import default_matrix
    from cellbin2.utils.config import Config
    from cellbin2.utils.stereo_chip import StereoChip

    config_file = r'E:\03.users\liuhuanlin\02.code\cellbin2\cellbin\config\cellbin.yaml'
    weights_root = r'E:\03.users\liuhuanlin\01.data\cellbin2\weights'

    image_file = default_matrix(file_path=r'E:/03.users/liuhuanlin/01.data/cellbin2/input/D04167E2/D04167E2.gem.gz',
                               tech_type='Transcriptomics')
    mfe = MatrixFeatureExtract(image_file, output_path=r'E:\03.users\liuhuanlin\01.data\cellbin2\output2')
    s = StereoChip()
    s.parse_info(chip_no='D04167E2')
    mfe.set_chip_param(s)
    mfe.set_config(Config(config_file, weights_root))
    # mfe.extract4stitched()
    mfe.tissue_segmentation()
    # mfe.cell_segmentation()


if __name__ == '__main__':
    main()