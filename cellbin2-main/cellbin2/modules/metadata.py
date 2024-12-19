import os.path

from prettytable import PrettyTable
from cellbin2.utils import clog
import json
import copy
import pprint

from typing import List, Dict, Type, Any, Tuple, Optional
from objtyping import objtyping
import pathlib
from pydantic import BaseModel, model_validator

from cellbin2.contrib import chip_detector
from cellbin2.contrib import clarity
from cellbin2.utils.common import TechType
from cellbin2.utils.config import Config
from cellbin2.utils import obj2dict, dict2json
from cellbin2.utils.ipr import bPlaceHolder, fPlaceHolder, iPlaceHolder, sPlaceHolder


def map_kit2type(kit_name: str):
    if 'Stereo-seq' in kit_name:
        return TechType.Transcriptomics.name
    if 'Stereo-CITE' in kit_name:
        return TechType.Protein.name
    return TechType.UNKNOWN


class ProcRegistration(BaseModel):
    fixed_image: int = 1
    trackline: bool = True
    reuse: int = -1


class ProcFile(BaseModel):
    file_path: str
    tech_type: str
    chip_detect: bool = True
    quality_control: bool = True
    tissue_segmentation: bool = True
    cell_segmentation: bool = True
    correct_r: int = 10
    channel_align: int = 0
    registration: ProcRegistration = ProcRegistration()
    _supported_matrix = ['.gef', '.gem.gz']
    _supported_image = ['.tif', '.tiff', '.TIF', '.TIFF']

    @property
    def tech(self, ) -> TechType:
        return TechType[self.tech_type]

    @property
    def is_image(self, ) -> bool:
        if pathlib.Path(self.file_path).suffix in self._supported_image:
            return True
        else:
            return False

    @property
    def is_matrix(self, ) -> bool:
        if pathlib.Path(self.file_path).suffix in self._supported_matrix:
            return True
        else:
            return False

    @property
    def tag(self, ):
        return os.path.basename(self.file_path).split('.')[0]

    @property
    def is_exists(self, ):
        return pathlib.Path.exists(pathlib.Path(self.file_path))

    def get_group_name(self, sn, pattern='IF'):
        """
        两种情况：
            1. SN_Protein_IF_xxx.tif
            2. SN_IF_xxx.tif
        """
        if pattern in self.tag:
            start_index = self.tag.find(sn) + len(sn)
            end_index = self.tag.find(pattern)
            middle = self.tag[start_index: end_index]
            middle = middle.strip("_")
            if middle:
                g_name = middle + "_" + self.tech.name
            else:
                g_name = self.tech.name
        else:
            g_name = self.tech.name
        return g_name

    def valid_check(self, cfg: Config):
        # chip_detect valid check
        if self.chip_detect and self.tech not in chip_detector.SUPPORTED_STAIN_TYPE:
            clog.warning(
                f"{self.tech.name} not in chip_detect supporting list "
                f"{[st.name for st in chip_detector.SUPPORTED_STAIN_TYPE]}. "
                f"Change {self.file_path} chip_detect to false"
            )
            self.chip_detect = False
        # quality_control(clarity) valid check
        if self.quality_control and self.tech not in clarity.SUPPORTED_STAIN_TYPE:
            clog.warning(
                f"{self.tech.name} not in quality_control supporting list "
                f"{[st.name for st in clarity.SUPPORTED_STAIN_TYPE]}. "
                f"Change quality_control to false"
            )
            self.quality_control = False
        # tissue_segmentation valid check
        if self.tissue_segmentation and self.tech not in cfg.tissue_segmentation.supported_model:
            clog.warning(
                f"{self.tech.name} not in tissue_segmentation supporting list "
                f"{[st.name for st in cfg.tissue_segmentation.supported_model]}. "
                f"Change {self.file_path} tissue_segmentation to false"
            )
            self.tissue_segmentation = False
        # cell_segmentation valid check
        if self.cell_segmentation and self.tech not in cfg.cell_segmentation.supported_model:
            clog.warning(
                f"{self.tech.name} not in cell_segmentation supporting list "
                f"{[st.name for st in cfg.cell_segmentation.supported_model]}. "
                f"Change {self.file_path} cell_segmentation to false"
            )
            self.cell_segmentation = False


def default_image(file_path: str, tech_type: str, clarity: bool = False) -> ProcFile:
    pif = ProcFile()

    pif.file_path = file_path
    pif.tech_type = tech_type
    pif.chip_detect = True
    pif.quality_control = clarity and True or False
    pif.tissue_segmentation = True
    pif.cell_segmentation = True
    pif.correct_r = 10
    pif.channel_align = -1
    pif.registration.fixed_image = 1
    pif.registration.trackline = True  # 固定图：1-默认是转录组矩阵

    return pif


def default_matrix(file_path: str, tech_type: str, stain_type: str) -> ProcFile:
    pif = ProcFile()
    pif.file_path = file_path
    pif.tech_type = tech_type
    pif.chip_detect = True
    pif.quality_control = False
    pif.tissue_segmentation = False
    pif.cell_segmentation = False
    pif.correct_r = 0
    pif.channel_align = -1
    pif.registration.fixed_image = -1  # 固定图
    pif.registration.trackline = True  # 固定图
    if stain_type == 'Null':
        pif.tissue_segmentation = True
        pif.cell_segmentation = True
        pif.chip_detect = False
        pif.registration.trackline = False  # 固定图

    return pif


class ProcMolecularFile(BaseModel):
    exp_matrix: int
    cell_mask: List[int]
    extra_method: str = ''  # 额外的分子归类方法，当前没有


class ProcAnalysisParam(BaseModel):
    report: bool = True
    annotation: bool = False


class ProcParam(BaseModel):
    image_process: Dict[str, ProcFile] = {}
    molecular_classify: Dict[str, ProcMolecularFile] = {}
    analysis: ProcAnalysisParam = ProcAnalysisParam()

    @staticmethod
    def print_files_info(files: dict, mode: str = 'imageQC'):
        a_table = PrettyTable()
        if mode == 'imageQC':
            a_table.field_names = ['ID', 'file_name', 'stain_type', 'template', 'detect_chip', 'clarity', 'align']
            for idx, pif in files.items():
                a_table.add_row([idx, os.path.basename(pif.file_path), pif.tech_type, pif.registration.trackline,
                                 pif.chip_detect, pif.quality_control, pif.channel_align])
        elif mode == 'Scheduler':
            a_table.field_names = ['ID', 'file_name', 'stain_type', 'registration', 'tissue_seg', 'cell_seg', 'align']
            for idx, pif in files.items():
                a_table.add_row([idx, os.path.basename(pif.file_path), pif.tech_type, pif.registration.fixed_image,
                                 pif.tissue_segmentation, pif.cell_segmentation, pif.channel_align])

        clog.info('{} files to be processed, the information as follows,\n{}'.format(len(files), a_table))

    def check_inputs(self, cfg: Config):
        for i, v in self.image_process.items():
            v.valid_check(cfg=cfg)

    def get_image_files(self, do_image_qc: bool = True,
                        do_scheduler: bool = True,
                        cheek_exists: bool = False) -> Dict[int, ProcFile]:
        images = copy.copy(self.image_process)

        if do_image_qc:
            images = {
                idx: image for idx, image in images.items()
                if image.is_image and
                   (image.chip_detect or
                    image.quality_control or
                    image.channel_align != -1 or
                    image.registration.trackline)
            }

        if do_scheduler:
            images = {
                idx: image for idx, image in images.items()
                if images[idx].tissue_segmentation or
                   images[idx].cell_segmentation or
                   images[idx].channel_align != -1
            }
            # add matrix
            for idx, matrix in self.molecular_classify.items():
                images[matrix.exp_matrix] = self.image_process[str(matrix.exp_matrix)]

        if cheek_exists:
            images = {int(idx): image for idx, image in images.items() if image.is_exists}

        return images

    def get_molecular_classify(self):
        files = {int(idx): f for idx, f in self.molecular_classify.items()}
        return files


def read_param_file(file_path: str, cfg: Config, out_path: Optional[str] = None) -> ProcParam:
    """
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as fd:
        dct = json.load(fd)
    pp = ProcParam(**dct)
    pp.check_inputs(cfg=cfg)
    if out_path is not None:
        dict2json(pp.model_dump(), json_path=out_path)
    return pp


def main():
    param_file = '/media/Data/dzh/code/cellbin2/cellbin2/config/default_param.json'
    cfg_file = "/media/Data/dzh/code/cellbin2/cellbin2/config/cellbin.yaml"
    cfg = Config(cfg_file)
    out = "/media/Data/dzh/data/cellbin2/tmp"
    im_path = "/media/Data/dzh/data/cellbin2/demo_data/C04042E3/C04042E3_fov_stitched.tif"
    track_s_type = "HE"
    matrix_path = "/media/Data/dzh/data/cellbin2/demo_data/C04042E3/C04042E3.raw.gef"
    # if_path = "/media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5_ATP_IF_fov_stitched.tif," \
    #           "/media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5_CD31_IF_fov_stitched.tif," \
    #           "/media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5_NeuN_IF_fov_stitched.tif"
    with open(param_file, 'r') as fd:
        dct = json.load(fd)
    pp = read_param_file(file_path=param_file, cfg=cfg, out_path=out)
    template = getattr(pp.image_process, track_s_type)
    template.file_path = im_path

    trans_tp = pp.image_process['Transcriptomics']
    trans_tp.file_path = matrix_path
    print()
    # print(pp.analysis.report)


if __name__ == '__main__':
    main()
