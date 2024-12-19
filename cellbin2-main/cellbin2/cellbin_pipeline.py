import os
import sys
from copy import deepcopy

from cellbin2.utils.common import TechType
from cellbin2.modules import naming
from cellbin2.utils import clog
import cellbin2
from cellbin2.utils import ipr
from cellbin2.modules.metadata import read_param_file, ProcParam
from cellbin2.utils.config import Config
from cellbin2.modules.metrics import ImageSource
from cellbin2.utils import dict2json
from cellbin2.utils.common import KIT_VERSIONS
from cellbin2.utils.pro_monitor import process_decorator

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(CURR_PATH, r'config/cellbin.yaml')
CHIP_MASK_FILE = os.path.join(CURR_PATH, r'config/chip_mask.json')
DEFAULT_PARAM_FILE = os.path.join(CURR_PATH, r'config/default_param.json')
SUPPORTED_STAINED_TYPES = [TechType.ssDNA.name, TechType.DAPI.name, TechType.HE.name]


class CellBinPipeline(object):

    def __init__(self, config_file: str, chip_mask_file: str, weights_root: str) -> None:
        # alg
        self._chip_mask_file = chip_mask_file
        self._config_file = config_file
        self._weights_root = weights_root

        # data
        self._chip_no: str = ''
        self._input_image: str = ''
        self._stain_type: str = TechType.ssDNA.name
        self._param_file: str = ''
        self._output_path: str = ''
        self._matrix_path: str = ''
        self._kit: str = ''

        # naming
        self._naming: naming.DumpPipelineFileNaming

        # 内部需要的
        self.pp: ProcParam = ProcParam()
        self.config: Config

    def image_quality_control(self, ):
        """ 完成图像 QC 流程 """
        from cellbin2.modules import image_qc
        if self._naming.ipr.exists():
            clog.info('Image QC has been done')
            return 0
        s_code = image_qc.image_quality_control(
            weights_root=self._weights_root,
            chip_no=self._chip_no,
            input_image=self._input_image,
            stain_type=self._stain_type,
            param_file=self._param_file,
            output_path=self._output_path
        )
        if s_code != 0:
            sys.exit(1)

    def image_analysis(self, ):
        """ 完成图像、配准、校准、分割、矩阵提取等分析流程 """
        from cellbin2.modules import scheduler
        if self._naming.rpi.exists():
            clog.info('scheduler has been done')
            return 0
        scheduler.scheduler_pipeline(weights_root=self._weights_root, chip_no=self._chip_no,
                                     input_image=self._input_image, stain_type=self._stain_type,
                                     param_file=self._param_file, output_path=self._output_path,
                                     matrix_path=self._matrix_path,
                                     ipr_path=self._naming.ipr, kit=self._kit)

    def metrics(self, ):
        """ 计算指标 """
        if self.pp.analysis.report:
            from cellbin2.modules import metrics
            if self._naming.metrics.exists():
                clog.info('Metrics step has been done')
                return
            ipr_file = str(self._naming.ipr)
            rpi_file = str(self._naming.rpi)
            # 图片类的输入
            ipr_r, channel_images = ipr.read(ipr_file)
            src_img_dict = {}
            for c_name, c_info in channel_images.items():
                c_pipeline_name = naming.DumpImageFileNaming(
                    sn=self._chip_no, stain_type=c_name,
                    save_dir=self._output_path
                )
                src_img_dict[c_name] = {}
                for filed_name, filed in ImageSource.__fields__.items():
                    if filed_name == "cell_correct_mask":
                        fp = getattr(self._naming, f"final_cell_mask")
                    else:
                        fp = getattr(c_pipeline_name, filed_name)
                    if not os.path.exists(fp):
                        fp = ""
                    src_img_dict[c_name][filed_name] = str(fp)

            config = Config(self._config_file, self._weights_root)
            pp = read_param_file(file_path=self._param_file, cfg=config)
            files = pp.get_image_files(do_image_qc=False, do_scheduler=True, cheek_exists=True)
            gene_matrix = pp.get_molecular_classify().items()
            matrix_dict = {}
            for idx, m in gene_matrix:
                matrix = files[m.exp_matrix]
                cur_m_type = matrix.tech.name
                if cur_m_type not in matrix_dict:
                    cur_m_name = naming.DumpMatrixFileNaming(sn=self._chip_no, m_type=cur_m_type, save_dir=self._output_path)
                    cur_m_src_files = metrics.MatrixArray(
                        tissue_bin_matrix=str(cur_m_name.tissue_bin_matrix),
                        cell_bin_matrix=str(cur_m_name.cell_bin_matrix),
                        cell_bin_adjusted_matrix=str(cur_m_name.cell_correct_bin_matrix),
                        bin1_matrix=str(matrix.file_path),
                        matrix_type=matrix.tech
                    )
                    matrix_dict[cur_m_type] = cur_m_src_files
            matrix_lists = list(matrix_dict.values())
            fs = metrics.FileSource(
                ipr_file=ipr_file, rpi_file=rpi_file, matrix_list=[matrix_lists[0]], sn=self._chip_no,
                image_dict=src_img_dict)  # TODO 蛋白矩阵没放进去
            metrics.calculate(param=fs, output_path=self._output_path)
            clog.info("Metrics generated")

    def export_report(self, ):
        if self.pp.analysis.report:
            """ 生成及导出报告 """
            from cellbin2.modules import report_m

            src_file_path = self._naming.metrics
            report_m.creat_report(matric_json=src_file_path, save_path=self._output_path)

    def usr_inp_to_param(self):
        if self._param_file == "":
            if self._input_image == "":
                raise Exception(f"the input image can not be empty if param file is empty")
            self.config = Config(self._config_file, self._weights_root)
            pp = read_param_file(file_path=DEFAULT_PARAM_FILE, cfg=self.config)
            new_pp = ProcParam()
            # track image (ssDNA, HE, DAPI)
            im_count = 0
            trans_exp_idx = -1
            protein_exp_idx = -1
            nuclear_cell_idx = -1

            template = pp.image_process[self._stain_type]
            template.file_path = self._input_image
            new_pp.image_process[str(im_count)] = template
            nuclear_cell_idx = im_count
            im_count += 1

            # 转录组矩阵
            trans_tp = pp.image_process[TechType.Transcriptomics.name]
            trans_tp.file_path = self._matrix_path
            new_pp.image_process[str(im_count)] = trans_tp
            trans_exp_idx = im_count
            im_count += 1

            # IF image if exists
            if self._if_image != "":
                if_im_paths = self._if_image.split(",")
                for idx, i_path in enumerate(if_im_paths):
                    if_template = deepcopy(pp.image_process[TechType.IF.name])
                    if_template.file_path = i_path
                    new_pp.image_process[str(idx + im_count)] = if_template
                    im_count += 1

            if self._protein_matrix_path != "":
                protein_tp = pp.image_process[TechType.Protein.name]
                protein_tp.file_path = self._protein_matrix_path
                new_pp.image_process[str(im_count)] = protein_tp
                protein_exp_idx = im_count

            # end of image part info parsing

            # 矩阵提取
            trans_m_tp = pp.molecular_classify[TechType.Transcriptomics.name]
            trans_m_tp.exp_matrix = trans_exp_idx
            trans_m_tp.cell_mask = [nuclear_cell_idx]
            new_pp.molecular_classify['0'] = trans_m_tp

            if protein_exp_idx != -1:
                protein_m_tp = pp.molecular_classify[TechType.Protein.name]
                protein_m_tp.exp_matrix = protein_exp_idx
                protein_m_tp.cell_mask = [nuclear_cell_idx]
                new_pp.molecular_classify['1'] = protein_m_tp
            new_pp.analysis.report = True if self._if_report else False
            param_f_p = self._naming.input_json
            dict2json(new_pp.dict(), json_path=param_f_p)
            self._param_file = param_f_p
            self.pp = new_pp

    def run(self, chip_no: str, input_image: str, if_image: str,
            stain_type: str, param_file: str,
            output_path: str, matrix_path: str, protein_matrix_path: str, kit: str, if_report):
        """ 全分析流程 """
        self._chip_no = chip_no
        self._input_image = input_image
        self._if_image = if_image
        self._stain_type = stain_type
        self._param_file = param_file
        self._output_path = output_path
        self._matrix_path = matrix_path
        self._protein_matrix_path = protein_matrix_path
        self._kit = kit
        self._if_report = if_report
        self._naming = naming.DumpPipelineFileNaming(chip_no=chip_no, save_dir=self._output_path)
        # self.pipe_run_state = PipelineRunState(self._chip_no, self._output_path)

        self.usr_inp_to_param()
        self.image_quality_control()  # 图像质控
        self.image_analysis()  # 图像分析
        self.metrics()  # 指标计算
        self.export_report()  # 生成报告


@process_decorator('GiB')
def pipeline(
        chip_no,
        input_image,
        if_image,
        stain_type,
        param_file,
        output_path,
        matrix_path,
        protein_matrix_path,
        kit,
        if_report,
        weights_root,
):
    os.makedirs(output_path, exist_ok=True)
    clog.log2file(output_path)
    clog.info(f"CellBin Version: {cellbin2.__version__}")
    stain_support = ('ssDNA', 'DAPI', 'HE', 'IF', 'Null')
    assert stain_type in stain_support, 'Parameter [stain_type] should in {}'.format(stain_support)

    if not os.path.isdir(weights_root):
        weights_root = os.path.join(CURR_PATH, 'weights')
    else:
        weights_root = weights_root

    cbp = CellBinPipeline(config_file=CONFIG_FILE, chip_mask_file=CONFIG_FILE, weights_root=weights_root)
    cbp.run(
        chip_no=chip_no,
        input_image=input_image,
        if_image=if_image,
        stain_type=stain_type,
        param_file=param_file,
        output_path=output_path,
        matrix_path=matrix_path,
        protein_matrix_path=protein_matrix_path,
        kit=kit,
        if_report=if_report
    )


def main(args, para):
    """
     :param weights_root: CNN权重文件本地存储目录路径
     :param chip_no: 样本芯片号
     :param input_image: 染色图本地路径
     :param stain_type: 染色图对应的染色类型
     :param param_file: 入参文件本地路径
     :param kit: 测序技术
     :param output_path: 输出文件本地存储目录路径
     :param matrix_path: 表达矩阵本地存储路径
     :return: int(状态码)
     """
    chip_no = args.chip_no
    input_image = args.input_image
    if_image = args.input_image_if
    stain_type = args.stain_type
    param_file = args.param_file
    output_path = args.output_path
    matrix_path = args.matrix_file
    protein_matrix_path = args.protein_matrix_file
    kit = args.kit
    weights_root = args.weights_root
    if_report = args.report

    pipeline(
        chip_no,
        input_image,
        if_image,
        stain_type,
        param_file,
        output_path,
        matrix_path,
        protein_matrix_path,
        kit,
        if_report,
        weights_root,
    )


if __name__ == '__main__':  # main()
    import argparse

    _VERSION_ = '2.0'
    usage_str = f"python {__file__} \n" \
                f"-c A03599D1 \n" \
                f"-i /media/Data/dzh/data/cellbin2/demo_data/A03599D1/A03599D1_DAPI_fov_stitched.tif \n" \
                f"-if /media/Data/dzh/data/cellbin2/demo_data/A03599D1/A03599D1_IF_fov_stitched.tif \n" \
                f"-s DAPI \n" \
                f"-m /media/Data/dzh/data/cellbin2/demo_data/A03599D1/A03599D1.raw.gef \n" \
                f"-pr /media/Data/dzh/data/cellbin2/demo_data/A03599D1/A03599D1.protein.raw.gef \n" \
                f"-w /media/Data/dzh/data/cellbin2/weights \n" \
                f"-o /media/Data/dzh/data/cellbin2/test/A03599D1_demo1_1 \n" \
                f"-r"

    parser = argparse.ArgumentParser(
        description="This is CellBin V2.0 Pipeline",
        usage=usage_str
    )
    parser.add_argument("-v", "--version", action="version", version=_VERSION_)
    parser.add_argument("-c", "--chip_no", action="store", type=str, required=True,
                        help="The SN of chip.")
    parser.add_argument("-i", "--input_image", action="store", type=str, default='',
                        help=f"The path of {{{','.join(SUPPORTED_STAINED_TYPES)}}} input file.")
    parser.add_argument("-s", "--stain_type", action="store", type=str,
                        choices=SUPPORTED_STAINED_TYPES,
                        help=f"The stain type of input image, choices are {{{','.join(SUPPORTED_STAINED_TYPES)}}}.")
    parser.add_argument("-if", "--input_image_if", action="store", type=str, default='',
                        help="The path of IF input file.")
    parser.add_argument("-m", "--matrix_file", action="store", type=str,
                        help="The path of transcriptomics matrix file.")
    parser.add_argument("-pr", "--protein_matrix_file", action="store", type=str, default="",
                        help="The path of protein matrix file.")
    parser.add_argument("-k", "--kit", action="store", type=str, choices=KIT_VERSIONS, help="Kit Type")
    parser.add_argument("-r", "--report", action="store_true", help="If run report.")
    parser.add_argument("-p", "--param_file", action="store", type=str,
                        default='', help="The path of input param file.")
    parser.add_argument("-w", "--weights_root", action="store", type=str, default="",
                        help="The weights root folder.")
    parser.add_argument("-o", "--output_path", action="store", type=str, required=True,
                        help="The results output folder.")
    parser.add_argument("-d", "--debug", action="store_true", default=True, help="Debug mode")

    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)
