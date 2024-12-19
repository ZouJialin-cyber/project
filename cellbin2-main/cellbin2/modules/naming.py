from pathlib import Path
from typing import Union, Callable
from cellbin2 import __version__


def my_property_dec(func: Callable[..., str]) -> property:
    def wrapper(self) -> Path:
        result = func(self)
        result = Path(self.save_dir).joinpath(result)
        return result
    wrapper.__doc__ = func.__doc__  # 保留原始函数的文档字符串
    return property(wrapper)


class DumpMatrixFileNaming:
    def __init__(self, sn: str, m_type, save_dir):
        self.sn = sn
        self.m_type = m_type
        self.save_dir = save_dir

    @my_property_dec
    def cell_bin_matrix(self, ):
        """CellBin gef"""
        return f'{self.sn}_{self.m_type}.cellbin.gef'

    @my_property_dec
    def cell_correct_bin_matrix(self, ):
        """CellBin correct gef"""
        return f'{self.sn}_{self.m_type}.adjusted.cellbin.gef'

    @my_property_dec
    def tissue_bin_matrix(self, ):
        """Tissuebin gef"""
        return f'{self.sn}_{self.m_type}.tissue.gef'

    @my_property_dec
    def matrix_template(self, ):
        """Track template on gene matrix"""
        return f'{self.sn}_{self.m_type}_matrix_template.txt'

    @my_property_dec
    def heatmap(self, ):
        """Gene matrix in tif format"""
        return f'{self.sn}_{self.m_type}.tif'

    @my_property_dec
    def cell_mask(self, ):
        """Cell mask on gene matrix"""
        return f'{self.sn}_{self.m_type}_mask.tif'

    @my_property_dec
    def cell_correct_mask(self, ):
        """Cell correct mask on gene matrix"""
        return f'{self.sn}_{self.m_type}_correct_mask.tif'

    @my_property_dec
    def tissue_mask(self, ):
        """Tissue mask on gene matrix"""
        return f'{self.sn}_{self.m_type}_tissue_cut.tif'


class DumpImageFileNaming:
    def __init__(self, sn, stain_type, save_dir):
        self.sn = sn
        self.stain_type = stain_type
        self.save_dir = save_dir

    @my_property_dec
    def stitch_image(self):
        """Stitched image"""
        return f'{self.sn}_{self.stain_type}_stitch.tif'

    @my_property_dec
    def transform_cell_mask(self, ):
        """Cell mask on transformed image"""
        return f'{self.sn}_{self.stain_type}_transform_mask.tif'

    @my_property_dec
    def transform_cell_mask_raw(self, ):
        """Cell raw mask on transformed image"""
        return f'{self.sn}_{self.stain_type}_transform_mask_raw.tif'

    # @property
    # def transform_cell_correct_mask(self, ):
    #     """Cell correct mask on transformed image"""
    #     return f'{self.sn}_{self.stain_type}_transform_mask_edm_dis_10.tif'

    @my_property_dec
    def transform_tissue_mask(self, ):
        """Tissue mask on transformed image"""
        return f'{self.sn}_{self.stain_type}_transform_tissue_cut.tif'

    @my_property_dec
    def transform_tissue_mask_raw(self, ):
        """Tissue raw mask on transformed image"""
        return f'{self.sn}_{self.stain_type}_transform_tissue_cut_raw.tif'

    @my_property_dec
    def transformed_image(self, ):
        """Transformed image"""
        return f'{self.sn}_{self.stain_type}_transformed.tif'

    @my_property_dec
    def transformed_template(self, ):
        """Track template on transformed image"""
        return f'{self.sn}_{self.stain_type}_transformed.txt'

    @my_property_dec
    def transformed_track_template(self, ):
        """Track detect result on transformed image"""
        return f'{self.sn}_{self.stain_type}_transformed_track.txt'

    @my_property_dec
    def register_template(self, ):
        """Track template on registered image"""
        return f'{self.sn}_{self.stain_type}_register.txt'

    @my_property_dec
    def register_track_template(self, ):
        """Track detect result on registered image"""
        return f'{self.sn}_{self.stain_type}_register_track.txt'

    @my_property_dec
    def cell_mask(self, ):
        """Cell mask on registered image"""
        return f'{self.sn}_{self.stain_type}_mask.tif'

    @my_property_dec
    def cell_mask_merged(self, ):  # 配准后的
        """Merged cell mask on registered image (nuclear + cell membrane)"""
        return f'{self.sn}_{self.stain_type}_mask_merged.tif'

    @my_property_dec
    def cell_mask_raw(self, ):  # 配准后的
        """Cell raw mask on registered image"""
        return f'{self.sn}_{self.stain_type}_mask_raw.tif'

    # @property
    # def cell_correct_mask(self, ):  # 配准后的
    #     """Cell correct mask on registered image"""
    #     return f'{self.sn}_{self.stain_type}_mask_edm_dis_10.tif'

    @my_property_dec
    def tissue_mask(self, ):
        """Tissue mask on registered image"""
        return f'{self.sn}_{self.stain_type}_tissue_cut.tif'

    @my_property_dec
    def tissue_mask_raw(self, ):
        """Tissue raw mask on registered image"""
        return f'{self.sn}_{self.stain_type}_tissue_cut_raw.tif'

    @my_property_dec
    def registration_image(self, ):
        """Registered image"""
        return f'{self.sn}_{self.stain_type}_regist.tif'

    @my_property_dec
    def stitched_template(self, ):
        """Track template on stitched image"""
        return f'{self.sn}_{self.stain_type}_stitch_template.txt'


class DumpPipelineFileNaming(object):
    """ 实现对CellBin输出文件的命名管理，输出文件内部关键字段的命名管理 """

    def __init__(self, chip_no: str, save_dir):
        self._chip_no = chip_no
        self.save_dir = save_dir

    @my_property_dec
    def report(self, ):
        """CellBin 2.0 report"""
        return 'CellBin_{}_report.html'.format(__version__)  # 报告文件

    @my_property_dec
    def metrics(self, ):
        """CellBin 2.0 Metrics"""
        return 'metrics.json'  # 统计指标文件

    @my_property_dec
    def ipr(self, ):
        """Image processing record"""
        return '{}.ipr'.format(self._chip_no)  # 图像记录文件

    @my_property_dec
    def rpi(self, ):
        """Recorded image processing (for visualization)"""
        return '{}.rpi'.format(self._chip_no)  # 图像金字塔文件

    @my_property_dec
    def final_nuclear_mask(self, ):
        """Final nuclear mask"""
        return '{}_mask.tif'.format(self._chip_no)

    @my_property_dec
    def final_cell_mask(self, ):
        """Final cell mask"""
        return '{}_cell_mask.tif'.format(self._chip_no)

    @my_property_dec
    def final_tissue_mask(self, ):
        """Final tissue mask"""
        return '{}_tissue_mask.tif'.format(self._chip_no)

    @my_property_dec
    def input_json(self):
        """CellBin 2.0 input params"""
        return f"{self._chip_no}_params.json"


def write_to_readme():
    import re
    from cellbin2.utils.common import TechType
    sn = 'A03599D1'
    save_dir = "/demo"
    im_type = TechType.DAPI
    im_type2 = TechType.IF
    m_type = TechType.Transcriptomics
    readme_p = "../../README.md"
    with open(readme_p, "r") as f:
        md_cont = f.read()
    pfn = DumpPipelineFileNaming(sn, save_dir=save_dir)
    ifn = DumpImageFileNaming(sn=sn, stain_type=im_type.name, save_dir=save_dir)
    ifn2 = DumpImageFileNaming(sn=sn, stain_type=im_type2.name, save_dir=save_dir)
    mfn = DumpMatrixFileNaming(sn=sn, m_type=m_type.name, save_dir=save_dir)
    all_ = [pfn, ifn, ifn2, mfn]
    table_md = "| File Name | Description |\n| ---- | ---- |\n"
    if_key_word = ["regist", "txt", "merged"]
    m_key_word = ["mask"]
    for n in all_:
        attrs = dir(n)
        for name in attrs:
            if name.startswith("__") or name.startswith("_") or not isinstance(n.__class__.__dict__.get(name),
                                                                               property):
                continue
            else:
                # isinstance(name, property)
                value = getattr(n, name)
                doc = getattr(n.__class__, name).__doc__
                value = value.name
                if isinstance(n, DumpImageFileNaming):
                    type_ = n.stain_type
                    if type_ not in [TechType.DAPI.name, TechType.ssDNA, TechType.HE.name] and\
                            any(kw in value for kw in if_key_word):
                        continue
                if isinstance(n, DumpMatrixFileNaming):
                    if any(kw in value for kw in m_key_word):
                        continue

                table_md += f"| {value} | {doc} |\n"

    print(table_md)
    updated_markdown_text = re.sub(r'# Outputs\n[\s\S]*?(?=\n#|\Z)', f'# Outputs\n\n{table_md}\n', md_cont)
    with open(readme_p, "w") as f:
        f.write(updated_markdown_text)


def main():
    # difn = DumpImageFileNaming(src_file='A03599D1_DAPI.tif')
    # print(difn.cell_correct_mask)
    # difn = DumpImageFileNaming(src_file=r'E:\03.users\liuhuanlin\01.data\cellbin2\output\A03599D1\A03599D1_DAPI.tif')
    # print(difn.cell_correct_mask)
    #
    # difn = DumpMatrixFileNaming(src_file='A03599D1_DAPI.raw.gef')
    # print(difn.cell_bin_matrix)
    # difn = DumpMatrixFileNaming(
    #     src_file=r'E:\03.users\liuhuanlin\01.data\cellbin2\output\A03599D1\A03599D1_DAPI.raw.gef')
    # print(difn.cell_bin_matrix)
    dmf = DumpMatrixFileNaming(sn="ddd", m_type="111", save_dir="/media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo_2")
    print()

if __name__ == '__main__':
    write_to_readme()
