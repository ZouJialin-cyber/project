import os.path
from typing import Optional, Any, Tuple
from pydantic import BaseModel, Field
import cv2
import numpy as np
import tifffile
from skimage.measure import label
from skimage.morphology import remove_small_objects

from cellbin2.image import CBImage
from cellbin2.contrib.param import ChipBoxInfo
from cellbin2.modules.metadata import TechType
from cellbin2.contrib.tissue_segmentor import TissueSegParam
from cellbin2.utils import clog
from cellbin2.utils.pro_monitor import process_decorator


class MaskManagerInfo(BaseModel):
    tissue_mask: Any = Field(None, description='组织分割mask')
    cell_mask: Any = Field(None, description='细胞分割mask')
    chip_box: ChipBoxInfo = Field(None, description='芯片框')
    stain_type: TechType = Field(None, description='染色类型')
    method: int = Field(None, description='使用方法，0：稳定版，1：研发版')


class BestTissueCellMaskInfo(BaseModel):
    best_tissue_mask: Any = Field(None, description='融合后的组织分割mask')
    best_cell_mask: Any = Field(None, description='融合后的细胞分割mask')


class BestTissueCellMask:
    init_flag = True

    @staticmethod
    def init(input_data: MaskManagerInfo) -> bool:

        if input_data.cell_mask is None or input_data.tissue_mask is None:
            clog.error(f"init failed-->cell mask or tissue mask is None")
            return False

        if input_data.cell_mask.shape != input_data.tissue_mask.shape:
            clog.error(f"init failed-->the shape of the cell mask and tissue mask do not match")
            return False

        if input_data.stain_type is None:
            clog.error(f"init failed-->stain type is None")
            return False

        if input_data.method is None:
            clog.error(f"init failed-->method is None")
            return False

        clog.info('init success')
        return True

    @staticmethod
    def crop_chip_mask(chip_box: ChipBoxInfo, tissue_mask: np.ndarray,
                       cell_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1, y1 = chip_box.LeftTop
        x2, y2 = chip_box.RightTop
        x3, y3 = chip_box.RightBottom
        x4, y4 = chip_box.LeftBottom
        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        points = np.array(points).astype(np.int32)
        full_zeros_mask = np.zeros_like(tissue_mask)
        cv2.fillConvexPoly(full_zeros_mask, points, 255)

        cell_mask[full_zeros_mask == 0] = 0
        tissue_mask[full_zeros_mask == 0] = 0

        clog.info('tissue mask and cell mask update with chip box')
        return tissue_mask, cell_mask

    @staticmethod
    def best_cell_mask(tissue_mask: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
        clog.info(f"calling function: best_cell_mask() ")
        cell_mask_filter = cv2.bitwise_and(cell_mask, tissue_mask)
        return cell_mask_filter

    @staticmethod
    def best_tissue_mask(tissue_mask: np.ndarray, cell_mask: np.ndarray, kernel_size: int) -> np.ndarray:
        clog.info(f"calling function: best_tissue_mask() ")
        clog.info(f"cell mask dilate kernel size:{kernel_size}")
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if cell_mask.shape != tissue_mask.shape:
            clog.error('the shape of the cell mask and tissue mask do not match')
            return tissue_mask

        cell_mask_filter = cv2.bitwise_and(cell_mask, tissue_mask)

        dilated_cell_mask = cv2.dilate(cell_mask_filter, kernel, iterations=1)

        dilated_cell_mask = dilated_cell_mask > 0
        min_size = int(np.sum(kernel) * 2.25)
        filter_mask = remove_small_objects(dilated_cell_mask, min_size=min_size)
        filter_mask = np.uint8(filter_mask)
        filter_mask[filter_mask > 0] = 1

        tmp_mask = np.zeros_like(filter_mask, dtype=np.uint8)
        contours, _ = cv2.findContours(filter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(img=tmp_mask, pts=[cnt], color=1)

        result_tissue_mask = cv2.bitwise_and(tissue_mask, tmp_mask)
        return result_tissue_mask

    @classmethod
    def get_best_tissue_cell_mask(cls, input_data: MaskManagerInfo) -> BestTissueCellMaskInfo:

        """

        :param input_data: MaskManagerInfo结构体类型，入参包含tissue mask, cell mask, method(目前可选0，1,0代表默认操作，
                            使用组织分割过滤细胞分割，1代表研发版本，使用所有信息输出过滤后的细胞分割和组织分割）,
                            stain type(TechType枚举类型）, chip box(ChipBoxInfo类型，可选）
        :return: BestTissueCellMaskInfo结构提类型，出参包含优化后的cell mask和优化后的tissue mask
        """
        cls.init_flag = False

        tissue_mask = input_data.tissue_mask
        cell_mask = input_data.cell_mask
        chip_box = input_data.chip_box
        stain_type = input_data.stain_type
        method = input_data.method

        kernel_size = 200
        crop_tissue_mask = None
        crop_cell_mask = None

        output_data = BestTissueCellMaskInfo()

        clog.info(f'processing stain type:{stain_type}')
        clog.info(f"received parameter method: {method}")
        cls.init_flag = cls.init(input_data)
        if not cls.init_flag:
            clog.info(f"return input tissue mask and cell mask")
            output_data.best_tissue_mask = tissue_mask
            output_data.best_cell_mask = cell_mask
            return output_data

        if method == 1 and stain_type == TechType.DAPI:
            clog.error(f"stain type: {stain_type} do not support method: {method}")
            clog.info(f"execute method 0 and return input tissue mask and best cell mask")
            output_data.best_tissue_mask = tissue_mask
            # output_data.best_cell_mask = cell_mask
            output_data.best_cell_mask = cls.best_cell_mask(tissue_mask=tissue_mask, cell_mask=cell_mask)
            return output_data


        if stain_type == TechType.HE:
            kernel_size = 250

        tissue_mask[tissue_mask > 0] = 1
        cell_mask[cell_mask > 0] = 1

        crop_tissue_mask, crop_cell_mask = tissue_mask, cell_mask  # TODO: hdd check this
        if chip_box:
            if chip_box.IsAvailable:
                crop_tissue_mask, crop_cell_mask = cls.crop_chip_mask(chip_box, tissue_mask, cell_mask)
            else:
                clog.warning('chip box is not available')
        else:
            clog.warning('chip box is None')

        if method == 0:
            output_data.best_cell_mask = cls.best_cell_mask(tissue_mask=tissue_mask, cell_mask=cell_mask)
            output_data.best_tissue_mask = tissue_mask
        elif method == 1:
            output_data.best_cell_mask = cls.best_cell_mask(tissue_mask=crop_tissue_mask, cell_mask=crop_cell_mask)
            output_data.best_tissue_mask = cls.best_tissue_mask(tissue_mask=crop_tissue_mask, cell_mask=crop_cell_mask,
                                                                kernel_size=kernel_size)
        else:
            clog.error(f'method only support 0 or 1, method:{method}')
            clog.info(f"return input tissue mask and cell mask")
            output_data.best_tissue_mask = tissue_mask
            output_data.best_cell_mask = cell_mask
            return output_data

        return output_data


def instance2semantics(ins: np.ndarray) -> np.ndarray:
    """
    :param ins: 实例mask（0-N）
    :return: 语义mask（0-1）
    """
    ins_ = ins.copy()
    h, w = ins_.shape[:2]
    tmp0 = ins_[1:, 1:] - ins_[:h - 1, :w - 1]
    ind0 = np.where(tmp0 != 0)

    tmp1 = ins_[1:, :w - 1] - ins_[:h - 1, 1:]
    ind1 = np.where(tmp1 != 0)
    ins_[ind1] = 0
    ins_[ind0] = 0
    ins_[np.where(ins_ > 0)] = 1
    return np.array(ins_, dtype=np.uint8)


@process_decorator('GiB')
def merge_cell_mask(
        nuclear_mask: np.ndarray,
        membrane_mask: np.ndarray,
        conflict_cover: str = "nuclear"
) -> CBImage:
    """
    :param nuclear_mask: 原始nuclear_mask
        -- 实例格式（背景是0，然后各个细胞赋值一个ID），图像最大值是细胞的个数
        -- 语义格式（01数组）

    :param membrane_mask: 原始membrane_mask
        -- 实例格式（背景是0，然后各个细胞赋值一个ID），图像最大值是细胞的个数
        -- 语义格式（01数组）

    :param conflict_cover: 冲突主体 nuclear | membrane

    :return: 合并mask
    """
    if len(np.unique(nuclear_mask)) != 2:
        nuclear_mask_sem = instance2semantics(nuclear_mask)
        if conflict_cover == "membrane":
            nuclear_mask_ins = nuclear_mask.copy()
    else:
        nuclear_mask_sem = nuclear_mask.copy()
        if conflict_cover == "membrane":
            nuclear_mask_ins = label(nuclear_mask, connectivity=1)
    del nuclear_mask

    if len(np.unique(membrane_mask)) != 2:
        membrane_mask_sem = instance2semantics(membrane_mask)
        if conflict_cover == "nuclear":
            membrane_mask_ins = membrane_mask.copy()
    else:
        membrane_mask_sem = membrane_mask.copy()
        if conflict_cover == "nuclear":
            membrane_mask_ins = label(membrane_mask, connectivity=1)
    del membrane_mask

    mask_ = np.uint8(nuclear_mask_sem & membrane_mask_sem)

    if conflict_cover == "nuclear":
        remove_cell_id = np.unique(mask_ * membrane_mask_ins)
        del mask_
        membrane_mask_ins = np.uint32(~np.isin(membrane_mask_ins, remove_cell_id)) * membrane_mask_ins
        membrane_mask_sem = instance2semantics(membrane_mask_ins)
        del membrane_mask_ins
    else:
        remove_cell_id = np.unique(mask_ * nuclear_mask_ins)
        del mask_
        nuclear_mask_ins = np.uint32(~np.isin(nuclear_mask_ins, remove_cell_id)) * nuclear_mask_ins
        nuclear_mask_sem = instance2semantics(nuclear_mask_ins)
        del nuclear_mask_ins

    _ = membrane_mask_sem * 2 + nuclear_mask_sem * 1

    return CBImage(_)


if __name__ == '__main__':
    from cellbin2.image import cbimread

    chip_box = ChipBoxInfo()

    # HE测试数据细胞分割mask和组织分割mask路径
    cell_mask_path = r"F:\01.users\hedongdong\cellbin2_test\cell_mask\C04042E3_HE_regist_v3_mask.tif"
    tissue_mask_path = r"F:\01.users\hedongdong\cellbin2_test\result_mask\C04042E3_HE_regist.tif"

    chip_point = {
        'left_top': [2183, 2190],
        'right_top': [22183, 2199],
        'right_bottom': [22189, 22195],
        'left_bottom': [2181, 22214]
    }
    stain_type = TechType.DAPI
    method = 1

    # # ssDNA测试数据细胞分割mask和组织分割mask路径
    # cell_mask_path = r"F:\01.users\hedongdong\cellbin2_test\cell_mask\A04535A4C6_fov_stitched_v3_mask.tif"
    # tissue_mask_path = r"F:\01.users\hedongdong\cellbin2_test\result_mask\A04535A4C6_fov_stitched.tif"
    #
    # chip_point = {
    #     'left_top': [1136, 1256],
    #     'right_top': [40680, 1396],
    #     'right_bottom': [40434, 60732],
    #     'left_bottom': [910, 60576]
    # }
    # stain_type = TechType.ssDNA

    chip_box.IsAvailable = False
    chip_box.LeftTop = chip_point['left_top']
    chip_box.RightTop = chip_point['right_top']
    chip_box.RightBottom = chip_point['right_bottom']
    chip_box.LeftBottom = chip_point['left_bottom']

    cell_mask = cbimread(cell_mask_path, only_np=True)
    tissue_mask = cbimread(tissue_mask_path, only_np=True)

    input_data = MaskManagerInfo()
    input_data.tissue_mask = tissue_mask
    input_data.cell_mask = cell_mask
    input_data.chip_box = chip_box
    input_data.method = method
    input_data.stain_type = stain_type

    best_tissue_cell_mask = BestTissueCellMask.get_best_tissue_cell_mask(input_data=input_data)
    # print(best_tissue_cell_mask.chip_box)

    output_cell_mask = best_tissue_cell_mask.best_cell_mask
    output_tissue_mask = best_tissue_cell_mask.best_tissue_mask

    output_cell_mask[output_cell_mask > 0] = 255
    output_tissue_mask[output_tissue_mask > 0] = 255

    tifffile.imwrite(os.path.join(r"F:\01.users\hedongdong\cellbin2_test\best_merge_result", 'cell_mask.tif'),
                     output_cell_mask, compression='zlib')
    tifffile.imwrite(os.path.join(r"F:\01.users\hedongdong\cellbin2_test\best_merge_result", 'tissue_mask.tif'),
                     output_tissue_mask, compression='zlib')
