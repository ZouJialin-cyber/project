#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : hedongdong1
# @Time    : 2024/11/28 15:34
# @File    : test_tissue_seg.py
# @annotation    :
import os.path

import pytest
import tifffile

from cellbin2.contrib.tissue_segmentor import segment4tissue, TissueSegInputInfo, TissueSegOutputInfo, TissueSegParam
from cellbin2.utils.common import TechType


TEST_DATA = [
    (
        # ssDNA
        r"F:\01.users\hedongdong\cellbin2_test_data\test_image",  # input image/dir path
        r"F:\01.users\hedongdong\cellbin2_test\tmp_pytest_result",  # output mask/dir path
        r"F:\01.users\hedongdong\cellbin2_test\model\tissueseg_bcdu_SDI_230523_tf.onnx",  # onnx file path
        "ssDNA", # stain type
        "onnx", # onnx mode or tf mode, currently only the onnx mode is supported
        "0" # GPU num, -1 represents the CPU
    ),
    (
        # DAPI
        r"F:\01.users\hedongdong\cellbin2_test_data\test_image",  # input image/dir path
        r"F:\01.users\hedongdong\cellbin2_test\tmp_pytest_result",  # output mask/dir path
        r"F:\01.users\hedongdong\cellbin2_test\model\tissueseg_bcdu_SDI_230523_tf.onnx",  # onnx file path
        "DAPI", # stain type
        "onnx", # onnx mode or tf mode, only the onnx mode is supported currently
        "0" # GPU num, -1 represents the CPU
    ),
    (
        # HE
        r"F:\01.users\hedongdong\cellbin2_test_data\test_image\C04042E3_HE_regist.tif", # input image/dir path
        r"F:\01.users\hedongdong\cellbin2_test\tmp_pytest_result\C04042E3_HE_regist.tif", # output mask/dir path
        r"F:\01.users\hedongdong\cellbin2_test\model\tissueseg_bcdu_H_20241018_tf.onnx", # onnx file path
        "HE", # stain type
        "onnx", # onnx mode or tf mode, currently only the onnx mode is supported
        "0" # GPU num, -1 represents the CPU
    ),
    (
        # IF do not need model
        r"F:\01.users\hedongdong\cellbin2_test_data\test_image\rna.tif", # input image/dir path
        r"F:\01.users\hedongdong\cellbin2_test\tmp_pytest_result\rna.tif", # output mask/dir path
        r"test",
        "IF", # stain type
        "onnx", # onnx mode or tf mode, currently only the onnx mode is supported
        "0" # GPU num, -1 represents the CPU
    ),
    (
        # IF do not need model
        r"F:\01.users\hedongdong\cellbin2_test_data\test_image\rna.tif", # input image/dir path
        r"F:\01.users\hedongdong\cellbin2_test\tmp_pytest_result\rna1.tif", # output mask/dir path
        r"F:\01.users\hedongdong\cellbin2_test\model\tissueseg_bcdu_rna_220909_tf.onnx",
        "Transcriptomics", # stain type
        "onnx", # onnx mode or tf mode, currently only the onnx mode is supported
        "0" # GPU num, -1 represents the CPU
    )
]

USR_STYPE_TO_INNER = {
        'ssDNA': TechType.ssDNA,
        'DAPI': TechType.DAPI,
        "HE": TechType.HE,
        "Transcriptomics": TechType.Transcriptomics,
        'Protein': TechType.Protein,
        'IF': TechType.IF
    }

class TestTissueSeg:
    @pytest.mark.parametrize("input_dir, output_dir, model_dir, stain_type, model_mode, gpu_num", TEST_DATA)
    def test_tissue_seg(self,
                        input_dir: str,
                        output_dir: str,
                        model_dir: str,
                        stain_type: str,
                        model_mode: str,
                        gpu_num: str
                        ):
        cfg = TissueSegParam()
        if stain_type != "IF":
            setattr(cfg, f"{stain_type}_weights_path", model_dir)
        print(f"info===> stain type: {stain_type}, set {stain_type} model path:{model_dir}")
        stain_type = USR_STYPE_TO_INNER[stain_type]
        if os.path.isdir(input_dir):
            assert os.path.isdir(output_dir), 'the input path is a folder, so the output path should also be a folder'
            for tmp in os.listdir(input_dir):
                input_path = os.path.join(input_dir, tmp)
                output_path = os.path.join(output_dir, tmp)
                input_data = TissueSegInputInfo(
                    weight_path_cfg=cfg,
                    input_path=input_path,
                    stain_type=stain_type
                )

                seg_result = segment4tissue(input_data=input_data)
                seg_mask = seg_result.tissue_mask
                print(seg_mask.shape)
                if seg_result.threshold_list:
                    print(*seg_result.threshold_list)
                seg_mask[seg_mask > 0] = 255
                tifffile.imwrite(output_path, seg_mask, compression='zlib')
        else:
            input_data = TissueSegInputInfo(
                weight_path_cfg=cfg,
                input_path=input_dir,
                stain_type=stain_type
            )

            seg_result = segment4tissue(input_data=input_data)
            seg_mask = seg_result.tissue_mask
            print(seg_mask.shape)
            if seg_result.threshold_list:
                print(*seg_result.threshold_list)
            seg_mask[seg_mask > 0] = 255
            tifffile.imwrite(output_dir, seg_mask, compression='zlib')
