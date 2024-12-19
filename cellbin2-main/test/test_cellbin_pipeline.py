import shutil
import sys
import os
import pytest
import traceback

CURR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# CB_PATH = os.path.join(CURR_PATH,)
print(CURR_PATH)
sys.path.append(CURR_PATH)
from cellbin2.cellbin_pipeline import pipeline
import cellbin2

WEIGHTS_ROOT = "/media/Data1/user/dengzhonghan/data/cellbin2/weights"
TEST_OUTPUT_DIR = "/media/Data1/user/dengzhonghan/data/cellbin2/auto_test"
DEMO_DATA_DIR = "/media/Data1/user/dengzhonghan/data/cellbin2/demo_data"
TEST_DATA = [
    (
        # DAPI + mIF
        "SS200000045_M5",  # sn
        "SS200000045_M5/SS200000045_M5_fov_stitched.tif",  # DAPI, HE, ssDNA path
        "SS200000045_M5/SS200000045_M5_ATP_IF_fov_stitched.tif,"
        "SS200000045_M5/SS200000045_M5_CD31_IF_fov_stitched.tif,"
        "SS200000045_M5_NeuN_IF_fov_stitched.tif",  # IF path
        "DAPI",  # stain_type (DAPI, HE, ssDNA)
        "SS200000045_M5/SS200000045_M5.raw.gef",  # transcriptomics gef path
        "",  # protein gef path
    ),
    (
        # H&E
        "C04042E3",
        "C04042E3/C04042E3_fov_stitched.tif",
        "",
        "HE",
        "C04042E3/C04042E3.raw.gef",
        ""
     ),
    (
        # ssDNA
        "SS200000135TL_D1",
        "SS200000135TL_D1/SS200000135TL_D1_fov_stitched_ssDNA.tif",
        "",
        "ssDNA",
        "SS200000135TL_D1/SS200000135TL_D1.raw.gef",
        ""
    ),
    (
        # DAPI + IF
        "A03599D1",
        "A03599D1/A03599D1_DAPI_fov_stitched.tif",
        "A03599D1/A03599D1_IF_fov_stitched.tif",
        "DAPI",
        "A03599D1/A03599D1.raw.gef",
        "A03599D1/A03599D1.protein.raw.gef",
    )
]


class TestPipelineMain:

    # test script mode
    @pytest.mark.parametrize("sn, im_path, if_path, s_type, trans_gef, p_gef", TEST_DATA)
    def test_run(self, sn, im_path, if_path, s_type, trans_gef, p_gef):
        print(sn, im_path, if_path, s_type, trans_gef, p_gef)
        git_commit = os.getenv('GITHUB_SHA')
        if git_commit is not None:
            cur_test_out = os.path.join(TEST_OUTPUT_DIR, git_commit)
        else:
            cur_test_out = os.path.join(TEST_OUTPUT_DIR, cellbin2.__version__)
        print(f"Test results will be saved at {cur_test_out}")
        os.makedirs(cur_test_out, exist_ok=True)
        cur_out = os.path.join(cur_test_out, sn)
        im_path = os.path.join(DEMO_DATA_DIR, im_path)
        if if_path != "":
            if_path = os.path.join(DEMO_DATA_DIR, if_path)
        trans_gef = os.path.join(DEMO_DATA_DIR, trans_gef)
        if p_gef != "":
            p_gef = os.path.join(DEMO_DATA_DIR, p_gef)
        try:
            pipeline(
                chip_no=sn,
                input_image=im_path,
                if_image=if_path,
                stain_type=s_type,
                param_file="",
                output_path=cur_out,
                matrix_path=trans_gef,
                protein_matrix_path=p_gef,
                kit="",
                if_report=True,
                weights_root=WEIGHTS_ROOT,
            )
        except Exception as e:
            print(traceback.print_exc())
        # shutil.rmtree(cur_test_out)
