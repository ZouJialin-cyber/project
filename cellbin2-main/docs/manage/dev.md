```shell
H&E
PYTHONPATH=/media/Data/dzh/code/cellbin2 python /media/Data/dzh/code/cellbin2/cellbin2/cellbin_pipeline.py \
-c C04042E3 \
-i /media/Data/dzh/data/cellbin2/demo_data/C04042E3/C04042E3_fov_stitched.tif \
-s HE \
-m /media/Data/dzh/data/cellbin2/demo_data/C04042E3/C04042E3.raw.gef \
-w /media/Data/dzh/data/cellbin2/weights \
-o /media/Data/dzh/data/cellbin2/test/C04042E3_demo \
-r 

DAPI+IF
PYTHONPATH=/media/Data/dzh/code/cellbin2 python /media/Data/dzh/code/cellbin2/cellbin2/cellbin_pipeline.py \
-c A03599D1 \
-i /media/Data/dzh/data/cellbin2/demo_data/A03599D1/A03599D1_DAPI_fov_stitched.tif \
-if /media/Data/dzh/data/cellbin2/demo_data/A03599D1/A03599D1_IF_fov_stitched.tif \
-s DAPI \
-m /media/Data/dzh/data/cellbin2/demo_data/A03599D1/A03599D1.raw.gef \
-pr /media/Data/dzh/data/cellbin2/demo_data/A03599D1/A03599D1.protein.raw.gef \
-w /media/Data/dzh/data/cellbin2/weights \
-o /media/Data/dzh/data/cellbin2/test/A03599D1_demo \
-r 

ssDNA
PYTHONPATH=/media/Data/dzh/code/cellbin2 python /media/Data/dzh/code/cellbin2/cellbin2/cellbin_pipeline.py \
-c SS200000135TL_D1 \
-i /media/Data/dzh/data/cellbin2/demo_data/SS200000135TL_D1/SS200000135TL_D1_fov_stitched_ssDNA.tif \
-s ssDNA \
-m /media/Data/dzh/data/cellbin2/demo_data/SS200000135TL_D1/SS200000135TL_D1.raw.gef \
-w /media/Data/dzh/data/cellbin2/weights \
-o /media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo \
-r 

DAPI+mIF
PYTHONPATH=/media/Data/dzh/code/cellbin2 python /media/Data/dzh/code/cellbin2/cellbin2/cellbin_pipeline.py \
-c SS200000045_M5 \
-i /media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5_fov_stitched.tif \
-if /media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5_ATP_IF_fov_stitched.tif,/media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5_CD31_IF_fov_stitched.tif,/media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5_NeuN_IF_fov_stitched.tif \
-s DAPI \
-m /media/Data/dzh/data/cellbin2/demo_data/SS200000045_M5/SS200000045_M5.raw.gef \
-w /media/Data/dzh/data/cellbin2/weights \
-o /media/Data/dzh/data/cellbin2/test/SS200000045_M5_demo \
-r
```

ztron
```shell
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/storeData/USER/data/01.CellBin/00.user/dengzhonghan/code/cellbin2test/cellbin2 python \
/storeData/USER/data/01.CellBin/00.user/dengzhonghan/code/cellbin2test/cellbin2/cellbin2/cellbin_pipeline.py \
-c SS200000135TL_D1 \ 
-i /storeData/USER/data/01.CellBin/00.user/dengzhonghan/data/cellbin2/test_data/demo_data/SS200000135TL_D1/SS200000135TL_D1_fov_stitched_ssDNA.tif \
-s ssDNA \ 
-m /storeData/USER/data/01.CellBin/00.user/dengzhonghan/data/cellbin2/test_data/demo_data/SS200000135TL_D1/SS200000135TL_D1.raw.gef \
-w /storeData/USER/data/01.CellBin/00.user/dengzhonghan/weights \
-o /storeData/USER/data/01.CellBin/00.user/dengzhonghan/data/cellbin2/test_data/test_result/20241113
-r
```


test
```shell

pip install pytest-html
python3 -m pytest # 一定要这样执行，不然会找不到cellbin
pytest test/test_cellbin_pipeline.py --html=/media/Data1/user/dengzhonghan/data/cellbin2/auto_test/0.0.1/report/0.01.html --self-contained-html
```

