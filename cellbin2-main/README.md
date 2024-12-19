<div align="center">
  <img src="docs/images/cellbin.png"><br/>
  <h1 align="center">
    cellbin2: A framework for generating single-cell gene expression data
  </h1>
</div>

## Installation
Linux
```shell
git clone https://github.com/STOmics/cellbin2
conda create --name cellbin2 python=3.8
python install.py
```


## Tutorials
### Default
***This mode will use the default parameters which stored in config/default_param.json***
```shell
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
-c C04042E3 \ # chip number
-i /media/Data/C04042E3_fov_stitched.tif \ # ssDNA, DAPI, HE data path
-s HE \ # stain type, (ssDNA, DAPI, HE)
-m /media/Data/C04042E3/C04042E3.raw.gef \ # Transcriptomics gef path
-w /media/Data/weights \ # weights save dir
-o /media/Data/C04042E3_demo \  # output dir
-r # if report 
```

```shell
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
-c A03599D1 \ # chip number
-i /media/Data/A03599D1_DAPI_fov_stitched.tif \  # ssDNA, DAPI, HE data path
-if /media/Data/A03599D1_IF_fov_stitched.tif \  # IF data path
-s DAPI \  # stain type，(ssDNA, DAPI, HE)
-m /media/Data/A03599D1.raw.gef \  # Transcriptomics gef path
-pr /media/Data/A03599D1.protein.raw.gef \  # protein gef path
-w /media/Data/weights \ # weights save dir
-o /media/Data/C04042E3_demo \ # output dir
-r  # if report 
```

### Customization
***You can customize parameters in this mode. You can use config/StereoCITE_param.json as an example***
```shell
CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
-c \
A03599D1 \
-s \
DAPI \
-w \
/media/Data/data/cellbin2/weights \
-o \
/media/Data/data/cellbin2/test/A03599D1_demo_3 \
-p \
/media/code/cellbin2/cellbin2/config/StereoCITE_param.json
```

## Outputs

| File Name | Description |
| ---- | ---- |
| A03599D1_cell_mask.tif | Final cell mask |
| A03599D1_mask.tif | Final nuclear mask |
| A03599D1_tissue_mask.tif | Final tissue mask |
| A03599D1_params.json | CellBin 2.0 input params |
| A03599D1.ipr | Image processing record |
| metrics.json | CellBin 2.0 Metrics |
| CellBin_0.0.1_report.html | CellBin 2.0 report |
| A03599D1.rpi | Recorded image processing (for visualization) |
| A03599D1_DAPI_mask.tif | Cell mask on registered image |
| A03599D1_DAPI_mask_merged.tif | Merged cell mask on registered image (nuclear + cell membrane) |
| A03599D1_DAPI_mask_raw.tif | Cell raw mask on registered image |
| A03599D1_DAPI_register.txt | Track template on registered image |
| A03599D1_DAPI_register_track.txt | Track detect result on registered image |
| A03599D1_DAPI_regist.tif | Registered image |
| A03599D1_DAPI_stitch.tif | Stitched image |
| A03599D1_DAPI_stitch_template.txt | Track template on stitched image |
| A03599D1_DAPI_tissue_cut.tif | Tissue mask on registered image |
| A03599D1_DAPI_tissue_cut_raw.tif | Tissue raw mask on registered image |
| A03599D1_DAPI_transform_mask.tif | Cell mask on transformed image |
| A03599D1_DAPI_transform_mask_raw.tif | Cell raw mask on transformed image |
| A03599D1_DAPI_transform_tissue_cut.tif | Tissue mask on transformed image |
| A03599D1_DAPI_transform_tissue_cut_raw.tif | Tissue raw mask on transformed image |
| A03599D1_DAPI_transformed.tif | Transformed image |
| A03599D1_DAPI_transformed.txt | Track template on transformed image |
| A03599D1_DAPI_transformed_track.txt | Track detect result on transformed image |
| A03599D1_IF_mask.tif | Cell mask on registered image |
| A03599D1_IF_mask_raw.tif | Cell raw mask on registered image |
| A03599D1_IF_stitch.tif | Stitched image |
| A03599D1_IF_tissue_cut.tif | Tissue mask on registered image |
| A03599D1_IF_tissue_cut_raw.tif | Tissue raw mask on registered image |
| A03599D1_IF_transform_mask.tif | Cell mask on transformed image |
| A03599D1_IF_transform_mask_raw.tif | Cell raw mask on transformed image |
| A03599D1_IF_transform_tissue_cut.tif | Tissue mask on transformed image |
| A03599D1_IF_transform_tissue_cut_raw.tif | Tissue raw mask on transformed image |
| A03599D1_IF_transformed.tif | Transformed image |
| A03599D1_Transcriptomics.cellbin.gef | CellBin gef |
| A03599D1_Transcriptomics.adjusted.cellbin.gef | CellBin correct gef |
| A03599D1_Transcriptomics.tif | Gene matrix in tif format |
| A03599D1_Transcriptomics_matrix_template.txt | Track template on gene matrix |
| A03599D1_Transcriptomics.tissue.gef | Tissuebin gef |
| A03599D1_Transcriptomics_tissue_cut.tif | Tissue mask on gene matrix |

## Other content


## Reference
https://github.com/MouseLand/cellpose <br>
https://github.com/matejak/imreg_dft <br>
https://github.com/rezazad68/BCDU-Net <br>
https://github.com/libvips/pyvips <br>
https://github.com/vanvalenlab/deepcell-tf <br>
https://github.com/ultralytics/ultralytics <br>
