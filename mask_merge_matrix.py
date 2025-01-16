import cv2
import os
import tifffile as tif
import numpy as np


def merge_images(folder1, folder2, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历第一个文件夹中的文件
    for filename in os.listdir(folder1):
        if filename.endswith('.tif'):
            file1_path = os.path.join(folder1, filename)
            file2_path = os.path.join(folder2, filename)

            # 检查第二个文件夹中是否有对应的文件
            if os.path.exists(file2_path):
                # 读取图像
                matrix_img = tif.imread(file1_path)
                mask_img = tif.imread(file2_path)

                if matrix_img is not None and mask_img is not None:
                    # 将灰度图转换为三通道图像
                    matrix_img = cv2.cvtColor(matrix_img, cv2.COLOR_GRAY2BGR)
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

                    # 将 mask 图中的 255 部分设为红色，0 部分设为透明
                    mask_img[(np.all(mask_img == [255, 255, 255], axis=2))] = [0, 0, 255]
                    mask_img[(np.all(mask_img == [0, 0, 0], axis=2))] = [0, 0, 0]

                    # 加权合并图像
                    merged_img = cv2.addWeighted(matrix_img, 0.6, mask_img, 0.4, 0)

                    # 保存合并后的图像
                    output_path = os.path.join(output_folder, filename)
                    tif.imwrite(output_path, merged_img, compression='zlib')
                    print(f"Merged and saved {filename}")


# matrix
folder1 = r"/storeData/USER/data/01.CellBin/00.user/zoujialin/矩阵组织分割算法优化/dataset/enhancement_temp"
# mask
folder2 = r"/storeData/USER/data/01.CellBin/00.user/zoujialin/矩阵组织分割算法优化/dataset/mask_512"

output_folder = r"/storeData/USER/data/01.CellBin/00.user/zoujialin/矩阵组织分割算法优化/dataset/merge"
merge_images(folder1, folder2, output_folder)