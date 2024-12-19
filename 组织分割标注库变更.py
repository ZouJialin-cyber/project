"""
    入库代码：
    ssDNA数据入库整理，原始结构目录的图复制到新结构目录下，更新json文件

    ss直接用，he需要提前处理格式，部分图像名需要加上_img后缀，部分mask需要resize到图像大小

    ss要手动剔除一例数据，he要手动处理部分新数据文件名以及mask大小

    所有数据均16转8，zlib压缩参数，he数据会额外比对三个shape参数的位置并进行错误修正
"""

import struct
import json
import os
from tqdm import tqdm
import numpy as np
import tifffile as tif


# 定义转换函数 f_ij_16_to_8
def f_ij_16_to_8(img, chunk_size=1000):
    if img.dtype == 'uint8':
        return img
    dst = np.zeros(img.shape, np.uint8)
    p_max = np.max(img)
    p_min = np.min(img)
    scale = 256.0 / (p_max - p_min + 1)
    for idx in range(img.shape[0] // chunk_size + 1):
        sl = slice(idx * chunk_size, (idx + 1) * chunk_size)
        win_img = np.copy(img[sl])
        win_img = np.int16(win_img)
        win_img = (win_img & 0xffff)
        win_img = win_img - p_min
        win_img[win_img < 0] = 0
        win_img = win_img * scale + 0.5
        win_img[win_img > 255] = 255
        dst[sl] = np.array(win_img).astype(np.uint8)
    return dst

def save_image_with_compression(output_path, img):
    try:
        tif.imwrite(output_path, img, compression='zlib')
    except struct.error as e:
        print(f"压缩保存失败，尝试不使用压缩参数保存: {e}")
        tif.imwrite(output_path, img)

# 读取 JSON 文件
json_file_path = '/storeData/USER/data/01.CellBin/00.user/zoujialin/ssDNA_test/merge_FF_HE.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 定义文件夹路径
folders = ['/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/backup01/FF_HE/FF/img',
           '/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/backup01/FF_HE/FF/mask',
           '/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/backup/FF_HE/test/old/img',
           '/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/backup/FF_HE/test/old/mask',
           '/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/backup/FF_HE/train/old/img',
           '/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/backup/FF_HE/train/old/mask']
output_folders = {'gold': '/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/FF_HE/gold',
                  'silver': '/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/FF_HE/silver',
                  'inaccurate': '/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/FF_HE/back_up/inaccurate'}

total_tasks = len(data)

# 遍历 JSON 数据
for key, attributes in tqdm(data.items(), total=total_tasks):
    # 构建可能的图像文件名
    image_filename = key + '_img.tif'
    mask_filename = key + '_mask.tif'

    output_folder = output_folders[attributes['image_properties']]
    if os.path.exists(os.path.join(output_folder, image_filename)):
        print(f"图像 {image_filename} 已存在，跳过处理。")
        continue

    # 遍历文件夹查找匹配的图像文件
    for folder in folders:
        image_path = os.path.join(folder, image_filename)
        mask_path = os.path.join(folder, mask_filename)

        # 检查图像文件是否存在
        if os.path.exists(image_path):
            # 读取和处理图像
            img = tif.imread(image_path)
            img = f_ij_16_to_8(img)

            # 检查图像通道顺序并调整
            if img.ndim == 3 and img.shape[0] in [3, 4]:  # 如果是 c h w 格式
                img = np.transpose(img, (1, 2, 0))  # 转换为 h w c 格式

            # 确定输出文件夹和路径
            output_folder = output_folders[attributes['image_properties']]
            output_path = os.path.join(output_folder, image_filename)

            # 保存处理后的图像
            # 保存处理后的图像
            save_image_with_compression(output_path, img)

            # 更新 JSON 数据
            attributes['Image_path'] = output_path

        if os.path.exists(mask_path):
            # 读取和处理图像
            img = tif.imread(mask_path)
            img = f_ij_16_to_8(img)

            # 检查图像通道顺序并调整
            if img.ndim == 3 and img.shape[0] in [3, 4]:  # 如果是 c h w 格式
                img = np.transpose(img, (1, 2, 0))  # 转换为 h w c 格式

            # 确定输出文件夹和路径
            output_folder = output_folders[attributes['image_properties']]
            output_path = os.path.join(output_folder, mask_filename)

            # 保存处理后的图像
            # 保存处理后的图像
            save_image_with_compression(output_path, img)

            # 更新 JSON 数据
            attributes['Mask_path'] = output_path

# 保存更新后的 JSON 文件
with open('/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/FF_HE/FF_HE_updated_20241218.json', 'w') as file:
    json.dump(data, file, indent=4)

print("处理完成，图像已根据 JSON 指令复制和保存。")