import cv2
import os
import tifffile as tif
from tqdm import tqdm

# 原始图像所在文件夹路径
source_folder = "/storeData/USER/data/01.CellBin/00.user/zoujialin/tseg_silver/HE/wjw/img"
# 目标图像保存的文件夹路径
target_folder = "/storeData/USER/data/01.CellBin/00.user/zoujialin/tseg_silver/HE/wjw/img_raw"

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 用于存放所有.tif图像文件的路径列表
tif_files = []
for file_name in os.listdir(source_folder):
    if file_name.endswith('.tif'):
        source_path = os.path.join(source_folder, file_name)
        tif_files.append(source_path)

# 使用tqdm为循环添加进度条
for source_path in tqdm(tif_files):
    # 读取图像
    img = tif.imread(source_path)
    if img is not None:
        # 获取原始图像的高度和宽度
        height, width = img.shape[:2]
        # 计算八倍上采样后的尺寸
        new_height = height * 8
        new_width = width * 8
        # 使用最近邻插值法进行八倍上采样
        upsampled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        # 提取文件名（包含扩展名）
        file_name = os.path.basename(source_path)
        # 构建目标图像的完整路径（文件名保持不变）
        target_path = os.path.join(target_folder, file_name)
        # 保存上采样后的图像到目标文件夹
        tif.imwrite(target_path, upsampled_img, compression='zlib')

print("图像上采样并保存完成！")