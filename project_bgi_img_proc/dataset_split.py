import os
import shutil
import random

# 原始数据集 tif 图像文件夹路径
tif_folder = r"C:\Users\zoujialin\Desktop\org_cut\img"
# 原始数据集 txt 标签文件夹路径
txt_folder = r"C:\Users\zoujialin\Desktop\org_cut\label"
# 新格式的数据集根目录
new_dataset_root = r"C:\Users\zoujialin\Desktop\org_cut"

# 创建新的 images 和 labels 文件夹，并在其中创建 train、val、test 子文件夹
os.makedirs(os.path.join(new_dataset_root, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_root, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_root, "images", "test"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_root, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_root, "labels", "val"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_root, "labels", "test"), exist_ok=True)

# 获取所有的 tif 文件和 txt 文件
tif_files = [f for f in os.listdir(tif_folder) if f.endswith(".tif")]
txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]

# 确保 tif 文件和 txt 文件数量一致
if len(tif_files)!= len(txt_files):
    print("tif 文件和 txt 文件数量不一致，请检查数据。")

# 随机打乱文件列表
random.shuffle(tif_files)

total_files = len(tif_files)
train_size = int(total_files * 0.8)
val_size = int(total_files * 0.1)

# 划分训练集
for file in tif_files[:train_size]:
    tif_source = os.path.join(tif_folder, file)
    txt_source = os.path.join(txt_folder, file[:-4] + ".txt")
    tif_destination = os.path.join(new_dataset_root, "images", "train", file)
    txt_destination = os.path.join(new_dataset_root, "labels", "train", file[:-4] + ".txt")
    shutil.copy(tif_source, tif_destination)
    shutil.copy(txt_source, txt_destination)

# 划分验证集
for file in tif_files[train_size:train_size + val_size]:
    tif_source = os.path.join(tif_folder, file)
    txt_source = os.path.join(txt_folder, file[:-4] + ".txt")
    tif_destination = os.path.join(new_dataset_root, "images", "val", file)
    txt_destination = os.path.join(new_dataset_root, "labels", "val", file[:-4] + ".txt")
    shutil.copy(tif_source, tif_destination)
    shutil.copy(txt_source, txt_destination)

# 划分测试集
for file in tif_files[train_size + val_size:]:
    tif_source = os.path.join(tif_folder, file)
    txt_source = os.path.join(txt_folder, file[:-4] + ".txt")
    tif_destination = os.path.join(new_dataset_root, "images", "test", file)
    txt_destination = os.path.join(new_dataset_root, "labels", "test", file[:-4] + ".txt")
    shutil.copy(tif_source, tif_destination)
    shutil.copy(txt_source, txt_destination)

print("数据集划分完成。")