import os
import shutil

# 定义文件夹路径
folder_a = '/storeData/USER/data/01.CellBin/00.user/zoujialin/HE_chip_program/chip_gold/data/53'
folder_b = '/storeData/USER/data/01.CellBin/00.user/zoujialin/HE_chip_program/chip_gold/data/gold_matrix_255'
destination_folder = '/storeData/USER/data/01.CellBin/00.user/zoujialin/HE_chip_program/chip_gold/data/53_matrix'

# 确保目的文件夹存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 获取文件夹A中的所有文件名
files_in_a = set(os.listdir(folder_a))

# 遍历文件夹B中的文件
for file_name in os.listdir(folder_b):
    # 检查文件名是否在文件夹A中
    if file_name in files_in_a:
        # 构建完整的文件路径
        file_path_b = os.path.join(folder_b, file_name)
        file_path_destination = os.path.join(destination_folder, file_name)

        # 复制文件
        shutil.copy(file_path_b, file_path_destination)
        print(f"文件 {file_name} 已复制到目的文件夹。")

print("所有匹配的文件已复制完成。")