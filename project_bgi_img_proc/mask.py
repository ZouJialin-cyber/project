"""
    FF_HE：标注库数据的json文件补充"Image_path", "Mask_path"路径信息
    "image_properties", "Image_type"在down_sample.py中补充
"""

import os
import json
from tqdm import tqdm

# 假设这里已经读取了JSON文件内容到data变量中，例如使用以下方式读取（示例，你需要根据实际路径替换）
json_file_path = '/storeData/USER/data/01.CellBin/00.user/zoujialin/ssDNA_test/merge_FF_HE.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

folders = ['/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/FF_HE/gold',
           '/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/FF_HE/silver',
           '/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/FF_HE/back_up/inaccurate']

total_lenth = len(data)

for key, attributes in tqdm(data.items(), total=total_lenth):
    image_filename = key + '_img.tif'
    mask_filename = key + '_mask.tif'
    image_path = None
    mask_path = None
    for folder in folders:
        image_full_path = os.path.join(folder, image_filename)
        mask_full_path = os.path.join(folder, mask_filename)
        if os.path.exists(image_full_path):
            image_path = image_full_path
        if os.path.exists(mask_full_path):
            mask_path = mask_full_path
    if image_path:
        attributes['Image_path'] = image_path
    if mask_path:
        attributes['Mask_path'] = mask_path

# 假设保存更新后的JSON文件到新文件（示例，你需要根据实际路径替换）
with open('/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut/FF_HE/FF_HE_updated_20241218.json', 'w') as file:
    json.dump(data, file, indent=4)

print("JSON文件已更新，添加了Image_path和Mask_path键值对。")