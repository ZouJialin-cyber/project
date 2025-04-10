import json
import os

def read_and_transform_geojson(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.geojson'):
            # 构建完整的文件路径
            file_path = os.path.join(source_folder, filename)
            # 读取GeoJSON文件
            with open(file_path, 'r') as file:
                data = json.load(file)

            # 检查文件内容是否已经是FeatureCollection格式
            if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
                # 已经是FeatureCollection格式，直接使用
                feature_collection = data
            elif isinstance(data, list):
                # 如果文件内容是一个列表，假定它包含Feature对象，并创建一个新的FeatureCollection
                feature_collection = {"type": "FeatureCollection", "features": data}
            else:
                # 如果文件内容既不是字典也不是列表，假定它是一个单独的Feature对象，并创建一个新的FeatureCollection
                feature_collection = {"type": "FeatureCollection", "features": [data]}

            # 构建目标文件路径
            target_file_path = os.path.join(target_folder, filename)
            # 将FeatureCollection写入目标文件
            with open(target_file_path, 'w') as target_file:
                json.dump(feature_collection, target_file, indent=2)

# 设置源文件夹和目标文件夹路径
source_folder = r"D:\HE_program\HE\HE_data\chip\he_data_stage_1\out\out_gene_label"
target_folder = r"D:\HE_program\HE\HE_data\chip\he_data_stage_1\out\out_gene_json"

# 读取GeoJSON文件，转换格式，并保存到目标文件夹
read_and_transform_geojson(source_folder, target_folder)