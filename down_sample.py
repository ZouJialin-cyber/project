"""
    补充好json的路径信息之后，更改其他信息
    更新标注库json，"image_properties"的值部分加上tissuecut
    for example: "silver" ——————> "tissuecut_silver"
    每一条数据增加新的内容："Image_type"拼接图or配准图
    标注库变更文档放入路径"/storeData/USER/data/01.CellBin/01.data/04.dataset/04.train_test_data/tissuecut"
"""

import json

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 写入JSON文件
def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# 修改JSON数据
def modify_json_data(data):
    for key, value in data.items():
        if 'image_properties' in value:
            value['image_properties'] = 'tissuecut_' + value['image_properties']
        value['Image_type'] = ""

# 主程序
def main():
    input_file_path = r"C:\Users\zoujialin\Desktop\111.json"
    output_file_path = r"C:\Users\zoujialin\Desktop\FFPE_HE_updated_20241218.json"

    # 读取JSON数据
    json_data = read_json_file(input_file_path)

    # 修改JSON数据
    modify_json_data(json_data)

    # 写入修改后的JSON数据
    write_json_file(output_file_path, json_data)

    print("JSON数据已成功修改并保存到文件。")

if __name__ == "__main__":
    main()