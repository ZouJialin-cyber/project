import os
import cv2
import json

# 定义源文件夹路径和目标文件夹路径
json_folder = r"C:\Users\zoujialin\Desktop\jjjson"  # JSON文件所在文件夹路径
tif_folder = r"C:\Users\zoujialin\Desktop\gengxin"  # TIF图片所在文件夹路径
output_folder = r"C:\Users\zoujialin\Desktop\output"  # 输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有JSON文件
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

# 遍历所有JSON文件
for json_file in json_files:
    # 获取无后缀的文件名
    file_name = os.path.splitext(json_file)[0]

    # 检查对应的TIF文件是否存在
    tif_file = file_name + '.tif'
    if tif_file in os.listdir(tif_folder):
        # 读取JSON文件
        with open(os.path.join(json_folder, json_file), 'r') as f:
            data = json.load(f)

        # 读取TIF图片
        tif_path = os.path.join(tif_folder, tif_file)
        image = cv2.imread(tif_path)

        # 绘制边界框
        for shape in data['shapes']:
            points = shape['points']
            pt1 = (int(points[0][0]), int(points[0][1]))
            pt2 = (int(points[1][0]), int(points[1][1]))
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

        # 保存带有边界框的图片
        output_path = os.path.join(output_folder, tif_file)
        cv2.imwrite(output_path, image)
        print(f"Processed and saved: {tif_file}")
    else:
        print(f"No corresponding TIF file found for: {json_file}")

print("Finished processing all JSON files.")