# 根据所标数据，对数据预处理，将其生成yolo格式并绘图

import os
import json
import math
import numpy as np
import cv2
import tifffile as tiff
import shutil


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
    # # 交点坐标小于0抛出
    # if px < 0 or py < 0:
    #     return 0

    return int(px), int(py)


def calculate_angle(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    vec1 = [(x2 - x1), (y2 - y1)]
    vec2 = [(x4 - x3), (y4 - y3)]

    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

    cos_angle = dot_product / (magnitude1 * magnitude2)
    angle_radians = math.acos(cos_angle)

    angle_degrees = math.degrees(angle_radians)
    if angle_degrees > 90:
        angle_degrees = 180 - angle_degrees

    return angle_degrees


def classify_and_find_intersections(lines):
    line1 = lines[0]
    for i in [1, 2, 3]:
        line2 = lines[i]
        angle = calculate_angle(line1, line2)
        if angle < 30:
            num = 3
            for j in [1, 2, 3]:
                if j!= i:
                    if num == 4:
                        line4 = lines[j]
                        break
                    if num == 3:
                        line3 = lines[j]
                        num += 1
            # print(line1, line2, line3, line4)
            # print(calculate_angle(line1, line2))
            intersection_points = []
            intersection_points.append(line_intersection(line1, line3))
            intersection_points.append(line_intersection(line1, line4))
            intersection_points.append(line_intersection(line2, line4))
            intersection_points.append(line_intersection(line2, line3))
            return intersection_points


def find_min_area_rect(points):
    points = np.array(points, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    # box = np.int0(box)

    return box


def show_point(points):
    canvas = np.ones((2400, 2500, 3), dtype="uint8") * 255
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 蓝色，绿色，红色，黄色
    for i, point in enumerate(points):
        cv2.circle(canvas, tuple(point), 10, colors[i], -1)  # 使用不同颜色绘制每个点

    # 连接所有点形成一个多边形
    cv2.polylines(canvas, [points], isClosed=True, color=(0, 0, 0), thickness=2)

    # 显示画布
    cv2.imshow('Polygon', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def my_shord(points):
    # 找到 y 值最小的点
    min_y_points = points[points[:, 1] == np.min(points[:, 1])]

    # 如果存在唯一的 y 最小点
    sorted_points = []
    if len(min_y_points) == 1:
        a = min_y_points[0]  # y 最小点
        # print(a)
        # 找到 x 最小的点
        min_x_point = points[points[:, 0] == np.min(points[:, 0])][0]
        b = min_x_point  # x 最小点
        l1 = abs(a[0] - b[0])  # x 值差异
        l2 = abs(b[1] - a[1])  # y 值差异

        if l1 < l2:
            # 顺序：y 最小 -> x 最大 -> y 最大 -> x 最小
            sorted_points.append(a)
            sorted_points.append(points[points[:, 0] == np.max(points[:, 0])][0])
            sorted_points.append(points[points[:, 1] == np.max(points[:, 1])][0])
            sorted_points.append(b)
        else:
            # 顺序：x 最小 -> y 最小 -> x 最大 -> y 最大
            sorted_points.append(b)
            sorted_points.append(a)
            sorted_points.append(points[points[:, 0] == np.max(points[:, 0])][0])
            sorted_points.append(points[points[:, 1] == np.max(points[:, 1])][0])
        sorted_points_list = [point.tolist() for point in sorted_points]
    else:
        sorted_points_list = []
        # 如果存在多个 y 最小点，按左上、右上、右下、左下顺序排列
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])

        sorted_points_list.append([min_x, min_y])
        sorted_points_list.append([max_x, min_y])
        sorted_points_list.append([max_x, max_y])
        sorted_points_list.append([min_x, max_y])

    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    # 打印排序后的点
    return sorted_points_list, min_x, max_x, min_y, max_y


num = 0
dic_path = r"D:\HE_program\HE\HE_data\chip\he_data_stage_1\HE_chip_label"
img_dir = r"D:\HE_program\HE\HE_data\chip\he_data_stage_1\HE_chip_img"
out_label = r"D:\HE_program\HE\HE_data\chip\he_data_stage_1\test_gold\label"
out_img = r"D:\HE_program\HE\HE_data\chip\he_data_stage_1\test_gold\img"

# 新增：定义用于绘制矩形的颜色
draw_color = (0, 255, 0)

for dir_label in os.listdir(dic_path):
    # print(dir_label)
    path = os.path.join(dic_path, dir_label)
    # print(path)
    if not os.path.isdir(path):
        continue
    for sn in os.listdir(path):
        # print(sn)
        sn_name = sn.split('.')[0]
        json_path = os.path.join(path, sn)
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
            # print(json_data)
            # print(type(json_data))

        chip_line = []
        json_data = json_data["features"]
        if isinstance(json_data, list):
            for dir in json_data:
                print(sn_name)
                print(dir["properties"]["classification"]["name"])
                if dir["properties"]["classification"]["name"] == "chip":
                    chip_line.append(dir["geometry"]["coordinates"])
        print(chip_line)
        if len(chip_line) == 4:
            extened_lines = []
            for coords in chip_line:
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                extened_lines.append([x1, y1, x2, y2])

            # 找交交点
            intersection_points = classify_and_find_intersections(extened_lines)
            num += 1
            # 找近似的正方形
            closest_rectangle = find_min_area_rect(intersection_points)
            # 排序成需要的点顺序
            sorted_points, min_x, max_x, min_y, max_y = my_shord(closest_rectangle)
            # print(sorted_points)
            img_path = os.path.join(img_dir, f"{dir_label}\\{sn_name}.tif")
            img_raw = tiff.imread(img_path)
            h, w, c = img_raw.shape

            if max_y <= h and min_y >= 0 and max_x <= w and min_x >= 0:
                out_img_path = os.path.join(out_img, f'{sn_name}.tif')
                out_txt_path = os.path.join(out_label, f'{sn_name}.txt')

                shutil.copy(img_path, out_img_path)
                # sorted_points = sorted_points[:, 0]/w
                # sorted_points = sorted_points[:, 1]/h
                normalized_points_list = [[x / w, y / h] for x, y in sorted_points]
                txt_list = [0]
                for i in range(4):
                    for j in range(2):
                        txt_list.append(normalized_points_list[i][j])

                # print(normalized_points_list)
                print(txt_list)
                write_list = ' '.join(map(str, txt_list))
                with open(out_txt_path, 'w') as txt_file:
                    txt_file.write(write_list)

                # 新增：在图像上绘制矩形
                img_with_rect = img_raw.copy()
                pts = np.array(sorted_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_with_rect, [pts], True, draw_color, 2)

                # 保存带有矩形的图像到新文件夹
                annotated_img_path = os.path.join(out_img, f'{sn_name}_annotated.tif')
                tiff.imwrite(annotated_img_path, img_with_rect)

# break

print(num)