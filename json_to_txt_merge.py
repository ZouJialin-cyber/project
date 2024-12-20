"""
    金标数据映射及生成txt坐标
    处理json坐标，延长找交点，在图像上绘制，生成txt文件
"""

import json
import os
import math
import cv2
import tifffile
from scipy.spatial import ConvexHull
import numpy as np


def is_rectangle(lines, image_width, image_height):
    """
    判断由四条直线组成的图形是否为矩形，并根据图像尺寸舍弃不符合要求的交点
    :param lines: 四条直线的坐标列表，每条直线由两个端点坐标表示
    :param image_width: 对应的tif图像的宽度
    :param image_height: 对应的tif图像的_height
    :return: 如果是矩形返回True，否则返回False
    """
    intersections = []

    # 计算四条直线两两相交的交点
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            intersection = find_intersection(line1, line2)
            if intersection:
                intersections.append(intersection)

    # 舍弃坐标带有负数或超出图像大小的交点
    valid_intersections = []
    for intersection in intersections:
        x, y = intersection
        if x >= 0 and y >= 0 and x <= image_width and y <= image_height:
            valid_intersections.append(intersection)

    if len(valid_intersections)!= 4:
        return False

    # 检查四个交点组成的四个角是否都是直角
    if not check_angles(valid_intersections):
        return False

    return True


def find_intersection(line1, line2):
    """
    计算两条直线的交点坐标
    :param line1: 第一条直线的坐标，由两个端点坐标表示
    :param line2: 第二条直线的坐标，由两个端点坐标表示
    :return: 交点坐标，如果不存在交点则返回None
    """
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        return None

    numerator_x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    numerator_y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

    x = numerator_x / denominator
    y = numerator_y / denominator

    return [x, y]

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

def process_geojson_files(folder_path, tif_folder_path, output_folder_path):
    """
    遍历文件夹下的所有.geojson文件，判断其中四条直线是否构成矩形并统计数量，同时读取对应的tif图像进行验证
    并为每个图像计算夹角的四条线的四个交点映射到图片上并连线，将新图保存到指定目的文件夹
    :param folder_path: 包含.geojson文件的文件夹路径
    :param tif_folder_path: 包含对应的tif图像的文件夹路径
    :param output_folder_path: 保存新图的目的文件夹路径
    :return: 矩形的数量和非矩形的数量，以元组形式返回 (矩形数量, 非矩形数量)
    """
    rectangle_count = 0
    non_rectangle_count = 0
    non_rectangle_angle_info = []  # 用于存储非矩形的角度信息
    non_aligned_rectangle_info = []  # 新增：用于存储非正放置矩形的信息
    aligned_rectangle_num = 0

    # 新增：用于统计不同偏离角度范围的矩形数量及对应文件名
    aligned_count = 0
    within_5_degrees_count = 0
    within_5_to_10_degrees_count = 0
    within_10_to_20_degrees_count = 0
    within_20_to_30_degrees_count = 0
    over_30_degrees_count = 0
    within_5_degrees_files = []
    within_5_to_10_degrees_files = []
    within_10_to_20_degrees_files = []
    within_20_to_30_degrees_files = []
    over_30_degrees_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.geojson'):
                file_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                tif_file_path = os.path.join(tif_folder_path, base_name + '.tif')

                with open(file_path, 'r') as f:
                    data = json.load(f)
                    lines = []
                    for feature in data['features']:
                        line_string = feature['geometry']['coordinates']
                        lines.append(line_string)

                try:
                    image = tifffile.imread(tif_file_path)
                    image = f_ij_16_to_8(image)
                    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    image_width, image_height = image.shape[1], image.shape[0]
                except FileNotFoundError:
                    print(f"未找到对应的tif图像: {tif_file_path}，跳过该.geojson文件的处理。")
                    continue

                intersections = []
                for i in range(len(lines)):
                    for j in range(i + 1, len(lines)):
                        line1 = lines[i]
                        line2 = lines[j]
                        intersection = find_intersection(line1, line2)
                        if intersection:
                            intersections.append(intersection)

                valid_intersections = []
                for intersection in intersections:
                    x, y = intersection
                    if x >= 0 and y >= 0 and x <= image_width and y <= image_height:
                        valid_intersections.append(intersection)

                # 对坐标值四舍五入处理
                for i in range(len(valid_intersections)):
                    for j in range(len(valid_intersections[i])):
                        valid_intersections[i][j] = round(valid_intersections[i][j])

                if len(valid_intersections) == 4:
                    angles, angle_diffs, max_diff = check_angles_extended(valid_intersections, base_name)
                    if check_angles(valid_intersections):
                        rectangle_count += 1
                    else:
                        non_rectangle_count += 1
                        non_rectangle_angle_info.append((angles, angle_diffs, max_diff))

                    if not is_aligned(valid_intersections, image_height):  # 新增：判断是否为正放置矩形
                        deviation_angle = calculate_deviation_angle(valid_intersections)
                        abs_deviation_angle = abs(deviation_angle)  # 将偏离角度转换为绝对值

                        non_aligned_rectangle_info.append((base_name, abs_deviation_angle))

                        if abs_deviation_angle == 0:
                            aligned_count += 1
                        elif abs_deviation_angle < 5:
                            within_5_degrees_count += 1
                            within_5_degrees_files.append(base_name)
                        elif 5 <= abs_deviation_angle < 10:
                            within_5_to_10_degrees_count += 1
                            within_5_to_10_degrees_files.append(base_name)
                        elif 10 <= abs_deviation_angle < 20:
                            within_10_to_20_degrees_count += 1
                            within_10_to_20_degrees_files.append(base_name)
                        elif 20 <= abs_deviation_angle < 30:
                            within_20_to_30_degrees_count += 1
                            within_20_to_30_degrees_files.append(base_name)
                        else:
                            over_30_degrees_count += 1
                            over_30_degrees_files.append(base_name)
                    else:
                        aligned_rectangle_num += 1

                # 无论是否为矩形，都绘制并保存图像
                plot_lines_and_intersections(image, valid_intersections, output_folder_path, base_name)
                save_intersection_coordinates(valid_intersections, txt_folder_path, base_name)


    # 输出所有角与90度的最大差值
    max_diff_all = 0
    for angles, angle_diffs, max_diff in non_rectangle_angle_info:
        max_diff_all = max(max_diff_all, max_diff)
    print(f"Max difference from 90 degrees for non-rectangles: {max_diff_all} degrees")

    return rectangle_count, non_rectangle_count, non_rectangle_angle_info, non_aligned_rectangle_info, aligned_count, within_5_degrees_count, within_5_to_10_degrees_count, within_10_to_20_degrees_count, within_20_to_30_degrees_count, over_30_degrees_count, within_5_degrees_files, within_5_to_10_degrees_files, within_10_to_20_degrees_files, within_20_to_30_degrees_files, over_30_degrees_files, aligned_rectangle_num


def check_angles(valid_intersections):
    """
    检查由四个交点构成的四边形的每个角是否为直角
    :param valid_intersections: 四个有效交点的坐标列表
    :return: 如果四个角都是直角返回True，否则返回False
    """
    if len(valid_intersections)!= 4:
        return False

    angles = []
    for i in range(4):
        angle = calculate_angle(valid_intersections[i], valid_intersections[(i + 1) % 4], valid_intersections[(i + 2) % 4])
        angles.append(angle)

    # 检查所有角是否恰好是90度
    for angle in angles:
        if not math.isclose(angle, math.pi / 2, abs_tol=0.0):  # 不允许任何误差
            return False

    return True


def check_angles_extended(valid_intersections, base_name):
    """
    扩展的检查角度函数，计算每个角的度数和与90度的差值，以及最大差值
    :param valid_intersections: 四个有效交点的坐标列表
    :param base_name: 文件名（不含扩展名），用于打印信息
    :return: 每个角的度数列表，每个角与90度的差值列表和最大差值
    """
    if len(valid_intersections)!= 4:
        return [], [], 0

    # 使用ConvexHull来确定点的顺序
    hull = ConvexHull(valid_intersections)
    sorted_indices = hull.vertices
    sorted_intersections = [valid_intersections[i] for i in sorted_indices]

    angles = []
    angle_diffs = []
    for i in range(4):
        angle = calculate_angle(sorted_intersections[i], sorted_intersections[(i + 1) % 4], sorted_intersections[(i + 2) % 4])
        angles.append(math.degrees(angle))  # 转换为度数
        diff = abs(angle - math.pi / 2)  # 90 degrees in radians
        angle_diffs.append(diff)

    max_diff = max(angle_diffs) * 180 / math.pi  # 转换为度数
    print(f"Angles for {base_name}: {angles}, Max difference from 90 degrees: {max_diff}")  # 打印每个角的度数和最大差值

    return angles, angle_diffs, max_diff


def calculate_angle(point1, point2, point3):
    """
    按照正确的向量选取方式（点2到点1的向量和点2到点3的向量）计算由三个点构成的角的度数
    :param point1: 第一个点的坐标
    :param point2: 第二个点的坐标，角的顶点
    :param point3: 一个点的坐标
    :return: 角的度数（以弧度为单位）
    """
    v1 = [point1[0] - point2[0], point1[1] - point2[1]]
    v2 = [point3[0] - point2[0], point3[1] - point2[1]]

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    cos_angle = dot_product / (mag1 * mag2)
    angle = math.acos(cos_angle)

    return angle


def plot_lines_and_intersections(image, valid_interections, output_folder_path, base_name):
    """
    在图片上绘制用于计算夹角的四条线以及它们的  四个交点，并将连线后的图片保存到指定文件夹
    :param image: 读取的tif图像数据
    :param valid_interections: 四个有效交点的坐标列表
    :param output_folder_path: 保存新图的目的文件夹路径
    :param base_name: 文件名（不含扩展名）
    """
    # 将图像转换为OpenCV可处理的格式（如果需要，这里假设tif图像读取后格式符合要求，可直接使用）
    image_cv = image.copy()

    # 确保交点是有序的，使用凸包来确定顺序
    hull = ConvexHull(valid_interections)
    sorted_indices = hull.vertices
    sorted_interections = [valid_interections[i] for i in sorted_indices]

    # 绘制四个交点
    # for point in sorted_interections:
    #     x, y = point
    #     cv2.circle(image_cv, (int(x), int(y)), 1, (255, 0, 0), -1)

    # 绘制连接四个交点的线段
    for i in range(len(sorted_interections)):
        start_point = sorted_interections[i]
        end_point = sorted_interections[(i + 1) % len(sorted_interections)]
        cv2.line(image_cv, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])),
                 (0, 255, 0), 1, lineType=cv2.LINE_AA)

    output_file_path = os.path.join(output_folder_path, base_name + '.png')
    tifffile.imwrite(output_file_path, image_cv)


def is_aligned(valid_intersections, image_height):
    """
    判断矩形是否为正放置（最上面的边和图片水平线平行）
    :param valid_intersections: 四个有效交点的坐标列表
    :param image_height: 图像的高度
    :return: 如果是正放置返回True，否则返回False
    """
    sorted_intersections = sorted(valid_intersections, key=lambda x: x[1])
    top_edge = (valid_intersections[0], valid_intersections[1])
    slope = calculate_slope(top_edge)

    if slope is None:
        return False

    return math.isclose(slope, 0, abs_tol=0.001)


def calculate_slope(line):
    """
    计算直线的斜率
    :param line: 由两个点组成的直线，以元组形式表示，每个点又是一个元组(x, y)
    :return: 直线的斜率
    """
    (x1, y1), (x2, y2) = line
    if x2 - x1 == 0:
        return None
    return (y2 - y1) / (x2 - x1)

def save_intersection_coordinates(valid_intersections, output_folder_path, base_name):
    """
    将四个交点坐标以左上 右上 右下 左下的顺序保存在一个txt文件中
    :param valid_intersections: 四个有效交点的坐标列表
    :param output_folder_path: 保存文件的目的文件夹路径
    :param base_name: 文件名（不含扩展名）
    """
    # 使用ConvexHull确定点的顺序
    hull = ConvexHull(valid_intersections)
    sorted_indices = hull.vertices
    sorted_intersections = [valid_intersections[i] for i in sorted_indices]

    txt_file_path = os.path.join(output_folder_path, base_name + '.txt')
    with open(txt_file_path, 'w') as f:
        for point in sorted_intersections:
            x, y = point
            f.write(f"{x} {y}\n")

def calculate_deviation_angle(valid_intersections):
    """
    计算矩形与图片水平线的偏离角度
    :param valid_intersections: 四个有效交点的坐标列表
    :return: 矩形与水平线的偏离角度（以度为单位）
    """
    sorted_intersections = sorted(valid_intersections, key=lambda x: x[1])
    top_edge = (sorted_intersections[0], sorted_intersections[1])
    slope = calculate_slope(top_edge)
    if slope is None:
        return 90
    return math.degrees(math.atan(slope))

if __name__ == '__main__':
    # 指定包含.geojson文件的文件夹路径
    folder_path = r"C:\Users\zoujialin\Desktop\gold\test\raw_json_label"
    # 指定包含对应的tif图像的文件夹路径
    tif_folder_path = r"C:\Users\zoujialin\Desktop\gold\test\raw_img"
    # 指定保存新图的目的文件夹路径
    output_folder_path = r"C:\Users\zoujialin\Desktop\gold\test\merge"
    txt_folder_path = r"C:\Users\zoujialin\Desktop\gold\test\txt"
    rectangle_count, non_rectangle_count, non_rectangle_angle_info, non_aligned_rectangle_info, aligned_count, within_5_degrees_count, within_5_to_10_degrees_count, within_10_to_20_degrees_count, within_20_to_30_degrees_count, over_30_degrees_count, within_5_degrees_files, within_5_to_10_degrees_files, within_10_to_20_degrees_files, within_20_to_30_degrees_files, over_30_degrees_files, aligned_rectangle_count = process_geojson_files(folder_path, tif_folder_path, output_folder_path)
    print(f"矩形的数量为: {rectangle_count}")
    print(f"非矩形的数量为: {non_rectangle_count}")
    print(f"=================非正放置矩形的信息=====================")
    for file_name, deviation_angle in non_aligned_rectangle_info:
        print(f"文件名: {file_name}, 偏离角度: {deviation_angle}度")

    print(f"摆正的（未偏离）矩形数量: {aligned_rectangle_count}")
    print(f"偏离5度以内的矩形数量: {within_5_degrees_count}，文件名: {within_5_degrees_files}")
    print(f"偏离5 - 10度的矩形数量: {within_5_to_10_degrees_count}，文件名: {within_5_to_10_degrees_files}")
    print(f"偏离10 - 20度的矩形数量: {within_10_to_20_degrees_count}，文件名: {within_10_to_20_degrees_files}")
    print(f"偏离20 - 30度的矩形数量: {within_20_to_30_degrees_count}，文件名: {within_20_to_30_degrees_files}")
    print(f"偏离30度以上的矩形数量: {over_30_degrees_count}，文件名: {over_30_degrees_files}")