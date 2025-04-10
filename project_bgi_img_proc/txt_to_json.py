"""
    点拟合直线
    输入：txt
    输出：json
"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression
import json


def fit_points_to_line(txt_path):
    """
    从给定的txt文件中读取坐标点数据，拟合直线并返回直线两端点坐标，
    根据直线是横向还是纵向来确定取最值的维度
    """
    points = np.loadtxt(txt_path)
    x_list = points[:, 0].reshape((-1, 1))
    y_list = points[:, 1].reshape((-1, 1))

    # 计算x坐标最大值与最小值之差以及y坐标最大值与最小值之差
    x_diff = np.max(x_list) - np.min(x_list)
    y_diff = np.max(y_list) - np.min(y_list)

    # 判断直线是横向还是纵向
    if x_diff > y_diff:
        # 横向情况，照常取x的最小最大
        x_start = np.min(x_list)
        x_end = np.max(x_list)
        model = LinearRegression()
        model.fit(x_list, y_list)
        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        y_start = slope * x_start + intercept
        y_end = slope * x_end + intercept
    else:
        # 纵向情况，取y的最小最大
        y_start = np.min(y_list)
        y_end = np.max(y_list)
        model = LinearRegression()
        # 交换x和y坐标，将y作为自变量，x作为因变量进行拟合
        model.fit(y_list, x_list)
        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        x_start = slope * y_start + intercept
        x_end = slope * y_end + intercept

    return [[x_start, y_start], [x_end, y_end]]


def create_geojson_feature(line_coordinates):
    """
    根据直线的坐标点创建GeoJSON格式的Feature对象
    """
    feature = {
        "type": "Feature",
        "id": "00135bf5-9823-452a-9921-e4974786d38c",
        "geometry": {
            "type": "LineString",
            "coordinates": line_coordinates
        },
        "properties": {
            "objectType": "annotation",
            "classification": {
                "name": "chip",
                "color": [125, 221, 230]
            }
        }
    }
    return feature


def process_folder(input_folder, output_folder):
    """
    遍历输入文件夹及其子文件夹，对每个子文件夹中的txt文件进行处理，
    将拟合直线结果转换为GeoJSON格式并保存到输出文件夹相应位置
    """
    for root, dirs, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        output_root = os.path.join(output_folder, relative_path)
        os.makedirs(output_root, exist_ok=True)
        for file in files:
            if file.endswith('.txt'):
                txt_file_path = os.path.join(root, file)
                line_coordinates = fit_points_to_line(txt_file_path)
                feature = create_geojson_feature(line_coordinates)
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": [feature]
                }
                output_file_path = os.path.join(output_root, file.replace('.txt', '.geojson'))
                with open(output_file_path, 'w') as f:
                    json.dump(geojson_data, f, indent=4)


if __name__ == "__main__":
    input_folder = r"C:\Users\zoujialin\Desktop\gjj_fangong"
    output_folder = r"C:\Users\zoujialin\Desktop\gjj_fangong"
    process_folder(input_folder, output_folder)