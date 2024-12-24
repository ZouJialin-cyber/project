import numpy as np
import json


def read_txt_points(txt_file_path):
    points = []
    with open(txt_file_path, 'r') as file:
        for line in file.readlines():
            x, y = map(float, line.strip().split())
            points.append([x, y])
    return np.array(points)


def fit_line(points):
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1].reshape(-1, 1)
    # 使用最小二乘法拟合直线，拟合直线的方程为 y = mx + b
    m, b = np.polyfit(x.flatten(), y.flatten(), 1)
    # 根据拟合出的直线方程，取最小和最大的x值对应的坐标作为直线的两个端点
    x_min, x_max = np.min(x), np.max(x)
    start_point = np.array([x_min, m * x_min + b])
    end_point = np.array([x_max, m * x_max + b])
    return start_point, end_point


def create_geojson(start_point, end_point):
    feature = {
        "type": "Feature",
        "id": "e38cca03-4116-4b6e-9ab6-734d0dd7a522",
        "geometry": {
            "type": "LineString",
            "coordinates": [start_point.tolist(), end_point.tolist()]
        },
        "properties": {
            "objectType": "annotation",
            "classification": {
                "name": "chip",
                "color": [125, 221, 230]
            }
        }
    }
    feature_collection = {
        "type": "FeatureCollection",
        "features": [feature]
    }
    return feature_collection


def txt_to_geojson(txt_file_path):
    points = read_txt_points(txt_file_path)
    start_point, end_point = fit_line(points)
    geojson_data = create_geojson(start_point, end_point)
    geojson_file_path = txt_file_path.replace('.txt', '.geojson')
    with open(geojson_file_path, 'w') as file:
        json.dump(geojson_data, file, indent=4)


# 使用示例，替换为你实际的txt文件路径
txt_file_path = r"C:\Users\zoujialin\Desktop\gold\test\mini_json_label\C03632D2\txt\C03632D2_bottom_17175.txt"
txt_to_geojson(txt_file_path)