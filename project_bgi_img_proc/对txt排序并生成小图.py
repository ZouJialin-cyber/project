"""
    金标代码：
    txt标签排序，裁剪小图
"""

import os
import cv2
import numpy as np

def sort_points(coords):
    # 计算中心点
    center = np.mean(coords, axis=0)
    sorted_coords = sorted(coords, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    return np.array(sorted_coords)


def crop_and_save(image, coords, out_dir, base_name):
    # 定义压缩比例
    compression_ratio = 1

    # 定义裁剪函数：裁剪300x300的小图
    def crop_around_point(image, center, size=1000):
        x, y = center
        h, w = image.shape[:2]
        # 将坐标索引强制转换为整数
        x1, y1 = max(0, int(x - size // 2)), max(0, int(y - size // 2))
        x2, y2 = min(w, int(x + size // 2)), min(h, int(y + size // 2))
        cropped = image[y1:y2, x1:x2]
        if cropped.shape[0] < size or cropped.shape[1] < size:
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            padded[:cropped.shape[0], :cropped.shape[1]] = cropped
            cropped = padded
        return cropped

    # 裁剪四个角
    for i, point in enumerate(coords):
        corner_name = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left'][i]
        cropped_image = crop_around_point(image, point)

        # 压缩图像大小
        compressed_image = cv2.resize(cropped_image, (0, 0), fx=compression_ratio, fy=compression_ratio)

        cv2.imwrite(os.path.join(out_dir, corner_name, f"{base_name}.png"), compressed_image)

    # 裁剪四条边（每条边以边的中心为 中心裁剪）
    edges = [
        (coords[0], coords[1]),  # Top-Edge
        (coords[1], coords[2]),  # Right-Edge
        (coords[2], coords[3]),  # Bottom-Edge
        (coords[3], coords[0])  # Left-Edge
    ]
    for i, (start, end) in enumerate(edges):
        edge_name = ['Top-Edge', 'Right-Edge', 'Bottom-Edge', 'Left-Edge'][i]
        # 边的中心点
        center = np.mean([start, end], axis=0)
        cropped_image = crop_around_point(image, center)

        # 压缩图像大小
        compressed_image = cv2.resize(cropped_image, (0, 0), fx=compression_ratio, fy=compression_ratio)

        cv2.imwrite(os.path.join(out_dir, edge_name, f"{base_name}.png"), compressed_image)


def process_labels(label_path):
    # 读取并排序标签
    coords = np.loadtxt(label_path)
    sorted_coords = sort_points(coords)
    return sorted_coords


def create_folders(out_dir):
    # 创建文件夹结构
    os.makedirs(out_dir, exist_ok=True)
    subfolders = ['Sorted Labels', 'Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left', 'Top-Edge', 'Right-Edge',
                  'Bottom-Edge', 'Left-Edge']
    for subfolder in subfolders:
        os.makedirs(os.path.join(out_dir, subfolder), exist_ok=True)


def process_images_and_labels(image_dir, label_dir, out_dir):
    # 批量处理图像和标签
    create_folders(out_dir)

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".png", ".txt"))

            if os.path.exists(label_path):
                # 读取图像
                image = cv2.imread(image_path)

                # 处理标签
                sorted_coords = process_labels(label_path)

                # 保存排序后的标签
                sorted_label_path = os.path.join(out_dir, 'Sorted Labels', filename.replace(".png", ".txt"))
                np.savetxt(sorted_label_path, sorted_coords, fmt="%d")

                # 裁剪图像并保存
                crop_and_save(image, sorted_coords, out_dir, filename.replace(".png", ""))


if __name__ == "__main__":
    image_dir = r"C:\Users\zoujialin\Desktop\gold\demo\B04372C211\2\image"
    label_dir = r"C:\Users\zoujialin\Desktop\gold\demo\B04372C211\2\label"
    out_dir = r"C:\Users\zoujialin\Desktop\gold\demo\B04372C211\s2"

    process_images_and_labels(image_dir, label_dir, out_dir)

    print("处理完成！")