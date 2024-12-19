import os
import tifffile
import numpy as np
import cv2

tif_folder = r"C:\Users\zoujialin\Desktop\org_cut\images\val"
yolo_label_folder = r"C:\Users\zoujialin\Desktop\org_cut\labels\val"
output_folder = r"C:\Users\zoujialin\Desktop\org_cut\output_with_boxes"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def convert_yolo_to_xywh(yolo_coords, image_shape):
    img_height, img_width = image_shape[:2]
    label_class, center_x, center_y, width, height = map(float, yolo_coords)
    x = center_x * img_width - width * img_width / 2
    y = center_y * img_height - height * img_height / 2
    w = width * img_width
    h = height * img_height
    return int(label_class), int(x), int(y), int(w), int(h)

def draw_bounding_boxes(image_path, label_path):
    tif_image = tifffile.imread(image_path)
    img_height, img_width = tif_image.shape[:2]
    with open(label_path, 'r') as f:
        lines = f.readlines()
    aspect_ratios = []
    for line in lines:
        label_class, x, y, w, h = convert_yolo_to_xywh(line.split(), (img_height, img_width))
        cv2.rectangle(tif_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if w > 0 and h > 0:
            aspect_ratio = h / w
            aspect_ratios.append(aspect_ratio)
    return tif_image, aspect_ratios, len(lines)

max_aspect_ratio = 0
most_labeled_file = None
max_label_count = 0

for tif_filename in os.listdir(tif_folder):
    base_name, _ = os.path.splitext(tif_filename)
    label_filename = f'{base_name}.txt'
    if label_filename in os.listdir(yolo_label_folder):
        tif_path = os.path.join(tif_folder, tif_filename)
        label_path = os.path.join(yolo_label_folder, label_filename)
        image_with_boxes, aspect_ratios_for_file, label_count = draw_bounding_boxes(tif_path, label_path)
        output_path = os.path.join(output_folder, f'{base_name}_with_boxes.png')
        cv2.imwrite(output_path, image_with_boxes)
        for ratio in aspect_ratios_for_file:
            if ratio > max_aspect_ratio:
                max_aspect_ratio = ratio
        if label_count > max_label_count:
            max_label_count = label_count
            most_labeled_file = base_name

print(f"最大高宽比为：{max_aspect_ratio}")
print(f"标签行最多的文件是：{most_labeled_file}，有{max_label_count}行标签。")