import cv2
import os
import tifffile

def is_connected(pixel_value, x, y, img):
    neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if 0 <= x + dx < img.shape[1] and 0 <= y + dy < img.shape[0]]
    return all(img[neighbor[1], neighbor[0]] == pixel_value for neighbor in neighbors)

def find_max_connected_component(img):
    height, width = img.shape
    labels = {}
    current_label = 0
    for y in range(height):
        for x in range(width):
            pixel = img[y, x]
            if pixel == 2:
                neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if 0 <= x + dx < width and 0 <= y + dy < height]
                neighbor_labels = [labels.get((nx, ny), 0) for nx, ny in neighbors if (nx, ny) in labels and img[ny, nx] == 2]
                if not neighbor_labels:
                    current_label += 1
                    labels[(x, y)] = current_label
                else:
                    min_label = min(neighbor_labels)
                    labels[(x, y)] = min_label
                    for label in set(neighbor_labels):
                        if label!= min_label:
                            for coord, lbl in labels.items():
                                if lbl == label:
                                    labels[coord] = min_label
    return labels

def convert_png_to_yolo_label_opencv_with_tif(png_folder_path, tif_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for png_filename in os.listdir(png_folder_path):
        if png_filename.endswith(".png"):
            # 提取 PNG 文件的前缀作为基础名称
            base_name = png_filename.split('_mask')[0]
            tif_filename = None
            for file in os.listdir(tif_folder_path):
                if file.startswith(base_name):
                    tif_filename = file
                    break
            if tif_filename is None:
                raise ValueError(f"No corresponding tif file found for {png_filename}")

            png_path = os.path.join(png_folder_path, png_filename)
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            connected_components = find_max_connected_component(img)
            unique_labels = set(connected_components.values())
            labels = []
            for label in unique_labels:
                points = [(x, y) for (x, y), lbl in connected_components.items() if lbl == label]
                if points:
                    min_x = min([p[0] for p in points])
                    max_x = max([p[0] for p in points])
                    min_y = min([p[1] for p in points])
                    max_y = max([p[1] for p in points])
                    object_width = max_x - min_x
                    object_height = max_y - min_y
                    if object_width == 0:
                        object_width = 1
                    if object_height == 0:
                        object_height = 1
                    x_center = (min_x + object_width / 2) / img.shape[1]
                    y_center = (min_y + object_height / 2) / img.shape[0]
                    object_width /= img.shape[1]
                    object_height /= img.shape[0]
                    labels.append(f"0 {x_center} {y_center} {object_width} {object_height}")

            txt_filename = os.path.splitext(tif_filename)[0] + ".txt"
            txt_path = os.path.join(output_folder_path, txt_filename)
            with open(txt_path, "w") as f:
                f.write("\n".join(labels))

png_folder = r"C:\Users\zoujialin\Desktop\org_cut\mask1\test"
tif_folder = r"C:\Users\zoujialin\Desktop\org_cut\images\test"
output_folder = r"C:\Users\zoujialin\Desktop\org_cut\labels\test"
convert_png_to_yolo_label_opencv_with_tif(png_folder, tif_folder, output_folder)