"""处理图像分类推理的mask, 获得最小外接矩形"""

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
import os


Image.MAX_IMAGE_PIXELS = None


def process_image(image_path):
    # 打开图像并转换为numpy数组（确保是灰度图像）
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    height, width = img_array.shape
    # 按密度找噪声并处理
    # for row in range(height):
    #     for col in range(width):
    #         if img_array[row, col] == 255:
    #             region_size = 0
    #             white_pixel_count = 0
    #             for sub_row in range(max(0, row - 1), min(height, row + 2)):
    #                 for sub_col in range(max(0, col - 1), min(width, col + 2)):
    #                     region_size += 1
    #                     if img_array[sub_row, sub_col] == 255:
    #                         white_pixel_count += 1
    #             density = white_pixel_count / region_size if region_size > 0 else 0
    #             if density < 0.3:
    #                 img_array[row, col] = 0

    # 先列遍历找上下255的边界
    col_result = np.zeros_like(img_array)
    for col in range(width):
        col_pixels = img_array[:, col]
        if np.any(col_pixels == 255):
            top_255_index = None
            bottom_255_index = None
            for row in range(height):
                if col_pixels[row] == 255:
                    if top_255_index is None:
                        top_255_index = row
                    bottom_255_index = row
            if top_255_index is not None and bottom_255_index is not None:
                col_result[top_255_index:bottom_255_index + 1, col] = 255

    # 再行遍历找左右255的边界
    row_result = np.zeros_like(img_array)
    for row in range(height):
        row_pixels = img_array[row, :]
        if np.any(row_pixels == 255):
            left_255_index = None
            right_255_index = None
            for col in range(width):
                if row_pixels[col] == 255:
                    if left_255_index is None:
                        left_255_index = col
                    right_255_index = col
            if left_255_index is not None and right_255_index is not None:
                row_result[row, left_255_index:right_255_index + 1] = 255

    # 取行列遍历的交集
    final_img_array = np.zeros_like(img_array)
    for row in range(height):
        for col in range(width):
            if col_result[row, col] == 255 and row_result[row, col] == 255:
                final_img_array[row, col] = 255

    # 对处理后的图像进行连通域标记
    labeled_img = label(final_img_array)
    regions = regionprops(labeled_img)
    if not regions:
        return Image.fromarray(np.zeros_like(final_img_array).astype('uint8'))

    final_result = np.zeros_like(final_img_array)
    # for region in regions:
    #     final_result[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] = 255
    for region in regions:
        bbox = region.bbox
        region_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if region_area <= 100 * 100:
            final_result[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
        else:
            final_result[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 255

    return Image.fromarray(final_result.astype('uint8'))

def process_mask_main():
    # 图像分类的mask路径
    original_folder = r"/storeData/USER/data/01.CellBin/00.user/zoujialin/MobileNetV3-master/process_3/inference_raw_processed_mask/class_mask"
    # 新文件夹路径, 转换为最小外接矩形的mask
    new_folder = r"/storeData/USER/data/01.CellBin/00.user/zoujialin/MobileNetV3-master/process_3/inference_raw_processed_mask/processed_mask"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # 遍历原始文件夹中的图像文件
    for filename in os.listdir(original_folder):
        print(f"processing\t{filename}")
        # 处理以.png、.jpg、.jpeg结尾的图像文件
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(original_folder, filename)
            result_image = process_image(image_path)
            new_image_path = os.path.join(new_folder, filename)
            result_image.save(new_image_path)

if __name__ == '__main__':
    process_mask_main()