"""
图像分类mask映射到组织分割mask得到后处理的mask，然后将三个mask分别映射到原图
"""

import os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2

# 定义输入文件夹路径
classification_mask_folder = '/storeData/USER/data/01.CellBin/00.user/zoujialin/MobileNetV3-master/process_3/已验收/inference_raw_processed_mask/class_mask'
segmentation_mask_folder = '/storeData/USER/data/01.CellBin/00.user/zoujialin/MobileNetV3-master/process_3/tissue_inference'
original_image_folder = '/storeData/USER/data/01.CellBin/00.user/zoujialin/MobileNetV3-master/process_3/test_img_png'

# 定义输出文件夹路径
combined_output_folder = '/storeData/USER/data/01.CellBin/00.user/zoujialin/MobileNetV3-master/process_3/post_merge'
new_mask_output_folder = '/storeData/USER/data/01.CellBin/00.user/zoujialin/MobileNetV3-master/process_3/post_mask'

# 遍历分类结果的 mask 文件夹
for classification_mask_filename in os.listdir(classification_mask_folder):
    # 提取文件名（不含后缀）
    classification_mask_name_without_ext = os.path.splitext(classification_mask_filename)[0]
    classification_mask_name_without_ext = classification_mask_name_without_ext.replace('_mask', '')

    # 根据名称对应关系找到分割结果的 mask 和原图
    segmentation_mask_filename = classification_mask_name_without_ext + '.png'
    original_image_filename = classification_mask_name_without_ext + '.png'

    classification_mask_path = os.path.join(classification_mask_folder, classification_mask_filename)
    segmentation_mask_path = os.path.join(segmentation_mask_folder, segmentation_mask_filename)
    original_image_path = os.path.join(original_image_folder, original_image_filename)

    # 读取图像
    classification_mask = np.array(Image.open(classification_mask_path))
    segmentation_mask = np.array(Image.open(segmentation_mask_path))
    original_image = np.array(Image.open(original_image_path))

    # 将分类结果的 mask 映射到分割结果的 mask
    new_segmentation_mask = np.where(classification_mask == 0, 0, segmentation_mask)

    # 对新生成的 mask 进行保存
    new_mask_filename = classification_mask_name_without_ext + '_new_mask.png'
    new_mask_path = os.path.join(new_mask_output_folder, new_mask_filename)
    Image.fromarray(new_segmentation_mask).save(new_mask_path)

    # 对所有图像进行八倍降采样
    original_image_resized = cv2.resize(original_image, None, fx=1/8, fy=1/8)
    classification_mask_resized = cv2.resize(classification_mask, None, fx=1/8, fy=1/8)
    new_segmentation_mask_resized = cv2.resize(new_segmentation_mask, None, fx=1/8, fy=1/8)
    segmentation_mask_resized = cv2.resize(segmentation_mask, None, fx=1/8, fy=1/8)

    # 将二维的 classification_mask_resized 扩展为三维，与 original_image_resized 形状匹配
    classification_mask_resized_3d = np.expand_dims(classification_mask_resized, axis=-1).repeat(3, axis=-1)
    new_segmentation_mask_resized_3d = np.expand_dims(new_segmentation_mask_resized, axis=-1).repeat(3, axis=-1)
    segmentation_mask_resized_3d = np.expand_dims(segmentation_mask_resized, axis=-1).repeat(3, axis=-1)

    # 将 mask 映射到原图上，并设置对比度为 0.5
    alpha = 0.5
    red_color = np.array([255, 0, 0])
    original_image_with_classification_mask = np.where(classification_mask_resized_3d == 255,
                                                       alpha * original_image_resized + (1 - alpha) * red_color,
                                                       original_image_resized)
    original_image_with_new_segmentation_mask = np.where(new_segmentation_mask_resized_3d == 255,
                                                         alpha * original_image_resized + (1 - alpha) * red_color,
                                                         original_image_resized)
    original_image_with_segmentation_mask = np.where(segmentation_mask_resized_3d == 255,
                                                     alpha * original_image_resized + (1 - alpha) * red_color,
                                                     original_image_resized)

    # 创建一个大的画布，将四张图片拼接到一起
    combined_image = np.zeros((original_image_resized.shape[0] * 2, original_image_resized.shape[1] * 2, 3), dtype=np.uint8)
    combined_image[0:original_image_resized.shape[0], 0:original_image_resized.shape[1]] = original_image_resized
    combined_image[0:original_image_resized.shape[0], original_image_resized.shape[1]:] = original_image_with_classification_mask
    combined_image[original_image_resized.shape[0]:, 0:original_image_resized.shape[1]] = original_image_with_new_segmentation_mask
    combined_image[original_image_resized.shape[0]:, original_image_resized.shape[1]:] = original_image_with_segmentation_mask

    # 添加标题
    cv2.putText(combined_image, 'Original Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_image, 'Classification Mask on Image', (original_image_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_image, 'New Segmentation Mask on Image', (10, original_image_resized.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_image, 'Segmentation Mask on Image', (original_image_resized.shape[1] + 10, original_image_resized.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 保存拼接后的图像
    output_filename = classification_mask_name_without_ext + '_combined.png'
    output_path = os.path.join(combined_output_folder, output_filename)
    cv2.imwrite(output_path, combined_image)