import cv2
import numpy as np
import os

def load_image(image_path):
    """加载图像并将其转换为float32类型，以便进行像素操作"""
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype('float32')

def apply_mask(image, mask):
    """将掩码应用到图像上"""
    # 将掩码转换为三通道
    mask_3channel = np.zeros_like(image)
    mask_3channel[mask == 255] = [255, 0, 0]  # 将掩码的255像素设置为红色
    masked_image = image * 0.5 + mask_3channel * 0.5
    return masked_image

def process_images(image_folder, mask_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历图像文件夹
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            # 构建完整的文件路径
            image_path = os.path.join(image_folder, filename)
            mask_filename = filename.replace('.png', '_mask.png')
            mask_path = os.path.join(mask_folder, mask_filename)

            # 检查mask文件是否存在
            if os.path.exists(mask_path):
                # 加载图像和mask
                image = load_image(image_path)
                mask = load_image(mask_path)

                # 应用掩码
                masked_image = apply_mask(image, mask)

                # 创建左右拼接的图片
                combined_image = np.concatenate((image, masked_image), axis=1)

                # 保存结果
                output_path = os.path.join(output_folder, f'{filename[:-4]}_combined.png')
                cv2.imwrite(output_path, (combined_image * 255).astype('uint8'))
                print(f'Processed and saved: {output_path}')
            else:
                print(f'Mask file not found for: {filename}')

# 定义文件夹路径
image_folder = r"C:\Users\zoujialin\Desktop\tissue_data\process_3\test_tif"  # 图像数据文件夹路径
mask_folder = r"C:\Users\zoujialin\Desktop\tissue_data\process_3\test_class_mask"    # Mask文件夹路径
output_folder = r"C:\Users\zoujialin\Desktop\tissue_data\process_3\output"  # 输出文件夹路径

# 处理图像
process_images(image_folder, mask_folder, output_folder)