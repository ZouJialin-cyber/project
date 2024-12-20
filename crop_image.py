"""
    对输入大图进行裁剪处理，分别裁剪为top, left, right和bottom，每条边各裁剪5000像素

    文件命名方式为*_top_0.tif,

    top和left的offset为0, bottom的offset为h-5000, right的offset为w-5000

    输入：文件夹，包含大图
    输出：主文件夹，内部会创建子文件夹，子文件夹包含子图
"""

import os
import cv2
import numpy as np
import tifffile as tif


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

def split_image(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for image_name in os.listdir(input_folder):
        if image_name.endswith('.tif'):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path, -1)

            image = f_ij_16_to_8(image)

            # Get image dimensions
            h, w, _ = image.shape

            # Compute cut positions
            top_image = image[0:5000, :]  # Cut the top 5000 pixels
            left_image = image[:, 0:5000]  # Cut the left 5000 pixels
            right_image = image[:, w-5000:w]  # Cut the right 5000 pixels
            bottom_image = image[h-5000:h, :]  # Cut the bottom 5000 pixels

            # Create a subfolder for each large image based on its name (without the .tif extension)
            base_name = os.path.splitext(image_name)[0]
            subfolder_path = os.path.join(output_folder, base_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            # Save the sub-images with the appropriate naming conventions
            tif.imwrite(os.path.join(subfolder_path, f"{base_name}_top_0.tif"), top_image, compression='zlib')
            tif.imwrite(os.path.join(subfolder_path, f"{base_name}_left_0.tif"), left_image, compression='zlib')
            tif.imwrite(os.path.join(subfolder_path, f"{base_name}_right_{w-5000}.tif"), right_image, compression='zlib')
            tif.imwrite(os.path.join(subfolder_path, f"{base_name}_bottom_{h-5000}.tif"), bottom_image, compression='zlib')

            print(f"Processed {image_name}")

# Example usage
input_folder = r"C:\Users\zoujialin\Desktop\gold\test\raw_img"
output_folder = r"C:\Users\zoujialin\Desktop\gold\test\mini_image"
split_image(input_folder, output_folder)