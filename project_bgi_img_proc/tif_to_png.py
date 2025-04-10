import os

import cv2
import numpy as np
import tifffile
from PIL import Image


class ImageConverter:
    def f_ij_16_to_8(self, img, chunk_size=1000):
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


if __name__ == "__main__":
    converter = ImageConverter()
    # 原始tif图片文件夹路径
    tif_folder = r"D:\HE_program\HE\HE_data\chip\he_data_stage_1\out\test2\左下角"
    # 保存png图片的新文件夹路径
    png_folder = r"D:\HE_program\HE\HE_data\chip\he_data_stage_1\out\test3\crop_left_btm"
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)
    for tif_file in os.listdir(tif_folder):
        if tif_file.endswith(".tif"):
            tif_path = os.path.join(tif_folder, tif_file)
            img = tifffile.imread(tif_path)
            img = converter.f_ij_16_to_8(img)
            img_pil = Image.fromarray(img)
            # img_pil = np.array(img_pil)
            # img_pil = cv2.resize(img_pil, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            # img_pil = Image.fromarray(img_pil)
            png_file = os.path.splitext(tif_file)[0] + ".png"
            png_path = os.path.join(png_folder, png_file)
            img_pil.save(png_path)