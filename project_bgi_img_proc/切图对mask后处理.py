import os
import tifffile
import numpy as np
from PIL import Image

image_folder = r"C:\Users\zoujialin\Desktop\org_cut\img_C02533C2"
mask_folder = r"C:\Users\zoujialin\Desktop\org_cut\mask_C02533C2"
output_folder = r"C:\Users\zoujialin\Desktop\org_cut\output_C02533C2"

os.makedirs(output_folder, exist_ok=True)

for i in range(644):
    image_path = os.path.join(image_folder, f'cropped_image_{i}.tif')
    mask_path = os.path.join(mask_folder, f'cropped_mask_{i}.tif')
    img_data = tifffile.imread(image_path)
    mask_data = tifffile.imread(mask_path)
    # 修改 mask 的像素值
    new_mask_data = np.where(mask_data == 1, 2, 1)
    # 将修改后的 mask 保存为 png 格式，文件名与对应的图像文件名一致加上_mask
    new_mask_filename = os.path.join(output_folder, f'cropped_image_{i}_mask.png')
    img = Image.fromarray(new_mask_data.astype(np.uint8))
    img.save(new_mask_filename)