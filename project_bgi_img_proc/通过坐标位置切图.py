from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None

def crop_png_image(png_path, x1, y1, x2, y2, output_folder):
    """
    对PNG图片进行坐标区域裁剪并保存结果

    :param png_path: PNG图片路径
    :param x1: 矩形左上角x坐标
    :param y1: 矩形左上角y坐标
    :param x2: 矩形右下角x坐标
    :param y2: 矩形右下角y坐标
    :param output_folder: 保存裁剪结果的文件夹路径
    """
    image = Image.open(png_path)
    cropped_image = image.crop((x1, y1, x2, y2))
    base_name = os.path.basename(png_path)
    output_path = os.path.join(output_folder, "crop_youhua_1_" + base_name)
    cropped_image.save(output_path)


# 示例用法
png_path = r"C:\Users\zoujialin\Desktop\youhua\png\B01020C2_img.png"
x1, y1, x2, y2 = 2352, 4752, 17808, 20256
output_folder = r"C:\Users\zoujialin\Desktop\youhua\ordinate_crop\1"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
crop_png_image(png_path, x1, y1, x2, y2, output_folder)