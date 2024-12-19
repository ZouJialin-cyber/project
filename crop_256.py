import cv2
import os
import tifffile as tif

def convert_images_in_folder(folder_path):
    """
    遍历指定文件夹下的所有.tif图像，将三通道图像转换为单通道图像（灰度图）并覆盖原图。

    参数:
    folder_path (str): 需要遍历的文件夹路径，该文件夹下存放着.tif图像文件。
    """
    # 遍历文件夹下的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tif'):
            file_path = os.path.join(folder_path, file_name)
            # 读取图像
            image = cv2.imread(file_path, -1)
            if image is not None:
                # 判断图像通道数
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # 将三通道图像转换为单通道图像（灰度图），使用cv2.cvtColor函数结合COLOR_BGR2GRAY参数实现
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # 保存转换后的图像覆盖原图
                    tif.imwrite(dest_folder, gray_image, compression='zlib')
                    print(f"{file_name} 已从三通道转换为单通道。")
                else:
                    print(f"{file_name} 为单通道图像，无需处理。")
            else:
                print(f"无法读取 {file_name}，可能文件格式有误或已损坏。")


if __name__ == "__main__":
    # 指定要遍历的文件夹路径，这里请替换为你实际存放.tif图像的文件夹路径
    target_folder = "your_folder_path"
    dest_folder = ""
    convert_images_in_folder(target_folder)