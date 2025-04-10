# 二阶段框检测数据预处理

import cv2
import numpy as np
import tifffile as tiff
import copy
import os
import random
from tqdm import tqdm


# 从 txt 文件中读取顶点坐标
def read_points_from_txt(file_path):
    with open(file_path, 'r') as file:
        line = file.readline().strip()  # 读取第一行
        line = line[2:]
        points = list(map(float, line.split()))  # 将字符串分割成浮点数
    return points


def t_int16_to_int8(img, chunk_size=1000):
    """
    16 bits img to 8 bits

    :param img: (CHANGE) np.array
    :param chunk_size: chunk size (bit)
    :return: np.array
    """

    if img.dtype == 'uint8':
        return img
    dst = np.zeros(img.shape, np.uint8)
    p_max = np.max(img)
    p_min = np.min(img)
    scale = 256.0 / (p_max - p_min + 1)
    for idx in range(img.shape[0] // chunk_size + 1):
        sl = slice(idx * chunk_size, (idx + 1) * chunk_size)
        win_img = copy.deepcopy(img[sl])
        win_img = np.int16(win_img)
        win_img = (win_img & 0xffff)
        win_img = win_img - p_min
        win_img[win_img < 0] = 0
        win_img = win_img * scale + 0.5
        win_img[win_img > 255] = 255
        dst[sl] = np.array(win_img).astype(np.uint8)
    return dst


def calculate_rotation_angle(points):
    """计算左上角和右上角的线段与水平轴的夹角"""
    # 排序点，以获取 y 最小的两个点
    sorted_points = sorted(points, key=lambda p: p[1])

    # 取 y 最小的两个点
    y_min_points = sorted_points[:2]

    # 从 y 最小的两个点中选择 x 最小和 x 最大的点
    p1 = min(y_min_points, key=lambda p: p[0])  # x 最小的点
    p2 = max(y_min_points, key=lambda p: p[0])  # x 最大的点

    # 计算线段 p1p2 的角度
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    print(dx, dy)  # 输出 dx 和 dy 以检查计算是否正确
    angle = np.arctan2(dy, dx)

    # 将角度转换为度制
    angle_degrees = np.degrees(angle)

    return angle_degrees


def rotate_image(image, angle):
    """旋转图像"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 旋转图像
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return rotated_image


def rotate_points(points, angle, image_width, image_height):
    """旋转顶点坐标"""
    # 将归一化坐标转换为实际坐标
    points = np.array(points).reshape(-1, 2)
    points[:, 0] *= image_width
    points[:, 1] *= image_height

    # 计算旋转矩阵
    center = (image_width // 2, image_height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 旋转角度取负值

    # 应用旋转矩阵
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    rotated_points_homogeneous = M @ points_homogeneous.T
    rotated_points = rotated_points_homogeneous[:2].T

    return rotated_points


def pad_image_point(image, points, add_size):
    img_add = cv2.copyMakeBorder(image, add_size, add_size, add_size, add_size, cv2.BORDER_CONSTANT, value=(0,0,0))    # 三通道和单通道区分开
    for i in range(4):
        for j in range(2):
            points[i][j] += add_size

    return img_add, points

def main():
    # 输入四个顶点坐标

    txt_file_dir = r'/storeData/USER/data/01.CellBin/00.user/zoujialin/HE_chip_program/dataset/label/'
    img_dir = r'/storeData/USER/data/01.CellBin/00.user/zoujialin/HE_chip_program/dataset/raw_img/'
    out_img_dir = r'/storeData/USER/data/01.CellBin/00.user/zoujialin/HE_chip_program/dataset/out_img/'
    out_label_dir = r'/storeData/USER/data/01.CellBin/00.user/zoujialin/HE_chip_program/dataset/out_label/'
    wa = []

    for sn_txt in tqdm(os.listdir(img_dir)):

        sn_f = sn_txt.split('.')[0]
        txt_file_path = os.path.join(txt_file_dir, f'{sn_f}.txt')
        img_path = os.path.join(img_dir, f'{sn_f}.tif')

        out_img_path_1 = os.path.join(out_img_dir, f'{sn_f}_1.tif')
        out_img_path_2 = os.path.join(out_img_dir, f'{sn_f}_2.tif')
        out_img_path_3 = os.path.join(out_img_dir, f'{sn_f}_3.tif')
        out_img_path_4 = os.path.join(out_img_dir, f'{sn_f}_4.tif')

        out_label_path_1 = os.path.join(out_label_dir, f'{sn_f}_1.txt')
        out_label_path_2 = os.path.join(out_label_dir, f'{sn_f}_2.txt')
        out_label_path_3 = os.path.join(out_label_dir, f'{sn_f}_3.txt')
        out_label_path_4 = os.path.join(out_label_dir, f'{sn_f}_4.txt')

        print(sn_f)


        box_size = 1000
        normalized_points = read_points_from_txt(txt_file_path)
        img_raw = tiff.imread(img_path)
        img_raw.squeeze()
        # if len(img_raw.shape) != 2:    # 单通道专属
        #     img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        # a, b = img_raw.shape    单通道
        a, b, c = img_raw.shape

        if a >= 32767 or b >= 32767:
            # cv2 旋转尺寸上限
            wa.append(sn_f)
            continue
        img = t_int16_to_int8(img_raw)
        # img = cv2.equalizeHist(img_raw)
        image_height, image_width, image_channel = img.shape

        points = np.array(normalized_points).reshape(-1, 2)
        points[:, 0] *= image_width
        points[:, 1] *= image_height
        # 计算旋转角度
        angle = calculate_rotation_angle(points)

        # 旋转图像
        print(angle)
        rotated_image = rotate_image(img, angle)
        rotated_points = rotate_points(normalized_points, angle, image_width, image_height)
        # print(points)
        # print(rotated_points)
        print(calculate_rotation_angle(rotated_points))

        padded_img, padded_points = pad_image_point(rotated_image, rotated_points, 1300)

        padded_points = padded_points.astype(int)

        sorted_points_list = []

        min_x = np.min(padded_points[:, 0])
        max_x = np.max(padded_points[:, 0])
        min_y = np.min(padded_points[:, 1])
        max_y = np.max(padded_points[:, 1])




        # tiff.imwrite(r'D:\data\box_data\two\ssDNA\img\SS200000863BR_E2_llll.tif', padded_img)


        sorted_points_list.append([min_x, min_y])
        sorted_points_list.append([max_x, min_y])
        sorted_points_list.append([min_x, max_y])
        sorted_points_list.append([max_x, max_y])



        # 计算裁剪区域
        # 一号图左上
        randx = random.randint(-300, 300)
        randy = random.randint(-300, 300)
        x, y = sorted_points_list[0]
        # cropped_image_1 = padded_img[y - box_size: y + box_size, x - box_size: x + box_size]
        cropped_image_1 = padded_img[y-box_size+randy: y+box_size+randy, x-box_size+randx: x+box_size+randx, :]    # 三通道
        x0 = 1000 - randx
        y0 = 1000 - randy
        x1 = 2000
        y1 = 2000
        width = 2000
        height = 2000
        w = x1 - x0
        h = y1 - y0
        xc = int((x0+x1)/2)
        yc = int((y0+y1)/2)
        txt_string_1 = '{} {} {} {} {}\n'.format(0, xc / width, yc / height, w / width, h / height)
        # cropped_image_1 = cv2.equalizeHist(cropped_image_1)    # 单通道(直方图均衡化)
        tiff.imwrite(out_img_path_1, cropped_image_1)
        with open(out_label_path_1, 'w') as txt_file:
            txt_file.writelines(txt_string_1)

        # 二号图右上
        randx = random.randint(-300, 300)
        randy = random.randint(-300, 300)
        x, y = sorted_points_list[1]
        cropped_image_2 = padded_img[y-box_size+randy: y+box_size+randy, x-box_size+randx: x+box_size+randx, :]    # 三通道
        x0 = 1000 - randx
        y0 = 1000 - randy
        x1 = 0
        y1 = 2000
        width = 2000
        height = 2000
        w = x0 - x1
        h = y1 - y0
        xc = int((x0+x1)/2)
        yc = int((y0+y1)/2)
        txt_string_2 = '{} {} {} {} {}\n'.format(0, xc / width, yc / height, w / width, h / height)
        # cropped_image_2 = cv2.equalizeHist(cropped_image_2)    # 单通道(直方图均衡化)
        tiff.imwrite(out_img_path_2, cropped_image_2)
        with open(out_label_path_2, 'w') as txt_file:
            txt_file.writelines(txt_string_2)

        # 三号图左下
        randx = random.randint(-300, 300)
        randy = random.randint(-300, 300)
        x, y = sorted_points_list[2]
        cropped_image_3 = padded_img[y-box_size+randy: y+box_size+randy, x-box_size+randx: x+box_size+randx, :]    # 三通道
        x0 = 1000 - randx
        y0 = 1000 - randy
        x1 = 2000
        y1 = 0
        width = 2000
        height = 2000
        w = x1 - x0
        h = y0 - y1
        xc = int((x0+x1)/2)
        yc = int((y0+y1)/2)
        txt_string_3 = '{} {} {} {} {}\n'.format(0, xc / width, yc / height, w / width, h / height)
        # cropped_image_3 = cv2.equalizeHist(cropped_image_3)    # 单通道直方图均衡化
        tiff.imwrite(out_img_path_3, cropped_image_3)
        with open(out_label_path_3, 'w') as txt_file:
            txt_file.writelines(txt_string_3)

        # 四号图右下
        randx = random.randint(-300, 300)
        randy = random.randint(-300, 300)
        x, y = sorted_points_list[3]
        cropped_image_4 = padded_img[y-box_size+randy: y+box_size+randy, x-box_size+randx: x+box_size+randx, :]    # 三通道
        x0 = 1000 - randx
        y0 = 1000 - randy
        x1 = 0
        y1 = 0
        width = 2000
        height = 2000
        w = x0 - x1
        h = y0 - y1
        xc = int((x0+x1)/2)
        yc = int((y0+y1)/2)
        txt_string_4 = '{} {} {} {} {}\n'.format(0, xc / width, yc / height, w / width, h / height)
        # cropped_image_4 = cv2.equalizeHist(cropped_image_4)    # 单通道直方图均衡化
        tiff.imwrite(out_img_path_4, cropped_image_4)
        with open(out_label_path_4, 'w') as txt_file:
            txt_file.writelines(txt_string_4)

    print(len(wa))
    print(wa)

# 运行主函数
if __name__ == "__main__":
    main()
