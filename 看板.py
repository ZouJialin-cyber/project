import openpyxl
from openpyxl.drawing.image import Image
import os

Image.MAX_IMAGE_PIXELS = None

excel_path = r"C:\Users\zoujialin\Desktop\10.xlsx"

# 打开Excel文件
wb = openpyxl.load_workbook(excel_path)

# 选择活动工作表
sheet = wb.active

# 图片文件夹路径
image_folder = r"C:\Users\zoujialin\Desktop\gold\demo\gold\matrix\out\Top-Left"

# 假设SN号从第二行开始（第一行是标题行）
sn_start_row = 2

# 遍历图片文件夹中的图片文件
for i, img_file in enumerate(sorted(os.listdir(image_folder))):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(image_folder, img_file)
        img = Image(img_path)

        # 计算当前图片对应的行（与SN号对齐）
        row = sn_start_row + i
        col = 6

        # 设置图片大小（可选，根据需要调整宽度和高度）
        img.width = 200  # 设置图片宽度为100像素（可根据实际需求调整）
        img.height = 200  # 设置图片高度为100像素（可根据实际需求调整）

        # 将图片插入到指定单元格
        sheet.add_image(img, f'{openpyxl.utils.get_column_letter(col)}{row}')

        # 根据图片大小调整单元格列宽和行高
        sheet.column_dimensions[openpyxl.utils.get_column_letter(col)].width = img.width / 6.75  # 这里的除数是根据经验调整的，以使列宽看起来合适，你可以根据实际情况微调
        sheet.row_dimensions[row].height = img.height

# 保存修改后的Excel文件
wb.save("C:/Users/zoujialin/Desktop/10_01.xlsx")