from ultralytics import YOLO
import os
import cv2
import socket
import mysql.connector

def send_image_data(image_data):
    client = socket.socket()
    client.connect(('192.168.85.128', 21011))
    client.send(image_data.tobytes())  # Send the image data directly
    print('Image data sent successfully')
    client.close()

def send_detected_frames(annotated_img):
    if annotated_img is not None:
        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        send_image_data(img_encoded)

class RoadDamageDatabase:
    def __init__(self, config):
        self.db = mysql.connector.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            database=config['database']
        )
        self.cursor = self.db.cursor()

    def update_road_data(self, class_stats, classes_to_update, deductions):
        update_query = """
        UPDATE road_data
        SET road_damage_num = %s, road_damage_area = %s, deduction_of_points = %s
        WHERE uid = %s
        """
        uid_counter = 1
        for class_name in classes_to_update:
            if class_name in class_stats:
                stats = class_stats[class_name]
                deduction = deductions.get(class_name, 0)  # Get deduction value, default to 0 if not found
                values = (stats['num'], stats['area_sum'], deduction, uid_counter)
                self.cursor.execute(update_query, values)
                uid_counter += 1  # Increment UID for each entry
        self.db.commit()

    def close(self):
        self.cursor.close()
        self.db.close()

# 计算病害密度
def calculate_density(class_stats, total_area):
    densities = {}
    for class_name, stats in class_stats.items():
        if total_area > 0:
            densities[class_name] = stats['area_sum'] / total_area
        else:
            densities[class_name] = 0
    return densities

# 插值法计算扣分值
def interpolate(density, damage_levels):
    if density == 0:
        return 0  # 如果密度为0，则直接返回0
    # 假设damage_levels按密度排序
    for i in range(len(damage_levels) - 1):
        if density == damage_levels[i][0]:
            return damage_levels[i][1]
        elif density < damage_levels[i+1][0]:
            # 线性插值
            x0, y0 = damage_levels[i]
            x1, y1 = damage_levels[i+1]
            return y0 + (y1 - y0) * ((density - x0) / (x1 - x0))
    return damage_levels[-1][1]  # 如果密度高于所有定义级别，则返回最高扣分值

# 计算PCI
def calculate_PCI(densities, damage_data):
    PCI = 100
    deductions = {}  # 存储每类病害的扣分值
    for class_name, density in densities.items():
        if class_name in damage_data:
            deduction_levels = damage_data[class_name]
            deduction = interpolate(density, deduction_levels)
            PCI -= 2 * deduction
            deductions[class_name] = deduction  # 存储每类病害的扣分值
    return PCI, deductions

model_path = r"F:\Yolov9\ultralytics-main\ultralytics-main\mycode\runs\detect\train28\train28\weights\best.pt"
video_path = '1_new.mp4'

#names: ['Alligator crack',  'Longitudinal cracks', 'Pothole', 'Repair', 'Transverse crack', 'White-line-wear']

interested_classes = [1,4,5]

# 定义类别名称列表
class_names =['Alligator crack',  'Longitudinal cracks', 'Pothole', 'Repair', 'Transverse crack', 'White-line-wear']

# 初始化类别统计字典
class_stats = {class_name: {'num': 0, 'area_sum': 0.0} for class_name in class_names}

if os.path.exists(model_path):
    print("Model file exists")
    model = YOLO(model_path)

    # 设置感兴趣的类别名称列表
    #model.names = interested_classes

    results = model.predict(source=video_path, stream=False, save_txt=True, classes=interested_classes,conf=0.3, iou=0.3)  # 进行预测  conf=0.3, iou=0.3

    for det in results:
        if det is not None and det.boxes is not None:
            annotated_img = det.plot()  # 获取带有标注的图像

            # 指定要保存的txt文件路径并保存
            txt_file_name = 'detected_objects.txt'
            det.save_txt(txt_file=txt_file_name, save_conf=True)  # 保存检测结果到txt文件

            # 检查文件是否存在后再读取并输出生成的txt文件内容
            if os.path.exists(txt_file_name):
                with open(txt_file_name, 'r') as file:
                    content = file.read()
                    print("Generated txt file content:")
                    print(content)

                # 如果内容不为空，则发送带有标注的图像数据
                if content.strip():  # 检查内容是否为空
                    send_detected_frames(annotated_img)  # 发送带有标注的图像数据
                    print("Annotated image data sent.")
                    #统计数据
                    # 读取并解析txt文件中的每行数据进行分类统计
                    with open(txt_file_name, 'r') as file:
                        for line in file:
                            parts = line.split()
                            if len(parts) >= 6:
                                class_idx = int(parts[0])  # 获取类别索引
                                if class_idx < len(class_names):
                                    area = float(parts[3]) * 1920 * 0.0045054 * float(parts[4]) * 1080 * 0.0045054  # 计算面积  检测框的宽度和高度相乘
                                    class_name = class_names[class_idx]  # 获取类别名称
                                    class_stats[class_name]['num'] += 1
                                    class_stats[class_name]['area_sum'] += area
                else:
                    print("No annotated image data sent as txt file content is empty.")

                # 删除txt文件
                os.remove(txt_file_name)
                print("Temporary txt file deleted.")
            else:
                print(f"File '{txt_file_name}' not found.")
    # 打印分类统计结果
    for class_name, stats in class_stats.items():
        print(f"{class_name}: 数量 = {stats['num']}, 总面积 = {stats['area_sum']} 平方米")
    # 计算PCI指数
    # 示例用法
    # 假设总道路面积（平方米）
    total_road_area = 6.50 * 200  # 示例值
    # 计算密度
    densities = calculate_density(class_stats, total_road_area)
    # 包含密度级别和对应扣分的病害数据
    damage_data = {
        'Alligator crack': [(0.01, 8), (0.1, 10), (1, 15), (10, 30), (50, 55), (100, 80)],
        'Gnaw the edges': [(0.01, 2), (0.1, 4), (1, 8), (10, 15), (50, 30), (100, 40)],
        'Longitudinal cracks': [(0.01, 3), (0.1, 5), (1, 8), (10, 16), (50, 38), (100, 48)],
        'Pothole': [(0.01, 10), (0.1, 15), (1, 25), (10, 40), (50, 65), (100, 72)],
        'Transverse crack': [(0.01, 3), (0.1, 5), (1, 8), (10, 16), (50, 38), (100, 48)],
    }
    # 计算PCI和每类病害的扣分值
    result_PCI, deductions = calculate_PCI(densities, damage_data)
    # 输出每类病害的扣分值
    for class_name, deduction in deductions.items():
        print(f"{class_name} 的扣分值为: {deduction}")
    print("计算得到的PCI:", result_PCI)
    # 将数据存入数据库中
    db_config = {'host': '192.168.85.128', 'user': 'bsqpb5g6e', 'password': '9t1yctq3', 'database': 'bsqpb5g6e'}
    database = RoadDamageDatabase(db_config)
    classes_to_update = ['Alligator crack',  'Longitudinal cracks', 'Pothole', 'Transverse crack', 'White-line-wear']
    database.update_road_data(class_stats, classes_to_update, deductions)
    database.close()
else:
    print("Model file not found")


