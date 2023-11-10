import numpy as np
import tensorflow as tf
import sys
import os
import time
from PIL import Image
sys.path.append("..")  # 添加上一级目录到sys.path
from load_model import get_image

# 模型路径
model_path = '/root/dacheng/Tensorflow-Train/tflite/quantized_model.tflite'

# 输入张量和输出张量的索引
input_index = 0  # 输入张量的索引
output_index = 0  # 输出张量的索引

# 加载模型
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 获取输入张量和输出张量的详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.resize_tensor_input(input_details[0]['index'], (1, 32, 32, 3))
interpreter.allocate_tensors()  # 重新分配

# 定义图像尺寸和通道数
input_shape = input_details[0]['shape']
image_width, image_height, channels = 32, 32, 3

# 定义图像预处理函数
def preprocess_image_int8(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image_width, image_height))
    # 图片本身就是int8型，因此不需要再进行放缩
    # image = np.array(image) / 255.0  # 将像素值缩放到0-1范围
    image = np.expand_dims(image, axis=0)  # 增加batch维度
    return image

def preprocess_image_float32(image_path):
    image = Image.open(image_path)
    image = image.resize((image_width, image_height))
    image = np.array(image, dtype=np.float32) / 255.0  # 将像素值缩放到0-1范围，并转换为float32类型
    print(np.mean(image))
    image = np.expand_dims(image, axis=0)  # 增加batch维度
    print(image.shape)
    return image

# 获取置信度最高的类别索引
def process_output(output_data):
    # 在这里编写你的输出结果处理逻辑
    class_id = np.argmax(output_data)  
    # print(class_id)
    # confidence = output_data[class_id]  # 获取置信度最高的类别的置信度值

    # 返回处理后的结果
    return class_id

# 类别标签映射
def class_mapping():
    return {
        0: 'Attack',
        1: 'CC',
        2: 'CC-HeartBeat',
        3: 'CSE-CIC-IDS2018',
        4: 'Delivery',
        5: 'InsteonCamera',
        6: 'IoT-23',
        7: 'NetatmoWelcome',
        8: 'PartOfAHorizontalPortScan',
        9: 'TelnetPenetration'
    }

# 根据地址获取类别
def get_class(file):
    # 使用split方法拆分路径
    split_path = file.split("/")
    print(split_path[6])
    return(split_path[6])

# 读取数据集文件夹中的图像文件
dataset_folder = '/root/dacheng/Tensorflow-Train/dataset/cv_ids_25/test'
image_paths = []
labels = []
label_index = 0
# 获取映射
class_mapping = class_mapping()

# 添加标签
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):

            image_paths.append(os.path.join(root, file))
            # 添加标签，用于检验，同一个文件夹的类别相同
            labels.append(label_index-1)
            # print(labels)
            # time.sleep(0.1)
    label_index += 1

# 创建一个数组来保存推理结果
results = []

# 初始化计时器和累计推理时间
total_inference_time = 0

# 遍历图像文件并进行推理
for image_path, label in zip(image_paths, labels):
    # 预处理图像

    # 当模型为int8
    # input_data = preprocess_image_int8(image_path) 
    # 当模型为float32
    input_data = preprocess_image_float32(image_path)

    # print(input_data.shape)

    # 设置输入数据到模型的输入张量
    interpreter.set_tensor(input_details[input_index]['index'], input_data)

    # 开始计时
    start_time = time.time()

    # 运行推理
    interpreter.invoke()

    # 结束计时并计算推理时间
    end_time = time.time()
    inference_time = end_time - start_time

    # 累计推理时间和推理次数
    total_inference_time += inference_time

    # 获取输出结果
    output_data = interpreter.get_tensor(output_details[output_index]['index'])

    # 处理输出结果
    processed_output = process_output(output_data)  # 需根据你的处理逻辑实现

    # 将结果保存到数组中，包括图像路径、标签和推理结果
    results.append({'image_path': image_path, 'label': label, 'result': processed_output})

    # 打印结果
    print("The class is: ",class_mapping[label],". The result is: ", class_mapping[processed_output])

# 计算准确率
correct_predictions = 0
total_predictions = len(results)

for result in results:
    if result['result'] == result['label']:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
average_inferences = total_inference_time / total_predictions
print(f"Accuracy: {accuracy * 100}%")
print(f"平均推理时间: {average_inferences * 1000}ms")