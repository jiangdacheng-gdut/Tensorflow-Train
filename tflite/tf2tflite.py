import tensorflow as tf
import sys
import numpy as np
import os
from PIL import Image
sys.path.append("..")  # 添加上一级目录到sys.path
from load_model import loaded_model

def load_and_preprocess_image(path, target_size=(32, 32)):
    image = Image.open(path)
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # 归一化
    return image

def representative_dataset_gen():
    dataset_dir = '/root/dacheng/Tensorflow-Train/dataset/cv_ids_25/test'  # 替换为你的数据集路径
    num_calibration_images = 100  # 你想用于校准的图片数量
    count = 0

    for class_dir in os.listdir(dataset_dir):
        class_dir_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_dir_path):
            for image_name in os.listdir(class_dir_path):
                if count >= num_calibration_images:
                    return
                image_path = os.path.join(class_dir_path, image_name)
                image = load_and_preprocess_image(image_path)
                image = np.expand_dims(image, axis=0)  # 扩展维度以匹配模型的输入
                image = image.astype(np.float32)
                yield [image]
                count += 1

model_path = '/root/dacheng/Tensorflow-Train/output/best_model1030'
loaded = loaded_model(model_path)
concrete_func = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# 设定输入形状
input_shape = [1, 32, 32, 3]
concrete_func.inputs[0].set_shape(input_shape)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, 
                                       tf.lite.OpsSet.SELECT_TF_OPS]

# 启用后训练量化
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 设置代表性数据集以校准量化
# converter.representative_dataset = representative_dataset_gen

# 进行转换
tflite_quantized_model = converter.convert()

# 保存转换后的量化模型
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_quantized_model)