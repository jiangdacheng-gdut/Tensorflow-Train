import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.applications import ImageDataGenerator
from module import *

# 数据集位置
dataset_path = '/Users/jiangdacheng/Desktop/8-Coding/tensorflow-proxylessnas/cv_ids_25'

# 加载配置文件
config_path = '/Users/jiangdacheng/Desktop/8-Coding/Tensorflow-Train/net.config'
with open(config_path, 'r') as f:
    config = json.load(f)

# 创建模型
model = MyModel(config)

# 数据预处理
batch_size = 16
img_height = 224  # 根据你的模型和数据集调整
img_width = 224   # 根据你的模型和数据集调整

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 假设我们使用20%的数据作为验证集

train_generator = train_datagen.flow_from_directory(
    dataset_path + '/train',   # 替换为数据集的路径
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path + '/test',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 创建、编译模型并查看模型摘要
model = MyModel(config)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, img_height, img_width, 3))
model.summary()

# 训练模型
epochs = 10  # 选择合适的epoch数
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# 评估模型
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")

# 保存模型
model.save('/Users/jiangdacheng/Desktop/8-Coding/tensorflow-proxylessnas/my_model.h5')
