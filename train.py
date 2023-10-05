import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint
from module import *

# 数据集位置
dataset_path = '/root/dacheng/Tensorflow-Train/dataset/cv_ids_25'

# 加载配置文件
config_path = '/root/dacheng/Tensorflow-Train/net.config'
with open(config_path, 'r') as f:
    config = json.load(f)

# 定义损失函数
def get_loss_fn(label_smoothing=0.1):
    def cross_entropy_with_label_smoothing(y_true, y_pred):
        # 获取类别数量
        num_classes = 10
        
        # 计算真实标签与平滑标签之间的权重
        y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
        
        # 计算交叉熵损失
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        
        return loss
    return cross_entropy_with_label_smoothing

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
optimizer = Adam(learning_rate=0.001)
loss_fn = get_loss_fn(label_smoothing=0.1)
model.compile(optimizer, 
              loss=loss_fn, 
              metrics=[CategoricalAccuracy()])
model.build(input_shape=(None, img_height, img_width, 3))
model.summary()

# 创建ModelCheckpoint回调
checkpoint_path = '/root/dacheng/Tensorflow-Train/output/best_model'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1, 
                             save_best_only=True, mode='max', save_format='tf')

# 训练模型
epochs = 10  # 选择合适的epoch数
# history = model.fit(
#     train_generator,
#     epochs=epochs,
#     validation_data=validation_generator
# )
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint]  # 添加回调
)

# 评估模型
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")

# 保存模型
# model.save('/root/dacheng/Tensorflow-Train/output/model', save_format='tf')
