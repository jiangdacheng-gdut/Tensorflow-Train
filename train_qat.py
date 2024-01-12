import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
import tensorflow_model_optimization as tfmot
from mymodule import build_model
from mymodule import MBInvertedConvLayer, MBInvertedConvQuantizeConfig

# 数据集位置、加载配置文件
dataset_path = '/root/dacheng/Tensorflow-Train/dataset/cv_ids_25'
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

# 数据预处理
batch_size = 16
img_height = 32  # 根据你的模型和数据集调整
img_width = 32   # 根据你的模型和数据集调整

train_datagen = ImageDataGenerator(rescale=1./255)  # 验证集的生成器
train_generator = train_datagen.flow_from_directory(
    dataset_path + '/train',   # 替换为数据集的路径
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    # subset='training'
)

test_datagen = ImageDataGenerator(rescale=1./255)  # 测试集的生成器
validation_generator = test_datagen.flow_from_directory(
    dataset_path + '/test',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    # subset='validation'
)

model = build_model(img_height, img_width, config)
optimizer = Adam(learning_rate=0.001)
loss_fn = get_loss_fn(label_smoothing=0.1)
model.compile(optimizer, 
              loss=loss_fn, 
              metrics=[CategoricalAccuracy()])
model.build(input_shape=(None, img_height, img_width, 3))
model.summary()

# 训练模型
epochs = 1  # 选择合适的epoch数
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
    )

# 修改batchsize
# train_generator = train_datagen.flow_from_directory(
#     dataset_path + '/train',   # 替换为数据集的路径
#     target_size=(img_height, img_width),
#     batch_size=1,
#     class_mode='categorical',
# )
# validation_generator = test_datagen.flow_from_directory(
#     dataset_path + '/test',
#     target_size=(img_height, img_width),
#     batch_size=1,
#     class_mode='categorical',
#     # subset='validation'
# )

# 创建模型
with tfmot.quantization.keras.quantize_scope(
    {'MBInvertedConvLayer': MBInvertedConvLayer, 
     'MBInvertedConvQuantizeConfig': MBInvertedConvQuantizeConfig}):
    annotated_model = model
    q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

# 编译模型
optimizer = Adam(learning_rate=0.001)
loss_fn = get_loss_fn(label_smoothing=0.1)
q_aware_model.compile(optimizer, loss=loss_fn, metrics=[CategoricalAccuracy()])

# 模型概要和训练
q_aware_model.summary()
print("==> training")
q_aware_model.fit(train_generator, 
                  epochs=1, 
                  validation_data=validation_generator)

# 使用数据生成器进行模型评估
print("==> evaluate")
test_loss, test_accuracy = q_aware_model.evaluate(validation_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# # 转化为TFLite
# print("--------------------------转为TFLite----------------------------")
# # 转换器
# converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)

# # 启用优化以应用量化
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # 进行转换
# tflite_quant_model = converter.convert()

# # 保存 TFLite 模型
# tflite_model_path = "/root/dacheng/Tensorflow-Train/output/best_model1110/quantized_model.tflite"
# with open(tflite_model_path, 'wb') as f:
#     f.write(tflite_quant_model)     