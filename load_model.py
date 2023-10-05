import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 指定模型的路径
model_path = '/root/dacheng/Tensorflow-Train/output/best_model'  # 替换为你的模型的实际路径

# 损失函数
def get_loss_fn(label_smoothing=0.1):
    def cross_entropy_with_label_smoothing(y_true, y_pred):
        # 获取类别数量
        num_classes = y_true.shape[-1]
        
        # 计算真实标签与平滑标签之间的权重
        y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
        
        # 计算交叉熵损失
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        
        return loss
    return cross_entropy_with_label_smoothing

# 加载模型
# loaded_model = tf.keras.models.load_model(model_path)
loss_fn = get_loss_fn(label_smoothing=0.1)
loaded_model = tf.keras.models.load_model(model_path, custom_objects={'cross_entropy_with_label_smoothing': loss_fn})

# 图片路径
image_path = '/root/dacheng/Tensorflow-Train/dataset/cv_ids_25/test/Attack/test-0072_18072.png'

# 加载图片
image = load_img(image_path, target_size=(224, 224))  # 例如，调整为224x224

# 将图片转换为数组
image_array = img_to_array(image)

# 扩展维度以匹配模型的输入尺寸
image_array = tf.expand_dims(image_array, axis=0)

# 使用模型进行预测
result = loaded_model.predict(image_array)

print(result)
