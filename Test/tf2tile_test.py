import tensorflow as tf
import tensorflow_model_optimization as tfmot

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


# 加载训练好的量化模型，使用 quantize_scope
with tfmot.quantization.keras.quantize_scope():
    quantized_model = tf.keras.models.load_model(
    "/root/Tensorflow-Train/Test/DSConv_model",
    # custom_objects={'cross_entropy_with_label_smoothing': 'categorical_crossentropy'}
    )

# 转换为 TensorFlow Lite 模型
converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 保存转换后的模型
with open("DSConv_model.tflite", "wb") as f:
    f.write(tflite_model)
