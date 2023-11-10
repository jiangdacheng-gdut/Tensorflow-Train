import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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

def loss_fn():
    return get_loss_fn(label_smoothing=0.1)

def loaded_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'cross_entropy_with_label_smoothing': loss_fn})

def get_image(image_path):
    # 加载图片
    image = load_img(image_path, target_size=(224, 224))  # 例如，调整为224x224

    # 将图片转换为数组
    image_array = img_to_array(image)

    # 扩展维度以匹配模型的输入尺寸
    image_array = tf.expand_dims(image_array, axis=0)

    return(image_array) 

def predict_image(model_path, image_path):
    # 1. 加载模型
    model = loaded_model(model_path)
    
    # 2. 获取要推理的图片
    image_array = get_image(image_path)
    
    # 3. 使用模型的 predict 方法进行推理
    predictions = model.predict(image_array)
    
    # 4. 处理模型的输出（这里只是返回输出，你可能需要进一步处理，例如取得最高概率的类别等）
    return predictions

# 用例
model_path = "/root/dacheng/Tensorflow-Train/output/best_model"
image_path = "/root/dacheng/Tensorflow-Train/dataset/cv_ids_25/test/InsteonCamera/16-09-30-1610_15287.png"
result = predict_image(model_path, image_path)
print(result)