import tensorflow as tf
import sys
sys.path.append("..")  # 添加上一级目录到sys.path
from load_model import loaded_model

# # 1. 加载你的模型
# model_path = '/root/dacheng/Tensorflow-Train/output/best_model'  # 请确保这是你模型的正确路径
# model = loaded_model(model_path)

# # 2. 使用TFLiteConverter转换模型
# converter = tf.lite.TFLiteConverter.from_saved_model('/root/dacheng/Tensorflow-Train/output/best_model')
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, 
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()

# # 3. 保存转换后的模型
# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)


model_path = '/root/dacheng/Tensorflow-Train/output/best_model'
loaded = loaded_model(model_path)
concrete_func = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# 设定输入形状
input_shape = [1, 224, 224, 3]
concrete_func.inputs[0].set_shape(input_shape)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, 
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# 保存转换后的模型
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

