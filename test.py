import tensorflow as tf
import json
from mymodule import *

''' 该文件用于打印模型结构
'''

# 加载配置文件
config_path = '/root/dacheng/Tensorflow-Train/net.config'
with open(config_path, 'r') as f:
    config = json.load(f)

# 创建输入
dummy_input = tf.random.normal([1, 32, 32, 3])

print(dummy_input)

# 创建模型
model = MyModel(config)

# 进行一次前向传播
_ = model(dummy_input)  

# 打印模型结构
# model.summary()  
