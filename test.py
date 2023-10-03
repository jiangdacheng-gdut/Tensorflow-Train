import tensorflow as tf
import json
from module import *

''' 该文件用于打印模型结构
'''

# 加载配置文件
config_path = '/Users/jiangdacheng/Desktop/8-Coding/tensorflow-proxylessnas/net.config'
with open(config_path, 'r') as f:
    config = json.load(f)

# 创建输入
dummy_input = tf.random.normal([1, 224, 224, 3])

# 创建模型
model = MyModel(config)

# 进行一次前向传播
_ = model(dummy_input)  

# 打印模型结构
model.summary()  
