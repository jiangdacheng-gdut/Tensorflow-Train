import tensorflow as tf

class MBInvertedConvLayer(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBInvertedConvLayer, self).__init__()
        self.expand_ratio = expand_ratio
        mid_channels = in_channels * expand_ratio
        self.conv1 = tf.keras.layers.Conv2D(mid_channels, 1, 1, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.001)
        self.act1 = tf.keras.layers.ReLU(max_value=6)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size, stride, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.001)
        self.act2 = tf.keras.layers.ReLU(max_value=6)
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 1, 1, padding='same', use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.001)
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        return x

class MobileInvertedResidualBlock(tf.keras.layers.Layer):
    def __init__(self, name, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()
        self.layer_name = name
        self.mobile_inverted_conv = MBInvertedConvLayer(**mobile_inverted_conv)
        if shortcut is not None:
            self.shortcut = tf.keras.layers.Layer()  # Identity layer
        else:
            self.shortcut = None
        
    def call(self, inputs):
        x = self.mobile_inverted_conv(inputs)
        if self.shortcut is not None:
            x = x + self.shortcut(inputs)
        return x

class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
      
        # 创建 first_conv 层
        self.first_conv = tf.keras.layers.Conv2D(
            filters=32,  # 输出通道数
            kernel_size=(3, 3),  # 卷积核大小
            strides=(2, 2),  # 步长
            padding='same',  # 填充方式，'same' 表示输入和输出具有相同的维度
            use_bias=False,  # 不使用偏置项，因为接下来会使用 BN 层
        )

        # 定义 first_conv 的 BatchNormalization 层
        self.first_conv_bn = tf.keras.layers.BatchNormalization()

        # 定义 first_conv 的激活函数层
        self.first_conv_activation = tf.keras.layers.Activation('relu6')
                
        # 去掉block中mobile_inverted_conv的name属性
        for block in config['blocks']:
            if 'name' in block['mobile_inverted_conv']:
                block['mobile_inverted_conv'].pop('name')
            if 'mid_channels' in block['mobile_inverted_conv']:
                block['mobile_inverted_conv'].pop('mid_channels')

        # 搭建MobileNet网络
        self.blocks = [MobileInvertedResidualBlock(**block) for block in config['blocks']]
        
        # 搭建feature_mix_layer层
        self.feature_mix_layer = tf.keras.layers.Conv2D(
            filters=1280,  # 输出通道数
            kernel_size=(1, 1),  # 卷积核大小
            strides=(1, 1),  # 步长
            padding='same',  # 填充方式，'same' 表示输入和输出具有相同的维度
            use_bias=False,  # 不使用偏置项，因为接下来会使用 BN 层
        )

        # 定义 feature_mix_layer 的 BatchNormalization 层
        self.feature_mix_layer_bn = tf.keras.layers.BatchNormalization()

        # 定义 feature_mix_layer 的激活函数层
        self.feature_mix_layer_activation = tf.keras.layers.Activation('relu6')
        
        # 定义全局平均池化层
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()

        # self.classifier = tf.keras.layers.Dense(**config['classifier'])
        self.classifier = tf.keras.layers.Dense(units = 10, use_bias = True)
        
    def call(self, inputs):
        x = self.first_conv(inputs)
        x = self.first_conv_bn(x)
        x = self.first_conv_activation(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.feature_mix_layer_bn(x)
        x = self.feature_mix_layer_activation(x)
        x = self.global_avg_pooling(x)
        x = self.classifier(x)
        return x
