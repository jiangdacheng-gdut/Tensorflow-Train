import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig

class MBInvertedConvLayer(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, **kwargs):
        super(MBInvertedConvLayer, self).__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        mid_channels = in_channels * expand_ratio

        self.conv1 = tf.keras.layers.Conv2D(mid_channels, 1, 1, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.001)
        # 注意：这里不再初始化 self.act1
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size, stride, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.001)
        # 注意：这里不再初始化 self.act2
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 1, 1, padding='same', use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.001)

    def call(self, inputs):
        x = self.conv1(inputs)
        print("After conv1:", x.shape)  # 打印形状以调试
        x = self.bn1(x)
        x = tf.keras.activations.relu(x, max_value=6)
        x = self.depthwise_conv(x)
        print("After depthwise_conv:", x.shape)  # 打印形状以调试
        x = self.bn2(x)
        x = tf.keras.activations.relu(x, max_value=6)
        x = self.conv2(x)
        print("After conv2:", x.shape)  # 打印形状以调试
        x = self.bn3(x)
        return x
    
    def get_config(self):
        config = super(MBInvertedConvLayer, self).get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio
        })
        return config

class MBInvertedConvQuantizeConfig(QuantizeConfig):

    def __init__(self):
        # 初始化 权重 量化器
        self.weight_quantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8, symmetric=True, narrow_range=False, per_axis=False)
        
        # 初始化 激活函数 量化器
        self.activation_quantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False)
        
    # 需要量化的权重
    def get_weights_and_quantizers(self, layer):
        weights_quantizers = []
        if layer.conv1.weights:
            weights_quantizers.append((layer.conv1.weights[0], self.weight_quantizer))
        if layer.conv2.weights:
            weights_quantizers.append((layer.conv2.weights[0], self.weight_quantizer))
        if layer.depthwise_conv.weights:
            weights_quantizers.append((layer.depthwise_conv.weights[0], self.weight_quantizer))
        return weights_quantizers

    def get_activations_and_quantizers(self, layer):
        # 返回要量化的激活及其量化器
        # return [(layer.act1, self.activation_quantizer),
        #         (layer.act2, self.activation_quantizer)]
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        # 设置量化后的权重
        layer.conv1.kernel = quantize_weights[0]
        layer.depthwise_conv.depthwise_kernel = quantize_weights[1]
        layer.conv2.kernel = quantize_weights[2]

    def set_quantize_activations(self, layer, quantize_activations):
        # 设置量化后的激活
        # layer.act1 = quantize_activations[0]
        # layer.act2 = quantize_activations[1]
        return []

    def get_output_quantizers(self, layer):
        # 输出的量化器
        return [self.activation_quantizer]

    def get_config(self):
        return {}

def build_model(img_height, img_width, config):
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    for block_config in config['blocks']:
        # 移除 'mid_channels' 如果存在
        if 'mid_channels' in block_config['mobile_inverted_conv']:
            del block_config['mobile_inverted_conv']['mid_channels']
        mobile_inverted_conv_config = {k: v for k, v in block_config['mobile_inverted_conv'].items() if k != 'layersname'}
        annotated_layer = tfmot.quantization.keras.quantize_annotate_layer(
            MBInvertedConvLayer(**mobile_inverted_conv_config), 
            quantize_config=MBInvertedConvQuantizeConfig()
        )
        x = annotated_layer(x)

    x = tf.keras.layers.Conv2D(1280, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=10, use_bias=True)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class ZeroLayer(tf.keras.layers.Layer):
    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def call(self, inputs):
        # 使用 tf.shape 获取运行时的实际张量维度
        shape = tf.shape(inputs)
        n, h, w, c = shape[0], shape[1] // self.stride, shape[2] // self.stride, shape[3]
        return tf.zeros((n, h, w, c), dtype=inputs.dtype)

class MobileInvertedResidualBlock(tf.keras.layers.Layer):
    def __init__(self, name, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()
        self.layer_name = name

        # Check if the layer is a ZeroLayer and handle accordingly
        if mobile_inverted_conv['layersname'] == 'ZeroLayer':
            self.mobile_inverted_conv = ZeroLayer(stride=mobile_inverted_conv['stride'])
        else:
            mobile_inverted_conv_filtered = {k: v for k, v in mobile_inverted_conv.items() if k != 'layersname'}
            self.mobile_inverted_conv = MBInvertedConvLayer(**mobile_inverted_conv_filtered)

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
            # if 'name' in block['mobile_inverted_conv']:
            #     block['mobile_inverted_conv'].pop('name')
            if 'mid_channels' in block['mobile_inverted_conv']:
                block['mobile_inverted_conv'].pop('mid_channels')

        # MobileInvertedResidualBlock 层
        for block_config in config['blocks']:
            # 如果是 ZeroLayer
            if block_config['mobile_inverted_conv']['layersname'] == 'ZeroLayer':
                block_layer = ZeroLayer(stride=block_config['mobile_inverted_conv']['stride'])
            else:
                # 移除 'layersname' 键
                mobile_inverted_conv_config = {k: v for k, v in block_config['mobile_inverted_conv'].items() if k != 'layersname'}
                block_layer = MBInvertedConvLayer(**mobile_inverted_conv_config)
        
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



