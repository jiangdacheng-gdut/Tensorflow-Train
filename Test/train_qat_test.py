import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Activation, MaxPooling2D, Lambda, Flatten, Concatenate, Input, Add, add, Dropout
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow_model_optimization.quantization.keras import quantize_model
from tensorflow.keras.callbacks import ModelCheckpoint

# 数据集配置
dataset_path = '/root/dacheng/proxylessnas/cv_ids_25'

# 数据预处理
batch_size = 32
img_height = 32  # 根据你的模型和数据集调整
img_width = 32   # 根据你的模型和数据集调整

# 验证集的生成器，进行了数据增强                                                                               
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # 旋转范围
    width_shift_range=0.2,  # 宽度偏移
    height_shift_range=0.2,  # 高度偏移
    horizontal_flip=True,  # 水平翻转
    zoom_range=0.2,  # 缩放范围
    brightness_range=[0.8,1.2]  # 亮度调整范围
)  
train_generator = train_datagen.flow_from_directory(
    dataset_path + '/train',   # 替换为数据集的路径
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
)

test_datagen = ImageDataGenerator(rescale=1./255)  # 测试集的生成器
validation_generator = test_datagen.flow_from_directory(
    dataset_path + '/test',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    # subset='validation'
)

# 加载数据集
def load_custom_dataset():
    train_dir = '/root/dacheng/proxylessnas/cv_ids_25/train'
    val_dir = '/root/dacheng/proxylessnas/cv_ids_25/test'
    img_height, img_width = 32, 32
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=None,
        label_mode='categorical',  # or 'binary' for binary classification
        subset=None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        validation_split=None,
        label_mode='categorical',  # or 'binary' for binary classification
        subset=None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds

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

# 定义CNN模型
def build_model(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        # 第一个卷积块
        Conv2D(4, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 第二个卷积块
        Conv2D(8, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 第三个卷积块
        Conv2D(16, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 全连接层
        Flatten(),
        Dense(16),
        Activation('relu'),
        Dense(num_classes),
        Activation('softmax')
    ])

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# 定义MobileNet模型
def build_custom_mobilenet(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        ###----------------主架构 MobileNet1----------------###
        # 拓展维度
        Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', input_shape=input_shape),
        BatchNormalization(),
        ReLU(max_value=6.0),

        # 深度卷积
        DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same'),
        BatchNormalization(),
        ReLU(max_value=6.0),

        # 逐点卷积
        Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same'),
        BatchNormalization(),

        ###----------------主架构 MobileNet2----------------###
        # 拓展维度
        Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same'),
        BatchNormalization(),
        ReLU(max_value=6.0),

        # 深度卷积        
        DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same'),
        BatchNormalization(),
        ReLU(max_value=6.0),

        # 逐点卷积
        Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same'),
        BatchNormalization(),

        ###----------------主架构 MobileNet3----------------###
        # 拓展维度
        Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same'),
        BatchNormalization(),
        ReLU(max_value=6.0),

        # 深度卷积        
        DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same'),
        BatchNormalization(),
        ReLU(max_value=6.0),

        # 逐点卷积
        Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same'),
        BatchNormalization(),

        ###--------------------特征混合器--------------------###
        Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        ReLU(max_value=6.0),
        GlobalAveragePooling2D(),
        ###----------------------分类器----------------------###
        Dense(num_classes)
    ])

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=get_loss_fn(),
                  metrics=['CategoricalAccuracy'])
    
    return model

def build_mobilenet(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    # 第一个3*3卷积
    x = Conv2D(filters=3, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 主架构 MobileNet1
    x = Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    # 主架构 MobileNet2
    # residual = x
    x = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    # x = Add()([residual, x])

    # 主架构 MobileNet3
    # residual = x
    x = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    # x = Add()([residual, x])

    # 特征混合器
    # x = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = ReLU(max_value=6.0)(x)
    # x = GlobalAveragePooling2D()(x)

    # 分类器
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',  # 假设使用分类交叉熵作为损失函数
                  metrics=['categorical_accuracy'])  # 假设使用分类准确率作为评估指标
    
    return model

# 实现channel_shuffle操作
def channel_shuffle(x, groups):
    """实现通道混洗"""
    batch_size, height, width, num_channels = x.shape
    channels_per_group = num_channels // groups
    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = layers.Permute((1, 2, 4, 3))(x)  # 交换group和channel_per_group的顺序
    x = tf.reshape(x, [-1, height, width, num_channels])
    return x

# 定义ShuffleNet单元
def shuffle_unit(inputs, in_channels, out_channels, strides=1, groups=1):
    """ShuffleNet的一个单元"""
    bottleneck_channels = out_channels // 4

    # 1x1 分组卷积（压缩）
    x = Conv2D(filters=bottleneck_channels, kernel_size=1, groups=groups, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = channel_shuffle(x, groups)

    # 3x3 深度可分离卷积
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # 1x1 分组卷积（扩张）
    x = Conv2D(filters=out_channels, kernel_size=1, groups=groups, use_bias=False)(x)
    x = BatchNormalization()(x)

    # 如果步长为1且输入输出通道数相同，则添加残差连接
    if strides == 1 and in_channels == out_channels:
        x = add([x, inputs])

    return x

# 定义ShuffleNet模型
def build_shufflenet(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # 初始卷积层
    x = Conv2D(filters=3, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # ShuffleNet单元
    x = shuffle_unit(x, in_channels=3, out_channels=8, strides=1, groups=1)
    x = shuffle_unit(x, in_channels=8, out_channels=8, strides=2, groups=1)
    x = shuffle_unit(x, in_channels=8, out_channels=16, strides=2, groups=1)

    # 全局平均池化和分类器
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x)

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['CategoricalAccuracy'])

    return model

# 定义DSConv层
def DSConvLayer(kernel_size, stride, channel, dropout_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))

    return model

# 定义DSConv模型
def DSConvModel(params, input_shape=(32, 32, 3), num_classes=10):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    for i in range(1, 6):
        layer_params = params.get(f'layer{i}', {})
        if layer_params.get('_name') == 'DSConv':
            kernel_size = layer_params['kernel_size']
            stride = layer_params['stride']
            channel = layer_params['channel']
            dropout_rate = layer_params['dropout_rate']
            model.add(DSConvLayer(kernel_size, stride, channel, dropout_rate))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params['hidden_size'], activation='relu'))
    model.add(tf.keras.layers.Dropout(params['dropout_rate']))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model

def build_dsconv_model(layer_params, hidden_size, dropout_rate, num_classes=10):
    """
    Build and return a DSConv model.

    Parameters
    ----------
    layer_params : list of dicts
        A list containing parameters for each layer.
    hidden_size : int
        Dimensionality of the last hidden layer.
    dropout_rate : float
        Dropout rate for the dropout layers.
    input_shape : tuple
        Shape of the input data.
    num_classes : int
        Number of classes for classification.
    """
    model = Sequential()

    # Add DSConv layers based on layer_params
    for layer_param in layer_params:
        if layer_param['_name'] == 'DSConv':
            model.add(DepthwiseConv2D(kernel_size=layer_param['kernel_size'], strides=layer_param['stride'], activation='relu', padding='same'))
            model.add(Conv2D(filters=layer_param['channel'], kernel_size=1, activation='relu'))
            if 'dropout_rate' in layer_param:
                model.add(Dropout(layer_param['dropout_rate']))

    model.add(Flatten())
    model.add(Dense(units=hidden_size, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model

# 创建普通模型实例
# model = build_model()
# model.summary()

# 创建量化模型实例
# quantized_model = build_quantized_model()
# quantized_model.summary()

# 创建MobileNet模型实例
# MobileNet_model = build_mobilenet()
# MobileNet_model.summary()

# 创建ShuffleNet模型实例
# ShuffleNet_model = build_shufflenet()
# ShuffleNet_model.summary()

# 创建DSConv模型实例
params = {
            "layer1": {
                "_name": "Empty"
            },
            "layer2": {
                "_name": "Empty"
            },
            "layer3": {
                "_name": "Empty"
            },
            "layer4": {
                "_name": "Empty"
            },
            "layer5": {
                "_name": "Empty"
            },
            "layer6": {
                "_name": "Empty"
            },
            "hidden_size": 10,
            "learning_rate": 0.01,
            "epochs": 5,
            "dropout_rate": 0.12968316349831058
        }

layer_params = [params[f"layer{i+1}"] for i in range(6)]  # 假设有9层

# DSConv_model = DSConvModel(params)

DSConv_model = build_dsconv_model(
        layer_params=layer_params,
        hidden_size=params['hidden_size'],
        dropout_rate=params['dropout_rate']
    )

# 编译模型
DSConv_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
# epochs = 3  # 选择合适的epoch数
# history = MobileNet_model.fit(
#     train_generator,
#     epochs=epochs,
#     validation_data=validation_generator
#     )

# 训练量化模型
# quantized_model = quantize_model(MobileNet_model)
# quantized_model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss=get_loss_fn(),
#                   metrics=['CategoricalAccuracy'])


epochs = params['epochs'] # 根据需要选择合适的epoch数

# checkpoint_callback = ModelCheckpoint(
#     'DSConv_model.h5',  # 保存模型的文件名
#     monitor='val_loss',          # 监控的指标是验证集损失
#     verbose=1,                   # 打印详细信息
#     save_best_only=True,         # 只保存最佳模型
#     mode='min'                   # 损失最小化时保存
# )

# 加载数据集
train_ds, val_ds = load_custom_dataset()

history = DSConv_model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    # callbacks=[checkpoint_callback]  # 加入回调
)

# 保存模型
DSConv_model.save("/root/Tensorflow-Train/Test/DSConv_model", save_format='tf')

# 保存模型
# ShuffleNet_model.save("ShuffleNet_model.h5")
