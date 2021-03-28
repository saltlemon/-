import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

def unet_model(output_channels,shape):
    #使用了tensorflow_exampe中的MobileNetV2模型
    base_model = tf.keras.applications.MobileNetV2(input_shape=[shape[0], shape[1], 3], include_top=False)

    # 使用这些层的激活设置
    layer_names = [
        'block_1_expand_relu',  
        'block_3_expand_relu',  
        'block_6_expand_relu',  
        'block_13_expand_relu', 
        'block_16_project',  
    ]
    #获取我们的译码器中的神经层
    layers = [base_model.get_layer(name).output for name in layer_names]

    # 创建译码器模型
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False
    
    #创建上采样层，也就是我们的解码器部分
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]
    inputs = tf.keras.layers.Input(shape=[shape[0], shape[1], 3])
    x = inputs

     # 得到我们的译码器输出
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

  # 上采样部分，进行残差连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

  # 输出，得到(w，h，output_channels)的输出结果
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
