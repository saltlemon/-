import tensorflow as tf

def myunet():
    #输入层
    inputs=tf.keras.layers.Input(shape=(128,128,3))
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)

    # 下采样
    x1 = tf.keras.layers.MaxPooling2D(padding="same")(x)  

    # 卷积 第二部分
    x1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x1) # 128*128*128

    # 下采样
    x2 = tf.keras.layers.MaxPooling2D(padding="same")(x1)  

    # 卷积 第三部分
    x2 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x2) # 64*64*256

    # 下采样
    x3 = tf.keras.layers.MaxPooling2D(padding="same")(x2)  

    # 卷积 第四部分
    x3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x3) # 32*32*512

    # 下采样
    x4 = tf.keras.layers.MaxPooling2D(padding="same")(x3)  
    # 卷积  第五部分
    x4 = tf.keras.layers.Conv2D(1024, 3, padding="same", activation="relu")(x4) # 16*16*1024


    ## unet网络结构上采样部分

    # 反卷积 第一部分      512个卷积核 卷积核大小2*2 跨度2 填充方式same 激活relu
    x5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x4)  # 32*32*512

    x6 = tf.concat([x3, x5], axis=-1)  # 合并 32*32*1024
    # 卷积
    x6 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x6)


    # 反卷积 第二部分
    x7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x6)  # 64*64*256

    x8 = tf.concat([x2, x7], axis=-1)  # 合并 64*64*512
    # 卷积
    x8 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x8)


    # 反卷积 第三部分
    x9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x8)  # 128*128*128

    x10 = tf.concat([x1, x9], axis=-1)  # 合并 128*128*256
    # 卷积
    x10 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x10)


    # 反卷积 第四部分
    x11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2,
                                          padding="same",
                                          activation="relu")(x10)  # 256*256*64

    x12 = tf.concat([x, x11], axis=-1)  # 合并 256*256*128
    # 卷积
    x12 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x12)


    # 输出层 第五部分
    output = tf.keras.layers.Conv2D(2, 1, padding="same", activation="softmax")(x12)  # 256*256*34

    return tf.keras.Model(inputs=inputs, outputs=output)

def my_full_unet():
    #输入层
    inputs=tf.keras.layers.Input(shape=(256,256,3))
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    # 下采样
    x1 = tf.keras.layers.MaxPooling2D(padding="same")(x)  # 128*128*64

    # 卷积 第二部分
    x1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x1)
    x1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x1)
    # 下采样
    x2 = tf.keras.layers.MaxPooling2D(padding="same")(x1)  # 64*64*128

    # 卷积 第三部分
    x2 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x2)
    x2 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x2)
    # 下采样
    x3 = tf.keras.layers.MaxPooling2D(padding="same")(x2)  # 32*32*256

    # 卷积 第四部分
    x3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x3)
    x3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x3)
    # 下采样
    x4 = tf.keras.layers.MaxPooling2D(padding="same")(x3)  # 16*16*512
    # 卷积  第五部分
    x4 = tf.keras.layers.Conv2D(1024, 3, padding="same", activation="relu")(x4)
    x4 = tf.keras.layers.Conv2D(1024, 3, padding="same", activation="relu")(x4)

    ## unet网络结构上采样部分

    # 反卷积 第一部分      512个卷积核 卷积核大小2*2 跨度2 填充方式same 激活relu
    x5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x4)  # 32*32*512

    x6 = tf.concat([x3, x5], axis=-1)  # 合并 32*32*1024
    # 卷积
    x6 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x6)
    x6 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x6)

    # 反卷积 第二部分
    x7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x6)  # 64*64*256

    x8 = tf.concat([x2, x7], axis=-1)  # 合并 64*64*512
    # 卷积
    x8 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x8)
    x8 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x8)


    # 反卷积 第三部分
    x9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x8)  # 128*128*128

    x10 = tf.concat([x1, x9], axis=-1)  # 合并 128*128*256
    # 卷积
    x10 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x10)
    x10 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x10)

    # 反卷积 第四部分
    x11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2,
                                          padding="same",
                                          activation="relu")(x10)  # 256*256*64

    x12 = tf.concat([x, x11], axis=-1)  # 合并 256*256*128
    # 卷积
    x12 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x12)
    x12 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x12)

    # 输出层 第五部分
    output = tf.keras.layers.Conv2D(2, 1, padding="same", activation="softmax")(x12)  # 256*256*34

    return tf.keras.Model(inputs=inputs, outputs=output)
