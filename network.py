import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from dataset import *
down_feature_list = []
# save down feature

OUTPUT_CHANNELS = 2

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# 使用这些层的激活设置
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 创建特征提取模型
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # 在模型中降频取样
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # 升频取样然后建立跳跃连接
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # 这是模型的最后一层
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
train_png_paths = ['E:/people\matting/1803151818/matting_00000000/',
                   'E:/people\matting/1803151818/matting_00000001/',
                   'E:/people\matting/1803151818/matting_00000002/',
                   'E:/people\matting/1803151818/matting_00000003/',
                   'E:/people\matting/1803151818/matting_00000004/',
                   'E:/people\matting/1803151818/matting_00000005/',
                   'E:/people\matting/1803151818/matting_00000006/',
                   'E:/people\matting/1803151818/matting_00000007/']
train_jpg_paths = ['E:/people\clip_img/1803151818/clip_00000000/',
                   'E:/people\clip_img/1803151818/clip_00000001/',
                   'E:/people\clip_img/1803151818/clip_00000002/',
                   'E:/people\clip_img/1803151818/clip_00000003/',
                   'E:/people\clip_img/1803151818/clip_00000004/',
                   'E:/people\clip_img/1803151818/clip_00000005/',
                   'E:/people\clip_img/1803151818/clip_00000006/',
                   'E:/people\clip_img/1803151818/clip_00000007/']
x_train, y_train=initial_by_array(train_jpg_paths,train_png_paths,(128,128))
print("dataset:")
print(x_train.shape)
print(y_train.shape)

test_png_path = 'E:/people\matting/1803151818/matting_00000008/'
test_jpg_path = 'E:/people\clip_img/1803151818/clip_00000008/'
x_test, y_test=initial(test_jpg_path,test_png_path,(128,128))
print(x_test.shape)
print(x_test[1].shape)
print(y_test.shape)
print(y_test[1].shape)
def showing(jpg_img,alpha,predict_alpha):
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(jpg_img)
    plt.title("img")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(alpha)
    plt.title("Matting Alpha img")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(predict_alpha)
    plt.title("predict Alpha img")
    plt.axis("off")
    plt.show()
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]
def show_predictions( num=1):
      image=x_test[num]
      mask=np.squeeze(y_test[num])#删除数据中数据为1的维度
      pred_img=image[tf.newaxis, ...]
      pred_mask = np.squeeze(model.predict(pred_img))
      pred_mask = tf.argmax(pred_mask, axis=2)
      showing(image, mask, pred_mask)

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

checkpoint_save_path = "./checkpoint/network.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

model_history = model.fit(x_train,y_train, batch_size=32, epochs=13,
                          validation_data=(x_test, y_test),
                          validation_freq=1,
                          callbacks=[cp_callback])
show_predictions( )
model.summary()
