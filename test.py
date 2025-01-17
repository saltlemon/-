from unet import *
from dataset import *
down_feature_list = []
# save down feature

OUTPUT_CHANNELS = 2

model = unet_model(OUTPUT_CHANNELS,(256,256))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
train_png_paths = ['E:/people\matting/1803151818/matting_00000000/',
                   'E:/people\matting/1803151818/matting_00000001/',
                   ]
train_jpg_paths = ['E:/people\clip_img/1803151818/clip_00000000/',
                   'E:/people\clip_img/1803151818/clip_00000001/',
                   ]
x_train, y_train=initial_by_array(train_jpg_paths,train_png_paths,(256,256))
print("dataset:")
print(x_train.shape)
print(y_train.shape)

test_png_path = 'E:/people\matting/1803151818/matting_00000008/'
test_jpg_path = 'E:/people\clip_img/1803151818/clip_00000008/'
x_test, y_test=initial(test_jpg_path,test_png_path,(256,256))
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

checkpoint_save_path = "./checkpoint/ppp.ckpt"
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
