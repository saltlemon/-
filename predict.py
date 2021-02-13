from unet import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
OUTPUT_CHANNELS=2
model = unet_model(OUTPUT_CHANNELS)
checkpoint_save_path = "./checkpoint/network.ckpt"
model.load_weights(checkpoint_save_path)

def show_pridicr(jpg_img,predict_alpha):
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(jpg_img)
    plt.title("img")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(predict_alpha)
    plt.title("predict Alpha img")
    plt.axis("off")
    plt.show()

jpg_path='C:/Users/Lenovo/Desktop/test.jpg'
def show_ppp(jpg_path):
    jpg = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)  # 根据路径读取jpg文件
    jpg_img = cv2.cvtColor(jpg[:, :, :3], cv2.COLOR_BGR2RGB)  # imread读取的文件为bgr通道顺序，将通道顺序转变为rgb
    jpg_img = cv2.resize(jpg_img, (128,128))  # 调整图片大小
    jpg_img = jpg_img / 255.0  # 数据归一化
    pred_img = jpg_img[tf.newaxis, ...]
    pred_mask = np.squeeze(model.predict(pred_img))
    pred_mask = tf.argmax(pred_mask, axis=2)
    show_pridicr(jpg_img, pred_mask)

show_ppp(jpg_path)