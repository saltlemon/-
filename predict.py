from unet import *
from transfer import *
from color import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
OUTPUT_CHANNELS=2
shape=(256,256)
model = unet_model(OUTPUT_CHANNELS,shape)
checkpoint_save_path = "./checkpoint/256x256.ckpt"
model.load_weights(checkpoint_save_path)

print(shape[0])
def image_stats(image):
    (r, g, b) = cv2.split(image)
    (rMean,gMean,bMean)=(r.mean(),g.mean(),b.mean())
    return (rMean, gMean, bMean)
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
def change_bg(jpg,bg,x_offset=0,y_offset=0):
    size = jpg.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    bg = cv2.cvtColor(bg[:, :, :3], cv2.COLOR_BGR2RGB)
    bg=bg[0:h,x_offset:x_offset+w,0:3]#通过切片获得我们想要的当前背景
    #print(bg.shape)
    jpg = cv2.cvtColor(jpg[:, :, :3], cv2.COLOR_BGR2RGB)
    jpg_img = cv2.resize(jpg, (shape[0],shape[1]))  # 调整图片大小
    jpg_img = jpg_img / 255.0  # 数据归一化
    pred_img = jpg_img[tf.newaxis, ...]
    #print('start pridct')
    pred_mask = np.squeeze(model.predict(pred_img)) #去除条目为1的维度
    #print('done pridict')
    pred_mask = tf.argmax(pred_mask, axis=2)#得到预测结果较大的条目
    pred_mask = np.array(pred_mask,dtype=np.uint8)#将结果从tf.tensor转为np的数组模式,以便后面使用cv2里面的函数
    pred_mask = cv2.resize(pred_mask,(w,h))#将预测结果转化为原始图像大小，resize函数第二个参数为图像大小，先宽后高
    pred_mask = pred_mask * 255.0  # 将0，1转化为0，255
    pred_mask = cv2.blur(pred_mask, (10, 10))  # 做平均值滤波，使得抠图边缘平滑一些
    pred_mask = np.array(pred_mask, dtype=np.float)
    pred_mask = pred_mask / 255.0  # 将0，255转化为0，1
    pred_mask_jpg = np.zeros_like(jpg)  #将单通道的mask转为3通道的mask，其中每个通道都为原始mask，这样是为了后面使用图像乘法
    pred_mask_jpg = np.array(pred_mask_jpg, dtype=np.float)
    pred_mask_jpg[:, :, 0] = pred_mask
    pred_mask_jpg[:, :, 1] = pred_mask
    pred_mask_jpg[:, :, 2] = pred_mask
    bg_mask_jpg = np.ones_like(jpg) #获得3通道的纯数字1矩阵
    bg_mask_jpg = np.array(bg_mask_jpg, dtype=np.float)
    bg_mask_jpg = cv2.subtract(bg_mask_jpg,pred_mask_jpg)#与pred_mask相见，得到背景mask的3通道矩阵
    #将图像和mask图像相乘得到对应的图像后相加得到最终图像
    peo=jpg*pred_mask_jpg
    peo=np.array(peo,dtype=np.uint8)
    background=bg*bg_mask_jpg
    background = np.array(background, dtype=np.uint8)
    I=hsv_change(peo,background,jpg)
    peo=I*pred_mask_jpg
    peo = np.array(peo, dtype=np.uint8)
    I=cv2.add(peo,background)
    return I