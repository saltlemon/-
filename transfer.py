import cv2
import numpy as np
def state(img):
    (h, s, v) = cv2.split(img)
    #获得各通道其中非零数的均值，应为我们使用的是抠图后的图像，直接使用mean()会导致很多0值进入计算
    h_mean = h.sum() / (h != 0).sum()
    s_mean = s.sum() / (s != 0).sum()
    v_mean = v.sum() / (v != 0).sum()
    return (h_mean,s_mean,v_mean)
def hsv_change(peo,bg,img):
    #将输入图片从RGB转化为hsv通道
    peo = cv2.cvtColor(peo[:, :, :3], cv2.COLOR_RGB2HSV)
    (peo_h_mean, peo_s_mean, peo_v_mean) = state(peo)
    bg = cv2.cvtColor(bg[:, :, :3], cv2.COLOR_RGB2HSV)
    (bg_h_mean, bg_s_mean, bg_v_mean) = state(bg)
    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2HSV)
    (h, s, v) = cv2.split(img)  # 色调，饱和度，亮度
    alpha = 0.02#色调调整参数
    h = cv2.add(h,alpha * (bg_h_mean - peo_h_mean))
    h = h % 180#色调在opencv中的范围为0-180，改变色调后对180求模防止超出阈值
    bata=0.5#饱和度调整参数
    s = cv2.add(s, bata*(bg_s_mean - peo_s_mean))
    gama=0.3#亮度调整参数
    v = cv2.add(v, gama*(bg_v_mean - peo_v_mean))
    transfer = cv2.merge([h, s, v])
    transfer = cv2.cvtColor(transfer[:, :, :3], cv2.COLOR_HSV2RGB)
    return transfer
