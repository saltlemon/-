import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
def show(jpg_img,alpha):
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(jpg_img)
    plt.title("img")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(alpha)
    plt.title("Matting Alpha img")
    plt.axis("off")
    plt.show()

def generate(jpg_path,png_path,shape):
    x, y_ = [], []  # 建立空列表
    jpg_dirs = os.listdir(jpg_path) # 输出为所有文件名称
    for file in jpg_dirs:
        img_jpg_path=jpg_path + file
        jpg = cv2.imread(img_jpg_path, cv2.IMREAD_UNCHANGED)  # 根据路径读取jpg文件
        if jpg is None:
            continue
        jpg_img = cv2.cvtColor(jpg[:, :, :3], cv2.COLOR_BGR2RGB)  # imread读取的文件为bgr通道顺序，将通道顺序转变为rgb
        jpg_img = cv2.resize(jpg_img, shape)  # 调整图片大小
        jpg_img = jpg_img / 255.0  # 数据归一化
        x.append(jpg_img)  # 归一化后的数据，贴到列表x

    png_dirs = os.listdir(png_path)# 输出为所有文件名称
    for file in png_dirs:
        img_png_path = png_path + file
        png = cv2.imread(img_png_path, cv2.IMREAD_UNCHANGED)  # imread读取png图片

        if png is None:
            continue
        png = cv2.resize(png, shape)  # 调整图片大小
        alpha = png[:, :, 3]  # 其中alpha通道为第四通道

        ret, alpha = cv2.threshold(alpha, 150, 255, cv2.THRESH_BINARY)  # 二值化操作 阈值设置为150 255为数据最大值
        alpha = alpha / 255.0  # 数据归一化

        y_.append(alpha)

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_[:, :, :,np.newaxis]
    y_ = y_.astype(np.int64)
    return x,y_

def initial(jpg_path,png_path,shape):
    png_save_path = png_path + 'people_png_train.npy'
    jpg_save_path = jpg_path + 'people_jpg_train.npy'
    if os.path.exists(png_save_path) and os.path.exists(jpg_save_path):#判断数据文件是否存在
        print('-------------Load Datasets-----------------')
        print('jpg_path:'+ jpg_path)
        print('png_path:' + png_path)
        x = np.load(jpg_save_path)#读取jpg文件
        y_ = np.load(png_save_path)#读取png文件
        x = np.reshape(x, (len(x), 128, 128 ,3))#将数据转化为（num，128，128，3）
        y_ = np.reshape(y_, (len(y_), 128, 128, 1))#将数据转化为（num，128，128，1）
    else:
        print('-------------Generate Datasets-----------------')
        x, y_ = generate(jpg_path,png_path,shape)

        print('-------------Save Datasets-----------------')
        x_train_save = np.reshape(x, (len(x), -1))#将数据转化为二维数组
        y_train_save = np.reshape(y_, (len(y_), -1))#将数据转化为二维数组
        np.save(jpg_save_path, x_train_save)#保存数据
        np.save(png_save_path, y_train_save)#保存数据

    return x,y_

def initial_by_array(jpg_paths,png_paths,shape):
    x,y_=initial(jpg_paths[0],png_paths[0],shape)
    for i in range(1, len(jpg_paths)):
        temp_x,temp_y_=initial(jpg_paths[i],png_paths[i],shape)
        x=np.concatenate((x,temp_x),axis=0)#(num,128,128,3)=(num1,128,128,3)+(num2,128,128,3)
        y_=np.concatenate((y_,temp_y_),axis=0)#(num,128,128,1)=(num1,128,128,1)+(num2,128,128,1)
    return x,y_
print(123)