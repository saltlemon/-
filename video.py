import cv2
from vidstab.VidStab import VidStab
from predict import *
def magic(videoPath,bg_path):
    stabilizer = VidStab()
    cap = cv2.VideoCapture(videoPath)
    fps = 30  # 保存视频的FPS，可以适当调整
    # 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    numFrame=0
    bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
    if cap.isOpened():
        while True:
            ret, img = cap.read()  # img 就是一帧图片
            if not ret: break  # 当获取完最后一帧就结束
            if numFrame==0:
                size = img.shape#得到原视频的大小
                w = size[1]
                h = size[0]
                videoWriter = cv2.VideoWriter('F:/5.avi', fourcc, fps, (w,h))#将输出视频大小设置为原视频
                bg = cv2.resize(bg, (w*2,h))#将背景大小设置，这一步按个人兴趣即可
            I=change_bg(img,bg,numFrame*2,0)#得到转换后的图像
            I=cv2.cvtColor(I, cv2.COLOR_RGB2BGR)#由于输出图像为RGB通道，要将其转化为GBR通道
            videoWriter.write(I)#写入文件
            numFrame=numFrame+1#帧数+1
            print('done the '+str(numFrame)+'frame')
    else:
        print('视频打开失败！')
    videoWriter.release()
videoPath='C:/Users/Lenovo/Desktop/test2.mp4'
bg_path='C:/Users/Lenovo/Desktop/bg.jpg'
magic(videoPath,bg_path)