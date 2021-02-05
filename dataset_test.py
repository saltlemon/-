from dataset import *
import os, sys

train_png_paths = ['E:/people\matting/1803151818/matting_00000000/',
                   'E:/people\matting/1803151818/matting_00000001/',
                   'E:/people\matting/1803151818/matting_00000002/',
                   'E:/people\matting/1803151818/matting_00000003/',
                   'E:/people\matting/1803151818/matting_00000004/']
train_jpg_paths = ['E:/people\clip_img/1803151818/clip_00000000/',
                   'E:/people\clip_img/1803151818/clip_00000001/',
                   'E:/people\clip_img/1803151818/clip_00000002/',
                   'E:/people\clip_img/1803151818/clip_00000003/',
                   'E:/people\clip_img/1803151818/clip_00000004/']
x_train, y_train=initial_by_array(train_jpg_paths,train_png_paths,(128,128))
print("x_trian:")
print(x_train.shape)
print(y_train.shape)
show(x_train[634],np.squeeze(y_train[634]))

