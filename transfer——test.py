from predict import *
peo_path='C:/Users/Lenovo/Desktop/test.jpg'
bg_path='C:/Users/Lenovo/Desktop/background.jpg'
peo = cv2.imread(peo_path, cv2.IMREAD_UNCHANGED)
bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
size = peo.shape
w = size[1]
h = size[0]
bg = cv2.resize(bg, (w*2,h))
I=change_bg(peo,bg)
I = cv2.cvtColor(I[:, :, :3], cv2.COLOR_RGB2BGR)
I=cv2.resize(I,(200,200))
cv2.imshow('img', I)
cv2.waitKey(0)
