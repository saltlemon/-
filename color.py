import cv2
import numpy as np

def image_stats(image):
    (r, g, b) = cv2.split(image)
    r_mean = r.sum() / (r != 0).sum()
    g_mean = g.sum() / (g != 0).sum()
    b_mean = b.sum() / (b != 0).sum()


    # return the color statistics
    return (r_mean,g_mean,b_mean)

def color_change(peo,bg,img,alpha,beta ):
    peo=peo.astype("float32")
    peo /= 255.0
    (peo_r_mean, peo_g_mean, peo_b_mean) = image_stats(peo)
    bg = bg.astype("float32")
    bg /= 255.0
    (bg_r_mean, bg_g_mean, bg_b_mean) = image_stats(bg)
    img = img.astype("float32")
    img /= 255.0
    (r, g, b) = cv2.split(img)
    r += alpha * (bg_r_mean - peo_r_mean)
    g += alpha * (bg_g_mean - peo_g_mean)
    b += alpha * (bg_b_mean - peo_b_mean)
    transfer = cv2.merge([r, g, b])

    (tr_r_mean, tr_g_mean, tr_b_mean) = image_stats(transfer)
    (img_r_mean, img_g_mean, img_b_mean) = image_stats(img)
    (tr_r, tr_g, tr_b) = cv2.split(img)
    tr_r += (img_r_mean - tr_r_mean)
    tr_g += (img_g_mean - tr_g_mean)
    tr_b += (img_b_mean - tr_b_mean)
    transfer = cv2.merge([tr_r, tr_g, tr_b])
    transfer*=beta
    transfer*=255.0
    transfer = np.clip(transfer, 0, 255)
    transfer=transfer.astype("uint8")

    return transfer



