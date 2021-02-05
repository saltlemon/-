from color_transfer import color_transfer

import cv2


def show_image(title, image, width=300):
    # resize the image to have a constant width, just to

    # make displaying the images take up less screen real

    # estate

    r = width / float(image.shape[1])

    dim = (width, int(image.shape[0] * r))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # show the resized image

    cv2.imshow(title, resized)


source = cv2.imread('E:/river.jpg')

target = cv2.imread('E:/people.jpg')

# transfer the color distribution from the source image

# to the target image

transfer = color_transfer(source, target)

# check to see if the output image should be saved


# show the images and wait for a key press

show_image("Source", source)

show_image("Target", target)

show_image("Transfer", transfer)

cv2.waitKey(0)