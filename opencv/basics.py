from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
import os
import imutils

# * Open CV uses (B, G, R)

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image")
# args = vars(ap.parse_args())
# args["image"]

path = "PyImageSearch/opencv/"
image_path = path + 'data.jpg'

# Show and save plot. Show doesn't work on vscode though.
# -----------------------------------------------

def plt_imshow(title, image, path=path):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (h, w, c) = image.shape[:3]

    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()
    plt.savefig(path + title)
    # cv2.imwrite(path + "opencv" + title + ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

plt_imshow('hi', cv2.imread(image_path))

# Pixel values, add, subtract
# -----------------------------------------------

image = cv2.imread(image_path)
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
tl = image[0:cY, 0:cX]
tr = image[0:cY, cX:w]
br = image[cY:h, cX:w]
bl = image[cY:h, 0:cX]
plt_imshow("Top-Left Corner", tl)
plt_imshow("Top-Right Corner", tr)
plt_imshow("Bottom-Right Corner", br)
plt_imshow("Bottom-Left Corner", bl)

# cv2.add(image, np.ones(image.shape))
# cv2.subtract(image, (image))

# Draw lines, rectangles, circles, text
# -----------------------------------------------

# canvas = np.zeros((300, 300, 3), dtype="uint8")
# cv2.line(canvas, (startX, startY), (endX, endY), color)
# cv2.rectangle(canvas, (startX, startY), (endX, endY), color, thickness(-1 for fill))
# cv2.circle(canvas, (centerX, centerY), r, (b, g, r))
# cv2.putText(img, 'text', (locX, locY), cv2.FONT, 1, 255, thickness)



# Translate, rotate, resize, flipping, cropping
# https://theailearner.com/tag/cv2-warpaffine/
# -----------------------------------------------
    # Translate

    # shift the image 25 pixels to the right and 50 pixels down
    # M = np.float32([[1, 0, 25], [0, 1, 50]])
    # shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # input_pts = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    # output_pts = np.float32([[cols-1,0], [0,0], [cols-1,rows-1]])
    # M= cv2.getAffineTransform(input_pts , output_pts)
    # dst = cv2.warpAffine(img, M, (cols,rows))

    # shifted = imutils.translate(image, X, Y)

    # Rotation
    # -----------------------------------------------

    # M = cv2.getRotationMatrix2D((cX, cY), angle, 1)
    # out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # rotated = imutils.rotate(image, 180)
    # rotated = imutils.rotate_bound(image, -33)

    # Resize
    # -----------------------------------------------

    # dim = ratio of new / old
    # keeping ratio, width fixed
    # r = 50.0 / image.shape[0]
    # dim = (int(image.shape[1] * r), 50)
    # resized = cv2.resize(image, (dimX, dimY), interpolation=cv2.INTER_AREA)

    # resized = imutils.resize(image, width=100)


    # Flipping
    # -----------------------------------------------
    # num 0: vertical 1: horizontal
    # cv2.flip(image, num)


    # Clipping
    # -----------------------------------------------
    # body = image[90:450, 0:290]


# Bitwise, masking, channels
# -----------------------------------------------
    # Bitwise
    # -----------------------------------------------
    # cv2.bitwise_and(a, b)
    # cv2.bitwise_or(a, b)
    # cv2.bitwise_xor(a, b)
    # cv2.bitwise_not(a)

    # Masking
    # -----------------------------------------------
    # cv2.circle(mask, (145, 200), 100, 255, -1)
    # bitwise_and

# Splitting and merging channels
# -----------------------------------------------
image = cv2.imread(path + 'data.jpg')
(B, G, R) = cv2.split(image)
zeros = np.zeros(image.shape[:2], dtype="uint8")
plt_imshow("Red", cv2.merge([zeros, zeros, R]))
plt_imshow("Green", cv2.merge([zeros, G, zeros]))
plt_imshow("Blue", cv2.merge([B, zeros, zeros]))