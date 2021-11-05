import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Erosion, Dilation, Opening, Clsoing, 
# Morphological gradient, Black hat, top hat (white hat)
path = os.getcwd() +'/opencv'
image_path = path + '/images/gauge.jpg'


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# erosion: cv2.erode
# dilation: cv2.dilate

for i in range(0, 4):
    morphed = cv2.dilate(image.copy(), None, iterations=i + 1)
    plt.imshow(morphed)
    plt.show()
    plt.savefig(path + "/display/" + str(i+1))




image = cv2.imread(path + '/images/gauge.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# opening: erosion and dilation --- cv2.MORPH_OPEN
# clsoing: dilation and erosion --- cv2.MORPH_CLOSE
# morphGradient: Outline of objects --- CV2.MORPH_GRADIENT
kernelSizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
for kernelSize in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    morphed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    plt.imshow(morphed)
    plt.show()
    plt.savefig(path + "/display2/" + "size_" + str(kernelSize[0]))




# Top Hat (White Hat): difference between original and opening
        # light against dark background
        # cv2.MORPH_TOPHAT
# Black Hat : black against light background
        # cv2.MORPH_BLACKHAT
image = cv2.imread(path + '/images/gauge.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, rectKernel)
plt.imshow(blackhat)
plt.show()
plt.savefig(path + "/display3/" + "hat")



# Smoothing and Blurring
# simple, weighted Gaussian, median, bilateral blurring

# cv2.blur(image, (kX, kY))
# cv2.GaussianBlur(image, (kX, ky), 0)
# cv2.medianBlur(image, k)
# cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)


## Color Spaces
# we wnat lighting conditions to be
# 1. High contrast
# 2. Generalizable
# 3. Stable
cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# Thresholding
(T, threshInv) = cv2.threshold(image, 200, 255,
	cv2.THRESH_BINARY_INV)
(T, thresh) = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
masked = cv2.bitwise_and(image, image, mask=threshInv)
(T, threshInv) = cv2.threshold(image, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


# Adaptive thresholding
