import cv2
from processing_utils import *

#img = cv2.imread("Data/sudokueasy.jpg")
img = cv2.imread("Data/sudokubig.jpg")

#show_image(img)

#get_grid(img, show=True)
grid = get_grid(img, True)
print(grid)
