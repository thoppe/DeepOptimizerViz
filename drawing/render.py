import numpy as np
import cv2

width = 512
height = 512

# Create a black image
background_color = [200, 200, 250]
img = np.zeros((width, height, 3), np.uint8)
img[:, :, :] = background_color

# Draw a diagonal blue line with thickness of 5 px
cv2.line(img, (0,0), (511,511), (255,0,0),5)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
