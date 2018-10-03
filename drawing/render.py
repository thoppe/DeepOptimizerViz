import numpy as np
import cv2
from load_data import get_frame

# Style ideas
# http://www.windytan.com/2017/12/animated-line-drawings-with-opencv.html

background_color = [200, 200, 250]
line_color = [255, 0, 0]

width = 512
height = 512
channels = 3

extent_x = 2.0
extent_y = 2.0

# Create a black image
img = np.zeros((width, height, channels), np.uint8)
img[:, :, :] = background_color


X, Y = get_frame(1210)

X *= width/2.0
X /= extent_x
X += width/2

Y *= -height/2.0
Y /= extent_y
Y += height/2

for lx, ly in zip(X, Y):

    for (x0,x1),(y0,y1) in zip(zip(lx,lx[1:]), zip(ly,ly[1:])):
        cv2.line(img, (x0,y0), (x1,y1), line_color, 1, 20)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
