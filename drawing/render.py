import numpy as np
import cv2
from tqdm import tqdm
from load_data import dataset_loader

# Style ideas
# http://www.windytan.com/2017/12/animated-line-drawings-with-opencv.html

name = "RMSProp"
#name = "GradientDescent"
#name = "ADAM"

background_color = [200, 200, 250]
line_color = [255, 0, 0]

upscale = 1.5
width = int(512*upscale)
height = int(512*upscale)
trail_iterations = 200

M = dataset_loader(
    name,
    width=width,
    #cutoff=40000,
    cutoff=4000,
    total_frames=600,
    
    height=height,
    extent_x = 2.0,
    extent_y = 2.0,
    trail_iterations=trail_iterations,
)



def render_frame(k):

    X, Y = M[k]

    # Create a black image
    img = np.zeros((width, height, 3), np.uint8)
    img[:, :, :] = background_color

    for lx, ly in zip(X, Y):
        pts = np.array([lx, ly], np.int32).T
        cv2.polylines(img, [pts], False, line_color, 1, cv2.LINE_AA)

        #for (x0,x1),(y0,y1) in zip(zip(lx,lx[1:]), zip(ly,ly[1:])):
        #    cv2.line(img, (x0,y0), (x1,y1), line_color, 1, 20)

    #img = cv2.GaussianBlur(img, (1,1), 0) 
  
    if upscale != 1:
        img = cv2.resize(img, (0,0), fx=1/upscale, fy=1/upscale)

    return img

#img = render_frame(600)
#cv2.imshow(f'image',img)
#cv2.waitKey(0)
#exit()

k = 1210
for k in tqdm(range(400, 2000, 50)):
    img = render_frame(k)
    cv2.imshow(f'image',img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
