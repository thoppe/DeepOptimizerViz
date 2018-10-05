import numpy as np
import cv2
from tqdm import tqdm
from load_data import dataset_loader

# Style ideas
# http://www.windytan.com/2017/12/animated-line-drawings-with-opencv.html

#name = "RMSProp"
#name = "GradientDescent"
name = "ADAM"

# Set to unity to not blend (much faster)
is_blended_alpha = 1.0

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

def Y_histnorm(img, clipLimit=2.0, tileGridSize=(8,8)):

    # Convert to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Equalize the histogram of the Y channel
    clahe = cv2.createCLAHE(
        clipLimit=clipLimit, tileGridSize=tileGridSize)
    
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    # Convert back to BGR format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def render_frame(k):

    X, Y = M[k]

    # Create a black image
    img = 255*np.ones((width, height, 3), np.uint8)
    img[:, :, :] = background_color
    
    for lx, ly in zip(X, Y):
        pts = np.array([lx, ly], np.int32).T

        if is_blended_alpha != 1:
            img_blend = img.copy()
            cv2.polylines(img_blend, [pts], False, line_color, 1, cv2.LINE_AA)
            cv2.addWeighted(
                img_blend, is_blended_alpha, img, 1-is_blended_alpha, 0, img)

        else:
            cv2.polylines(img, [pts], False, line_color, 1, cv2.LINE_AA)


    #edges = cv2.Canny(img,100,400)
    

    '''
    mask = (np.linalg.norm(img, axis=2) > (255/2)).astype(np.uint8).reshape(width, height, -1)
    print(mask)
    img *= mask
    return mask
    
    #kernel = np.ones((2,2),np.uint8)
    #erosion = cv2.dilate(img,kernel,iterations = 1)
    
    return erosion
    print(erosion)
    exit()
    #background=(np.ones((width, height, 3))*background_color).astype(np.uint8)
    
    #cv2.addWeighted(
    #    background, 0.8, img, 0.85, 0, img)
    '''


    img = Y_histnorm(img)
    
    
    if upscale != 1:
        img = cv2.resize(img, (0,0), fx=1/upscale, fy=1/upscale)



    return img

# Problem with frames < 500

img = render_frame(1240)
cv2.imshow(f'image', img)
cv2.waitKey(0)
exit()

k = 1210
for k in tqdm(range(600, 2000, 50)):
    img = render_frame(k)
    cv2.imshow(f'image',img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
