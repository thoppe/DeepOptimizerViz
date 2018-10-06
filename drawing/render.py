import numpy as np
import cv2
from tqdm import tqdm
from load_data import dataset_loader

# Style ideas
# http://www.windytan.com/2017/12/animated-line-drawings-with-opencv.html

# name = "zero_data/RMSProp.h5"
# name = "zero_data/GradientDescent.h5"
f_h5 = "zero_data/ADAM.h5"

# Set to unity to not blend (much faster)
is_blended_alpha = 1.0
line_color = [255, ] * 3

upscale = 1.5
width = int(512 * upscale)
height = int(512 * upscale)

M = dataset_loader(
    f_h5,
    # cutoff=40000,
    cutoff=4000,
    total_frames=600,
    trail_iterations=200,

    width=width,
    height=height,
    extent_x=2.0,
    extent_y=2.0,
)


def Y_histnorm(img, clipLimit=2.0, tileGridSize=(8, 8)):

    # Convert to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Equalize the histogram of the Y channel
    clahe = cv2.createCLAHE(
        clipLimit=clipLimit, tileGridSize=tileGridSize)

    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

    # Convert back to BGR format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def render_frame(k, background_color=[0, 0, 0]):

    X, Y = M[k]

    # Create a black image
    img = 255 * np.ones((width, height, 3), np.uint8)
    img[:, :, :] = background_color

    for lx, ly in zip(X, Y):
        pts = np.array([lx, ly], np.int32).T

        if is_blended_alpha != 1:
            img_blend = img.copy()
            cv2.polylines(img_blend, [pts], False, line_color, 1, cv2.LINE_AA)
            cv2.addWeighted(
                img_blend, is_blended_alpha, img, 1 - is_blended_alpha, 0, img)

        else:
            cv2.polylines(img, [pts], False, line_color, 1, cv2.LINE_AA)

    mask = np.average(img, axis=2) < 120

    kernal_size = 3
    blur = cv2.GaussianBlur(img, (kernal_size, kernal_size), 0)
    blur[mask] = img[mask]
    img = blur

    img = Y_histnorm(img)

    if upscale != 1:
        img = cv2.resize(img, (0, 0), fx=1 / upscale, fy=1 / upscale)

    return img

# img = render_frame(0)
# cv2.imshow(f'image', img)
# cv2.waitKey(0)
# exit()

for k in tqdm(range(0, len(M), 50)):
    img = render_frame(k)
    img2 = render_frame(len(M) - k)

    img = cv2.add(img, img2)

    cv2.imshow(f'image', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
