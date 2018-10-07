import numpy as np
import cv2
import os
from tqdm import tqdm
from load_data import dataset_loader

# Style ideas
# http://www.windytan.com/2017/12/animated-line-drawings-with-opencv.html

f_h5 = "zero_data/RMSProp_zeros.h5"
#f_h5 = "zero_data/GradientDescent_zeros.h5"
#f_h5 = "zero_data/ADAM_zeros.h5"

# Set to unity to not blend (much faster)
is_blended_alpha = 1.0
line_color = [255, ] * 3

upscale = 1.5

image_args = {
    "f_h5" : f_h5,
    'total_frames':600,
    'trail_iterations':200,
    'width':int(512 * upscale),
    'height':int(512 * upscale),
    'extent_x':2.0,
    'extent_y':2.0,
}

M1 = dataset_loader(
    cutoff=4000,
    **image_args,
)

M2 = dataset_loader(
    cutoff=4000,
    offset=4000,
    **image_args,
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


def render_frame(M, k, background_color=[0, 0, 0]):

    X, Y = M[k]

    # Create a black image
    img = 255 * np.ones((M.width, M.height, 3), np.uint8)
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

# img = render_frame(M1, 0)
# cv2.imshow(f'image', img)
# cv2.waitKey(0)
# exit()



# Compute a frame schedule that moves slowly then quickly
N = len(M1)
x = np.sin(np.linspace(0,np.pi, N))
frame_n = np.round(N*np.cumsum(x)/x.sum()).astype(int)
frame_skip = 50
save_dest = "frames"
os.system(f'rm -rvf {save_dest} && mkdir -p {save_dest}')

ITR = frame_n[::frame_skip]

alpha = np.linspace(0, 1, len(ITR))

for i, k in enumerate(tqdm(ITR)):
    
    img = render_frame(M1, k)
    img2 = render_frame(M2, N - k)

    img = cv2.add(img, img2)
    f_png = os.path.join(save_dest, f'{i:04d}.png')
    cv2.imwrite(f_png, img)

    #cv2.imshow(f'image', img)
    #cv2.waitKey(1)

last_known_frame = i

for i, k in enumerate(tqdm(ITR)):
    
    img = render_frame(M2, k)
    img2 = render_frame(M1, N - k)

    img = cv2.add(img, img2)
    f_png = os.path.join(
        save_dest, f'{last_known_frame+i:04d}.png')
    cv2.imwrite(f_png, img)


cv2.destroyAllWindows()

cmd = f'ffmpeg -framerate 30 -i {save_dest}/%04d.png  -c:v libx264 -profile:v high -crf 10 -pix_fmt yuv420p {os.path.basename(f_h5)}.mp4'
print(cmd)
