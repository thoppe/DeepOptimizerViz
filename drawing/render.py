import numpy as np
import cv2
import os
from tqdm import tqdm
from load_data import dataset_loader

# Style ideas
# http://www.windytan.com/2017/12/animated-line-drawings-with-opencv.html

is_debug = True
is_debug = False

f_h5 = "zero_data/RMSProp_zeros.h5"
#f_h5 = "zero_data/GradientDescent_zeros.h5"
#f_h5 = "zero_data/ADAM_zeros.h5"
#f_h5 = "zero_data/FTRL_zeros.h5"

# Set to unity to not blend (much faster)
is_blended_alpha = 1.0

line_color0 = [206, 187, 134][::-1]
background_color = [0, 0, 0]

# Goldfish theme
#background_color = [224,228,204][::-1]
#line_color0 = [250,105,0][::-1]
#line_color1 = [105,210,231][::-1]

frame_skip = 250

upscale = 1.5
extent = 1.25
n_trails = 10000

image_args = {
    "f_h5" : f_h5,
    'total_frames':600,
    'trail_iterations':200,
    #'width':int(512 * upscale),
    #'height':int(512 * upscale),

    'width':int(1280 * upscale),
    'height':int(720 * upscale),
    
    'extent_x':extent,
    'extent_y':extent,
    'cutoff':n_trails,
}

image_args['extent_x'] *= image_args['width']/image_args['height']

M1 = dataset_loader(
    **image_args,
)

if not is_debug:
    M2 = dataset_loader(
        offset=10,
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


def render_frame(
        M, k, upscale=1.5,
        background_color=[0,0,0],
        line_color=[255,255,255],
        kernel_blur_size = 17,
):

    X, Y = M[k]
    print(len(X))

    # Create a black image
    img = 255 * np.ones((M.height, M.width, 3), np.uint8)
    img[:, :, :] = background_color

    for lx, ly in zip(X, Y):
        pts = np.array([lx, ly], np.int32).T

        if is_blended_alpha != 1:
            img_blend = img.copy()
            cv2.polylines(img_blend, [pts], False,
                              line_color, 1, cv2.LINE_AA)
            cv2.addWeighted(
                img_blend, is_blended_alpha,
                img, 1 - is_blended_alpha, 0, img)

        else:
            cv2.polylines(img, [pts], False,
                              line_color, 1, cv2.LINE_AA)

    dist = np.linalg.norm(img - line_color,axis=2)
    mask = dist < 50

    blur = cv2.GaussianBlur(img, (kernel_blur_size,)*2, 0)
    blur[mask] = img[mask]
    img = blur

    img = Y_histnorm(img)

    if upscale != 1:
        factor = 1/upscale
        img = cv2.resize(img, (0, 0), fx=factor, fy=factor)

    return img


'''
render_args = {
    "background_color": background_color,
    "line_color":line_color0,
}

if not is_debug:
    img = render_frame(M1, 0, **render_args)
    cv2.imshow(f'image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if is_debug:
    exit()
'''

    
# Compute a frame schedule that moves slowly then quickly
N = len(M1)
x = np.sin(np.linspace(0,np.pi, N))
#x = np.linspace(0,np.pi, N)

frame_n = np.round(N*np.cumsum(x)/x.sum()).astype(int)
save_dest = "frames"
os.system(f'rm -rvf {save_dest} && mkdir -p {save_dest}')

ITR = frame_n[::frame_skip]
alpha = np.linspace(0, 1, len(ITR))

render_args0 = {
    "background_color": background_color,
    "line_color":line_color0,
}


for i, k in enumerate(tqdm(ITR)):
    
    img = render_frame(M1, k, **render_args0)
    img2 = render_frame(M2, N - k, **render_args0)

    img = cv2.add(img, img2)
    
    f_png = os.path.join(save_dest, f'{i:04d}.png')
    cv2.imwrite(f_png, img)

    #cv2.imshow(f'image', img)

last_known_frame = i

for i, k in enumerate(tqdm(ITR)):
    
    img = render_frame(M2, k, **render_args0)
    img2 = render_frame(M1, N - k, **render_args0)

    img = cv2.add(img, img2)
    
    f_png = os.path.join(
        save_dest, f'{last_known_frame+i:04d}.png')
    cv2.imwrite(f_png, img)


cv2.destroyAllWindows()
f_mp4 = f'{os.path.basename(f_h5)}.mp4'

cmd = f'ffmpeg -y -framerate 30 -i {save_dest}/%04d.png  -c:v libx264 -profile:v high -crf 10 -pix_fmt yuv420p {f_mp4}'
print(cmd)
os.system(cmd)

cmd = f'xdg-open {f_mp4}'
os.system(cmd)
