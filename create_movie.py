import cv2
import joblib
from tqdm import tqdm
from glob import glob

name = "ADAM"
name = "GradientDescent"

f_video = f"{name}_animation.avi"
fps = 29.0
skip_frames = 1
fractional_size = 0.75

F_IMAGE = sorted(glob(f"images/{name}/*.png"))[:][::skip_frames]
first_frame = cv2.imread(F_IMAGE[0])

height, width, layers = first_frame.shape

small = cv2.resize(first_frame, (0,0),
                   fx=fractional_size, fy=fractional_size)
height, width, layers = small.shape

print(f"Image found with height {height} width {width}")

fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
video = cv2.VideoWriter(f_video, fourcc, fps, (width,height))

def process(f):
    img = cv2.imread(f)
    return cv2.resize(img, (0,0), fx=fractional_size, fy=fractional_size)

func = joblib.delayed(process)
with joblib.Parallel(1) as MP:
    images = MP(func(f) for f in tqdm(F_IMAGE))

for img in tqdm(images):
    video.write(img)

cv2.destroyAllWindows()
video.release()

print (f_video)
