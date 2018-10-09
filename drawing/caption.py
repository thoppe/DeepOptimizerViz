from PIL import Image,ImageDraw,ImageFont
import numpy as np
import cv2

def focus_blur(
        img,
        kernel_blur_size = 3,
        color=(255,255,255),
        threshold=50,
):
    dist = np.linalg.norm(img - color, axis=2)
    mask = dist < threshold

    blur = cv2.GaussianBlur(img, (kernel_blur_size,)*2, 0)
    blur[mask] = img[mask]
    img = blur
    return img

def caption(
        text,
        f_image,
        f_font,
        font_size=60,
        font_color = (255,255,255), # RGB
        alpha=0.25,
        blur=0,
        x_padding = 0.01,
        y_padding = 0.01,
        orientation = "bottom_left",
):
    font = ImageFont.truetype(f_font, font_size)
    img = cv2.imread(f_image)
    h, w, _ = img.shape
    
    # Measure the font
    tw,th = font.getsize(text)

    if orientation=="bottom_left":
        x = int(x_padding*w)
        y = h - th - int(y_padding*h)
    else:
        raise ValueError(f"Unknown orientation {orientation}")

    print(w, h, tw, th)
    print(x,y)
    
    # Convert the image to RGB (OpenCV uses BGR)
    canvas = Image.fromarray(np.zeros_like(img))

    # Draw the text onto the text canvas
    draw = ImageDraw.Draw(canvas)
    draw.text((x,y), text, font_color, font)

    # Convert the image back to something CV2 likes
    timg = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    if blur:
        timg = focus_blur(timg, blur)

    img = cv2.addWeighted(timg, 1-alpha, img, 1.0, 0)
    
    return img

if __name__ == "__main__":
    f_img = "frames/0000.png"

    f_font = "design_src/slkscr.ttf"

    img = caption("RMS Prop", f_img, f_font)
    cv2.imshow(f'image', img)
    cv2.waitKey(0)



