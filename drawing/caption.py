from PIL import Image,ImageDraw,ImageFont
import numpy as np
import cv2

def caption(
        text,
        f_image,
        f_font,
        font_size=100,
        font_color = (255,75,255), # RGB
        orientation = None,
        x_padding = 0.01,
        y_padding = 0.01,
):
    font = ImageFont.truetype(f_font, font_size)
    img = cv2.imread(f_image)
    h, w, _ = img.shape
    
    # Measure the font
    tw,th = font.getsize(text)

    if orientation is None:
        x, y = 0, 0
    elif orientation=="bottom_left":
        x = int(x_padding*w)
        y = h - th - int(y_padding*h)

    print(w, h, tw, th)
    print(x,y)

    
    # Convert the image to RGB (OpenCV uses BGR)
    cv2_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    canvas = Image.fromarray(cv2_img_rgb)

    # Draw the text onto the text canvas
    draw = ImageDraw.Draw(canvas)
    draw.text((x,y), text, font_color, font)

    # Convert the image back to something CV2 likes
    img = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    return img

if __name__ == "__main__":
    f_img = "frames/0000.png"
    f_font = "design_src/leaguegothic-regular-webfont.ttf"
    #f_font = "design_src/Sniglet-webfont.ttf"

    img = caption("AASJHAS", f_img, f_font, orientation="bottom_left")
    cv2.imshow(f'image', img)
    cv2.waitKey(0)



