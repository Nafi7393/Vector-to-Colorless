import cv2
import os


def img_color_remover(files):
    for file in files:
        image = cv2.imread(f"input/{file}", cv2.IMREAD_UNCHANGED)
        try:
            trans_mask = image[:, :, 3] == 0
            image[trans_mask] = [255, 255, 255, 255]
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        except: pass
        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        invert = cv2.bitwise_not(grey_img)
        blur = cv2.GaussianBlur(invert, (21, 21), 0)
        inverted_blur = cv2.bitwise_not(blur)
        sketch = cv2.divide(grey_img, inverted_blur, scale=256.0)

        cv2.imwrite(f"output/{file}", sketch)


def get_question_files(folder_name="input"):
    filenames = next(os.walk(f"{folder_name}"), (None, None, []))[2]  # [] if no file
    return filenames


img_color_remover(get_question_files())














