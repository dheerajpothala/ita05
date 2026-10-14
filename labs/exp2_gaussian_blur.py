import os
import cv2
from common import load_image, show, OUTPUT_DIR

img = load_image()

blur_small = cv2.GaussianBlur(img, (5, 5), 0)
blur_medium = cv2.GaussianBlur(img, (15, 15), 0)
blur_large = cv2.GaussianBlur(img, (35, 35), 0)

cv2.imwrite(os.path.join(OUTPUT_DIR, "exp2_blur.jpg"), blur_medium)
show(
    [img, blur_small, blur_medium, blur_large],
    ["Original", "Kernel 5x5", "Kernel 15x15", "Kernel 35x35"],
    "exp2_result.png",
)
