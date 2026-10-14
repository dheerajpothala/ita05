import os
import cv2
from common import load_image, show, OUTPUT_DIR

img = load_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Original shape:", img.shape)
print("Gray shape:", gray.shape)

cv2.imwrite(os.path.join(OUTPUT_DIR, "exp1_gray.jpg"), gray)
show([img, gray], ["Original", "Grayscale"], "exp1_result.png")
