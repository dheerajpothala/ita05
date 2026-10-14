import os
import cv2
from common import load_image, show, OUTPUT_DIR

img = load_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = cv2.GaussianBlur(gray, (5, 5), 0)

edges_low = cv2.Canny(smooth, 50, 100)
edges_mid = cv2.Canny(smooth, 100, 200)
edges_high = cv2.Canny(smooth, 150, 300)

cv2.imwrite(os.path.join(OUTPUT_DIR, "exp3_canny.jpg"), edges_mid)
show(
    [img, edges_low, edges_mid, edges_high],
    ["Original", "Canny 50-100", "Canny 100-200", "Canny 150-300"],
    "exp3_result.png",
)
