import os
import cv2
import matplotlib.pyplot as plt
from common import load_image, OUTPUT_DIR

img = load_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)

ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
equalized_color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imwrite(os.path.join(OUTPUT_DIR, "exp4_equalized_gray.jpg"), equalized)
cv2.imwrite(os.path.join(OUTPUT_DIR, "exp4_equalized_color.jpg"), equalized_color)

fig, ax = plt.subplots(2, 3, figsize=(15, 8))

ax[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
ax[0, 0].set_title("Original (gray)")
ax[0, 1].imshow(equalized, cmap="gray", vmin=0, vmax=255)
ax[0, 1].set_title("Equalized (gray)")
ax[0, 2].imshow(cv2.cvtColor(equalized_color, cv2.COLOR_BGR2RGB))
ax[0, 2].set_title("Equalized (color)")
for a in ax[0]:
    a.axis("off")

ax[1, 0].hist(gray.ravel(), 256, [0, 256], color="gray")
ax[1, 0].set_title("Original histogram")
ax[1, 1].hist(equalized.ravel(), 256, [0, 256], color="black")
ax[1, 1].set_title("Equalized histogram")

for data, label in [(gray, "original"), (equalized, "equalized")]:
    hist = cv2.calcHist([data], [0], None, [256], [0, 256]).ravel()
    ax[1, 2].plot(hist.cumsum() / hist.sum(), label=label)
ax[1, 2].set_title("Cumulative distribution")
ax[1, 2].legend()

for a in ax[1]:
    a.set_xlabel("Intensity")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exp4_result.png"), dpi=120)
plt.show()
