import importlib
import os
import subprocess
import sys


def ensure_packages():
    required = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
    }
    for module, pip_name in required.items():
        try:
            importlib.import_module(module)
        except ImportError:
            print("Installing", pip_name)
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


ensure_packages()

import cv2
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import numpy as np  # pyright: ignore[reportMissingImports]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_sample_image():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    for x in range(600):
        img[:, x] = (40 + x // 4, 60 + x // 6, 120)
    cv2.rectangle(img, (60, 60), (240, 220), (0, 200, 255), -1)
    cv2.circle(img, (400, 150), 80, (50, 220, 50), -1)
    cv2.putText(img, "OpenCV", (180, 340), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return img


def load_image(path=None):
    if path is None:
        path = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    img = cv2.imread(path)
    if img is None:
        print("Could not read", path, "- using a sample image instead")
        return make_sample_image()
    print("Loaded", path, img.shape)
    return img


def show(images, titles, filename=None, cols=None):
    n = len(images)
    cols = cols or n
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (im, title) in enumerate(zip(images, titles), 1):
        plt.subplot(rows, cols, i)
        if im.ndim == 2:
            plt.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.show()
