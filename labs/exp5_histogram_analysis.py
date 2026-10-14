import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from common import load_image, OUTPUT_DIR


def analyze_histogram(img, show_plot=True):
    channels = {"Blue": 0, "Green": 1, "Red": 2}
    plot_colors = {"Blue": "b", "Green": "g", "Red": "r"}
    results = {}

    print(f"{'Channel':<8}{'Mean':>8}{'Std':>8}{'Min':>6}{'Max':>6}{'Peak':>6}"
          f"{'Dark%':>8}{'Mid%':>8}{'Bright%':>9}")
    print("-" * 67)

    for name, idx in channels.items():
        channel = img[:, :, idx]
        hist = cv2.calcHist([img], [idx], None, [256], [0, 256]).ravel()
        total = hist.sum()

        stats = {
            "hist": hist,
            "mean": float(channel.mean()),
            "std": float(channel.std()),
            "min": int(channel.min()),
            "max": int(channel.max()),
            "peak": int(np.argmax(hist)),
            "dark": hist[:85].sum() / total * 100,
            "mid": hist[85:170].sum() / total * 100,
            "bright": hist[170:].sum() / total * 100,
        }
        results[name] = stats
        print(f"{name:<8}{stats['mean']:>8.1f}{stats['std']:>8.1f}{stats['min']:>6}"
              f"{stats['max']:>6}{stats['peak']:>6}{stats['dark']:>8.1f}"
              f"{stats['mid']:>8.1f}{stats['bright']:>9.1f}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_g, std_g = gray.mean(), gray.std()
    brightness = "dark" if mean_g < 85 else "bright" if mean_g > 170 else "medium brightness"
    contrast = "low contrast" if std_g < 40 else "high contrast" if std_g > 80 else "normal contrast"
    print(f"\nThe image is {brightness} with {contrast} (mean={mean_g:.1f}, std={std_g:.1f})")

    if show_plot:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Input image")
        ax[0].axis("off")

        for name in channels:
            ax[1].plot(results[name]["hist"], color=plot_colors[name], label=name)
        ax[1].set_title("Color histogram")
        ax[1].set_xlabel("Level")
        ax[1].set_ylabel("Pixels")
        ax[1].legend()

        ax[2].hist(gray.ravel(), 256, [0, 256], color="gray")
        ax[2].set_title("Gray histogram")
        ax[2].set_xlabel("Level")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "exp5_result.png"), dpi=120)
        plt.show()

    return results


if __name__ == "__main__":
    analyze_histogram(load_image())
