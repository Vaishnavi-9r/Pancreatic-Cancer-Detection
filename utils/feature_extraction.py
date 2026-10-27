import numpy as np


def extract_features(image, mask):

    pancreas_pixels = image[mask > 0]

    volume = np.sum(mask)

    mean = np.mean(pancreas_pixels)

    std = np.std(pancreas_pixels)

    maximum = np.max(pancreas_pixels)

    minimum = np.min(pancreas_pixels)

    return {

        "Volume": float(volume),

        "Mean Intensity": float(mean),

        "Standard Deviation": float(std),

        "Maximum Intensity": float(maximum),

        "Minimum Intensity": float(minimum)

    }