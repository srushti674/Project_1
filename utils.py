"""
utils.py
Data loading, preprocessing, and augmentation utilities for the Self-Driving Car behavioral cloning project.
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# 1. Loading the driving log
def load_driving_log(csv_path):
    """
    Loads driving_log.csv produced by the simulator.
    Columns: center, left, right, steering, throttle, brake, speed

    Returns a DataFrame with just the columns we need (center image path + steering angle), and also extracts just the filename (not the full absolute path) so the data is portable across machines.
    """
    columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    data = pd.read_csv(csv_path, names=columns)

    # Keep only filename, not full path (paths differ machine to machine)
    data["center"] = data["center"].apply(lambda x: os.path.basename(x.strip()))
    data["left"] = data["left"].apply(lambda x: os.path.basename(x.strip()))
    data["right"] = data["right"].apply(lambda x: os.path.basename(x.strip()))

    return data


# 2. Visualizing steering angle distribution
def plot_steering_histogram(data, num_bins=25, samples_per_bin=400, save_path=None):
    """
    Plots a histogram of steering angles to check dataset balance.
    Also returns the bin edges + counts so you can balance the data.
    """
    hist, bins = np.histogram(data["steering"], num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5

    plt.figure(figsize=(8, 4))
    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(data["steering"]), np.max(data["steering"])),
             (samples_per_bin, samples_per_bin), "r-")
    plt.title("Steering Angle Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return hist, bins


def balance_data(data, num_bins=25, samples_per_bin=400):
    """
    Trims over-represented steering angle bins (usually near 0) down to
    samples_per_bin so the model doesn't just learn to drive straight.
    """
    hist, bins = np.histogram(data["steering"], num_bins)
    remove_indices = []

    for j in range(num_bins):
        bin_indices = []
        for i in range(len(data["steering"])):
            if bins[j] <= data["steering"][i] <= bins[j + 1]:
                bin_indices.append(i)
        bin_indices = shuffle(bin_indices)
        remove_indices.extend(bin_indices[samples_per_bin:])

    print(f"Removed {len(remove_indices)} over-represented samples")
    data = data.drop(data.index[remove_indices])
    data = data.reset_index(drop=True)
    print(f"Remaining samples: {len(data)}")
    return data


# 3. Building (image_path, steering) pairs, using center/left/right
def load_image_steering(img_dir, data, side_camera_correction=0.2):
    """
    Builds a flat list of (image_path, steering_angle) using center, left, and right camera images with a steering correction applied to the left/right images.
    """
    image_paths = []
    steerings = []

    for i in range(len(data)):
        row = data.iloc[i]
        steering = float(row["steering"])

        # center
        image_paths.append(os.path.join(img_dir, row["center"]))
        steerings.append(steering)

        # left -> steer more right (positive correction)
        image_paths.append(os.path.join(img_dir, row["left"]))
        steerings.append(steering + side_camera_correction)

        # right -> steer more left (negative correction)
        image_paths.append(os.path.join(img_dir, row["right"]))
        steerings.append(steering - side_camera_correction)

    return np.asarray(image_paths), np.asarray(steerings)


# 4. Augmentation (applied randomly, TRAINING SET ONLY)
def zoom(image):
    zoom_factor = np.random.uniform(1, 1.3)
    h, w = image.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    y0 = np.random.randint(0, h - new_h + 1)
    x0 = np.random.randint(0, w - new_w + 1)
    cropped = image[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(cropped, (w, h))


def pan(image):
    h, w = image.shape[:2]
    tx = np.random.uniform(-0.1, 0.1) * w
    ty = np.random.uniform(-0.1, 0.1) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h))


def adjust_brightness(image):
    factor = np.random.uniform(0.4, 1.3)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def flip(image, steering):
    image = cv2.flip(image, 1)
    steering = -steering
    return image, steering


def random_rotation(image, max_angle=5):
    h, w = image.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))


def augment_image(image, steering):
    """
    Applies a RANDOM SUBSET of augmentations, each with 50% chance,
    rather than all of them uniformly every time.
    """
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = adjust_brightness(image)
    if np.random.rand() < 0.5:
        image = random_rotation(image)
    if np.random.rand() < 0.5:
        image, steering = flip(image, steering)
    return image, steering


# 5. Preprocessing (applied to EVERY image, train + val + inference)
def preprocess(image):
    """
    Crop -> YUV -> Gaussian Blur -> Resize (200x66, NVIDIA input size) -> Normalize to [0, 1]
    Expects an RGB image, e.g. read via mpimg.imread or cv2 (converted to RGB).
    """
    # Crop off sky (top) and hood/dashboard (bottom)
    image = image[60:135, :, :]

    # Convert to YUV color space (matches NVIDIA paper)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # Slight blur to reduce noise
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Resize to NVIDIA model input size
    image = cv2.resize(image, (200, 66))

    # Normalize
    image = image / 255.0

    return image


# 6. Batch generator
def batch_generator(image_paths, steering_angles, batch_size, is_training):
    """
    Infinite generator that yields (X_batch, y_batch).
    Applies augmentation only when is_training=True.
    """
    while True:
        batch_images = []
        batch_steerings = []

        for _ in range(batch_size):
            idx = np.random.randint(0, len(image_paths))
            image = cv2.imread(image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            steering = steering_angles[idx]

            if is_training:
                image, steering = augment_image(image, steering)

            image = preprocess(image)
            batch_images.append(image)
            batch_steerings.append(steering)

        yield np.asarray(batch_images), np.asarray(batch_steerings)
