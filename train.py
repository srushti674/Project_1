"""
train.py
Loads recorded simulator data, balances it, splits train/val, trains
the NVIDIA model with augmented batches, plots the training curves,
and saves the final model
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import save_model
import matplotlib.pyplot as plt

from utils import (
    load_driving_log,
    plot_steering_histogram,
    balance_data,
    load_image_steering,
    batch_generator,
)
from model import build_nvidia_model


# CONFIG - update these paths for your machine

CSV_PATH = r"C:\Users\kumud\OneDrive\Documents\self_driving\driving_log.csv"
IMG_DIR = r"C:\Users\kumud\OneDrive\Documents\self_driving\IMG"
MODEL_OUT = "model.h5"

NUM_BINS = 25
SAMPLES_PER_BIN = 100
BATCH_SIZE = 100
EPOCHS = 10
STEPS_PER_EPOCH = 150
VAL_STEPS = 100
LEARNING_RATE = 1e-3


# DATA
print("[INFO] loading driving log...")
data = load_driving_log(CSV_PATH)
print(f"[INFO] {len(data)} rows loaded")

print("[INFO] steering distribution before balancing:")
plot_steering_histogram(data, num_bins=NUM_BINS, samples_per_bin=SAMPLES_PER_BIN)

data = balance_data(data, num_bins=NUM_BINS, samples_per_bin=SAMPLES_PER_BIN)

print("[INFO] steering distribution after balancing:")
plot_steering_histogram(data, num_bins=NUM_BINS, samples_per_bin=SAMPLES_PER_BIN)

image_paths, steerings = load_image_steering(IMG_DIR, data)
print(f"[INFO] {len(image_paths)} image/steering pairs (center+left+right)")

X_train, X_test, y_train, y_test = train_test_split(
    image_paths, steerings, test_size=0.2, random_state=6
)
print(f"[INFO] train: {len(X_train)} | test: {len(X_test)}")


# MODEL
model = build_nvidia_model(learning_rate=LEARNING_RATE)
model.summary()


# TRAIN
H = model.fit(
    batch_generator(X_train, y_train, BATCH_SIZE, is_training=True),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=batch_generator(X_test, y_test, BATCH_SIZE, is_training=False),
    validation_steps=VAL_STEPS,
)


# EVALUATE
plt.plot(np.arange(0, EPOCHS), H.history['loss'], label='loss')
plt.plot(np.arange(0, EPOCHS), H.history['val_loss'], label='val loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()

save_model(model, MODEL_OUT)
print(f"[INFO] model saved to {MODEL_OUT}")
