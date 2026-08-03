"""
model.py
Builds the NVIDIA End-to-End CNN architecture for predicting
steering angle from a front-camera image.

Reference: "End to End Learning for Self-Driving Cars" (NVIDIA, 2016)
Input: 66 x 200 x 3 (YUV, normalized)
Output: 1 (steering angle)
"""

from keras import Sequential, layers
from keras.optimizers import Adam


def build_nvidia_model(input_shape=(66, 200, 3), learning_rate=1e-3):

    # MODEL
    model = Sequential([
        # Convolutional feature extraction (matches Figure 7 in the assignment)
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='elu', input_shape=input_shape),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(64, (3, 3), activation='elu'),
        layers.Conv2D(64, (3, 3), activation='elu'),

        layers.Dropout(0.5),
        layers.Flatten(),

        # Fully connected layers
        layers.Dense(100, activation='elu'),
        layers.Dropout(0.5),
        layers.Dense(50, activation='elu'),
        layers.Dense(10, activation='elu'),

        # Output: single steering angle value (regression, no activation)
        layers.Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae'])

    return model


if __name__ == '__main__':
    model = build_nvidia_model()
    model.summary()