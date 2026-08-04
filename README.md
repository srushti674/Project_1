# Self-Driving Car Simulation with a CNN

This project trains a neural network to drive a car around a track in the
Udacity self-driving car simulator. The car has a front camera, and the
model's only job is to look at what the camera sees and decide how much
to turn the steering wheel — left, right, or straight. This is a classic
behavioural cloning setup: the model learns to imitate how a human
drove the track, purely from images.

Built for DPS920 (Computer Vision) - Final Project By Group 8 - Yash Akbhari, Kumudhini, Srushti Patel.

## How it works, in plain terms

1. **Drive the track yourself** in the simulator's Training Mode, while it
   records what the front-facing camera sees and what steering angle you
   used at that exact moment.
2. **Feed that data to a CNN** (an NVIDIA-style architecture built for
   exactly this task) so it learns the relationship between "what the
   road looks like" and "how much to steer."
3. **Let the trained model drive** in Autonomous Mode - it now predicts
   the steering angle itself, frame by frame, in real time.

## Project files

| File | What it does |
|---|---|
| `driving_log.csv` | The recording from Training Mode — one row per frame, with the camera image filenames and the steering angle/throttle/brake/speed at that moment. |
| `IMG/` | The actual camera images referenced in `driving_log.csv` (center, left, and right camera angles). |
| `utils.py` | Helper functions: loading the CSV, checking/balancing the steering angle distribution, image preprocessing (crop, color conversion, resize), data augmentation, and the batch generator used during training. |
| `model.py` | Defines the CNN architecture (based on NVIDIA's "End to End Learning for Self-Driving Cars" paper) that takes an image and outputs a single steering angle. |
| `train.py` | The main training script — loads the data, balances it, trains the model, and saves the result as `model.h5`. |
| `check_data.py` | A small standalone script for quickly checking the steering angle distribution in a CSV before committing to a full training run. |
| `model.h5` | The trained model — this is what actually drives the car. |
| `TestSimulation.py` | Connects the trained model to the simulator. Run this, then start Autonomous Mode in the simulator — the script receives each camera frame, predicts a steering angle, and sends it back to the simulator in real time. |

## Setup

1. Create a virtual environment and activate it.
2. Install the required packages using the provided requirements.txt file
3. Download the [self-driving car simulator](https://github.com/udacity/self-driving-car-sim)
   for your OS and extract it.

## Step 1 — Collect your own driving data

1. Launch the simulator, choose **Training Mode**.
2. Pick a track and drive a few laps — smooth steering (mouse works better
   than keyboard) and a mix of both directions (forward and reverse) gives
   more balanced data.
3. Click the record button, choose an output folder, and drive. This
   produces an `IMG` folder and a `driving_log.csv`.

## Step 2 — Check your data

Before training, it's worth checking that your steering angles are
actually varied (not just driving straight the whole time):

```
python check_data.py
```

This shows a histogram of steering angles before and after balancing.
A good dataset has a visible spike near 0° (straight driving is common)
but shouldn't be *overwhelmingly* just that - you want turns
represented too.

<img width="1800" height="1023" alt="ss_img2" src="https://github.com/user-attachments/assets/7aa68ce3-c500-4cd5-8740-86c067022547" />
<img width="1797" height="1023" alt="ss_img3" src="https://github.com/user-attachments/assets/885c8590-e374-4586-8168-6ab3736e61a5" />

## Step 3 — Train the model

Open `train.py` and update the paths at the top to point to your own
`driving_log.csv` and `IMG` folder:

```python
CSV_PATH = r"path\to\your\driving_log.csv"
IMG_DIR = r"path\to\your\IMG"
```

Then run:

```
python train.py
```

This will:
- Show the steering angle histogram before and after balancing
- Print the model architecture summary
- Train for several epochs, showing training/validation loss as it goes
- Plot the final loss curve
- Save the trained model as `model.h5`

<img width="1770" height="1413" alt="ss_img4" src="https://github.com/user-attachments/assets/cc961983-7214-4aad-986d-db1115e98519" />


## Step 4 — Test it in the simulator

```
python TestSimulation.py
```

Leave that running, then in the simulator choose **Autonomous Mode** and
pick the same track you trained on. The script will connect automatically
and start sending steering predictions — the car should drive itself.

## A note on the model architecture

<img width="1134" height="1512" alt="ss_img1" src="https://github.com/user-attachments/assets/20e14bda-9e5b-4cd3-95ed-bd86bcc40e45" />

The CNN follows the NVIDIA architecture: five convolutional layers for
extracting visual features from the road, followed by three fully
connected layers that turn those features into a single steering angle
prediction. Two dropout layers were added (not in the original NVIDIA
diagram) to help the model generalize better, since the training dataset
here is much smaller than NVIDIA's original.

