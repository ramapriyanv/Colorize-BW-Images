import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths
GRAY_PATH = "D:/Colorize BW Images/preprocessed/grayscale_images"
COLOR_PATH = "D:/Colorize BW Images/preprocessed/color_targets"
SAVE_PATH = "D:/Colorize BW Images/models/Fine_Tuned_Colorization_Model.h5"

# Load and preprocess data
def load_data(gray_path, color_path, num_samples=2000):
    X, Y = [], []
    gray_files = sorted(os.listdir(gray_path))[:num_samples]
    color_files = sorted(os.listdir(color_path))[:num_samples]

    for gray_file, color_file in zip(gray_files, color_files):
        gray_img = cv2.imread(os.path.join(gray_path, gray_file), cv2.IMREAD_GRAYSCALE)
        color_img = np.load(os.path.join(color_path, color_file))

        if gray_img is not None and color_img is not None:
            gray_resized = cv2.resize(gray_img, (256, 256)).reshape(256, 256, 1) / 255.0
            color_resized = cv2.resize(color_img, (256, 256)) / 128.0  # Normalize color channels
            X.append(gray_resized)
            Y.append(color_resized)
    
    print(f"Loaded {len(X)} samples.")
    return np.array(X), np.array(Y)


# U-Net model with MobileNetV2 Encoder
# U-Net model with MobileNetV2 Encoder
def build_unet():
    base_model = MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze encoder weights

    # Input
    inputs = Input(shape=(256, 256, 1))
    x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])  # Duplicate grayscale input to 3 channels

    # Encoder
    encoder_output = base_model(x)

    # Decoder (upsampling layers to ensure output size is 256x256)
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(encoder_output)  # 8x8 -> 16x16
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)  # 16x16 -> 32x32
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)  # 32x32 -> 64x64
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x)  # 64x64 -> 128x128
    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x)  # 128x128 -> 256x256

    # Final output layer
    outputs = Conv2D(2, (1, 1), activation="tanh", padding="same")(x)  # 256x256x2

    # Compile model
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Train and save model
def train_model():
    X, Y = load_data(GRAY_PATH, COLOR_PATH)
    model = build_unet()
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(SAVE_PATH, monitor="val_loss", save_best_only=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # Train
    model.fit(X, Y, validation_split=0.1, epochs=5, batch_size=8, callbacks=[checkpoint, early_stopping])
    print(f"Model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train_model()
