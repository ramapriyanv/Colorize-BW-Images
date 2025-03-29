import os
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb

def preprocess_images(input_base_folder, output_base_folder, num_samples=8000):
    """
    Preprocess images: Convert to LAB color space.
    Save grayscale (L channel) and color targets (A/B channels).
    """
    # Create output folders
    gray_folder = os.path.join(output_base_folder, "grayscale_images")
    color_folder = os.path.join(output_base_folder, "color_targets")

    os.makedirs(gray_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)

    subsets = ["train_samples", "valid_samples", "test_samples"]
    count = 0

    for subset in subsets:
        subset_folder = os.path.join(input_base_folder, subset)
        if not os.path.exists(subset_folder):
            print(f"Warning: {subset_folder} does not exist. Skipping...")
            continue

        print(f"Processing {subset}...")
        for file in os.listdir(subset_folder):
            if count >= num_samples:
                break

            filepath = os.path.join(subset_folder, file)
            if os.path.isfile(filepath):
                # Read and resize image
                img = cv2.imread(filepath)
                img_resized = cv2.resize(img, (256, 256))

                # Convert to LAB color space
                img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
                L, A, B = cv2.split(img_lab)

                # Save grayscale image (L channel) and A/B channels
                gray_output_path = os.path.join(gray_folder, f"{count}.jpg")
                color_output_path = os.path.join(color_folder, f"{count}.npy")

                cv2.imwrite(gray_output_path, L)  # Save L channel
                np.save(color_output_path, np.dstack((A, B)))  # Save A/B channels

                count += 1

    print(f"Preprocessing complete! Processed {count} images.")

if __name__ == "__main__":
    input_base_folder = "D:/Colorize BW Images/datasets"
    output_base_folder = "D:/Colorize BW Images/preprocessed"
    preprocess_images(input_base_folder, output_base_folder)
