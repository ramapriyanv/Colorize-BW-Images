# Black and White Image Colorizer

This project is a deep learning-based tool that colorizes grayscale images using a fine-tuned MobileNetV2 model. It provides a simple web interface built with Streamlit, allowing users to upload grayscale images, process them, and view or download the colorized results. The model is trained on the Places365 dataset to achieve high-quality and context-aware colorization.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download and Extract the Dataset:
Download Link: [Places365 Dataset](https://www.kaggle.com/datasets/pankajkumar2002/places365)

2. Extract the dataset into a folder named dataset.

3. Adjust the paths in the code to match your directory structure if needed. Comments are mentioned to add absolute paths.

4. Ensure the models folder is in the main directory.

## Preprocessing
Preprocess the dataset using the following command:

```bash
python src/preprocess.py
```

## Training
Fine-tune the MobileNetV2 model using your preprocessed dataset:

```bash
python src/train.py
```
Ensure colorize.py is present in the directory.

Run the Streamlit App
Start the Streamlit application for interactive colorization:

```bash
streamlit run src/app.py
```
## Usage
1. Run the Streamlit App using the command above.

2. Upload grayscale images in the web interface.

## License
Click "Colorize Image" to process the image.

View and download the colorized image from the interface.

Distribute under the MIT License. See [LICENSE](./LICENSE) for more information.
