import streamlit as st
import os
from datetime import datetime
from colorize import colorize_image

# Paths
BASE_DIR = "D:/Colorize BW Images/"
RESULTS_DIR = os.path.join(BASE_DIR, "results/")
INPUTS_DIR = os.path.join(RESULTS_DIR, "inputs/")
OUTPUTS_DIR = os.path.join(RESULTS_DIR, "outputs/")

# Ensure the results, inputs, and outputs folders exist
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Streamlit App Design
st.set_page_config(
    page_title="Image Colorizer",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Header
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 3em;
            color: #4CAF50;
        }
        .sub-title {
            text-align: center;
            color: #777;
            font-size: 1.2em;
            margin-top: -15px;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            margin-top: 40px;
            color: #aaa;
        }
    </style>
    <h1 class="main-title">Colorization of Black-and-White Images Using Fine-Tuned Deep Learning Models 🎨</h1>
    <p class="sub-title">Upload your black-and-white images to bring them to life with a deep learning model!</p>
    """,
    unsafe_allow_html=True
)

# Sidebar Information
st.sidebar.title("About")
st.sidebar.info(
    """
    **This tool uses a deep learning model to colorize black-and-white images.**  
    - Upload a grayscale image (JPG, PNG).  
    - Click 'Colorize Image' to see the magic!  
    - Download the colorized output.  
    """
)

st.sidebar.write("### Supported Formats")
st.sidebar.write("✅ JPG\n✅ PNG\n✅ JPEG")

# File Upload
uploaded_file = st.file_uploader(
    "Choose a grayscale image...",
    type=["jpg", "jpeg", "png"],
    help="Upload only JPG or PNG files."
)

# Main Content
if uploaded_file is not None:
    # Create file paths
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    input_image_name = f"input_{timestamp}_{uploaded_file.name}"
    output_image_name = f"colorized_{timestamp}_{uploaded_file.name}"

    input_image_path = os.path.join(INPUTS_DIR, input_image_name)
    output_image_path = os.path.join(OUTPUTS_DIR, output_image_name)

    # Save uploaded image to INPUTS folder
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Display uploaded image
    st.image(input_image_path, caption="Uploaded Grayscale Image", use_column_width=True)

    # Colorize Button
    if st.button("🎨 Colorize Image"):
        st.write("⏳ **Processing...** Please wait.")
        try:
            # Ensure the colorize function directly outputs the colorized image to the OUTPUTS folder
            colorized_image_path = colorize_image(input_image_path, output_image_path)

            st.success("✅ Image successfully colorized!")
            st.image(colorized_image_path, caption="Colorized Image", use_column_width=True)

            # Download Button
            with open(colorized_image_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Colorized Image",
                    data=file,
                    file_name=output_image_name,
                    mime="image/jpg"
                )

            st.info(f"Saved input image in: **{INPUTS_DIR}**")
            st.info(f"Saved colorized image in: **{OUTPUTS_DIR}**")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown(
    """
    <div class="footer">
        Powered by Deep Learning 🚀
    </div>
    """,
    unsafe_allow_html=True
)
