import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Load the MNIST model
model = load_model('minist.keras')  # Update the path to your model

def preprocess_image(image_data):
    # Convert to grayscale and normalize
    image_data = image_data[:, :, :3]  # Ignore alpha channel
    image = Image.fromarray(image_data.astype("uint8")).convert("L")  # Convert to grayscale

    # Thresholding to create a binary image (black and white)
    threshold = 200  # You can adjust this value
    image = image.point(lambda p: 255 if p > threshold else 0)

    # Center the digit in the image
    image = ImageOps.invert(image)  # Invert to get white background and black digits
    image = image.resize((28, 28))  # Resize to MNIST input shape
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():
    st.set_page_config(page_title="Predict Handwritten Number", page_icon=":pencil2:")
    st.title("Draw a Number to Predict")
    st.write("Draw clearly, filling the square space completely to get better results.")

    # Create a canvas component for free drawing
    drawing_mode = "freedraw"  # Set only to freedraw
    stroke_width = 25  # Default stroke width
    stroke_color = "#000000"  # Set pen color to black
    bg_color = "#FFFFFF"  # Set background color to white
    realtime_update = True  # Update in real-time

    canvas_result = st_canvas(
        fill_color=bg_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        height=280,  # Set height to make the canvas bigger
        width=280,   # Set width to make the canvas bigger
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Prediction button
    if st.button("Predict"):
        # Prepare the image for prediction
        if canvas_result.image_data is not None:
            image_data = canvas_result.image_data
            image = preprocess_image(image_data)  # Call the preprocess function

            # Make prediction
            prediction = model.predict(image)
            predicted_number = np.argmax(prediction, axis=1)[0]

            # Display the predicted number
            st.write(f"Predicted Number: {predicted_number}")

if __name__ == "__main__":
    main()
