import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Vision | CNN Image Classifier", page_icon="🧠", layout="wide")

CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]
MODEL_PATH = "cnn_image_classifier.keras"

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

st.title("🧠 AI Vision")
st.subheader("Image Classification System Using Convolutional Neural Networks")
st.write("Upload an image and let the trained CNN predict which CIFAR-10 category it most closely resembles.")

with st.expander("About this project"):
    st.write(
        "This application uses a CNN trained on the CIFAR-10 dataset. "
        "The model achieved 70.84% test accuracy in the recorded evaluation. "
        "Predictions on everyday photos may be less reliable because CIFAR-10 images "
        "are small (32 × 32 pixels) and come from a specific dataset."
    )

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp"],
    help="For best results, choose a clear image of one object from a supported category."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        left, right = st.columns([1, 1])

        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Loading CNN model and predicting..."):
            model = load_cnn_model()
            resized_image = image.resize((32, 32))
            image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
            image_batch = np.expand_dims(image_array, axis=0)

            logits = model.predict(image_batch, verbose=0)
            probabilities = tf.nn.softmax(logits[0]).numpy()
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index]) * 100

        with right:
            st.success("Prediction complete!")
            st.metric("Predicted class", CLASS_NAMES[predicted_index])
            st.metric("Model confidence", f"{confidence:.2f}%")
            st.caption("Confidence is the model's score distribution, not a guarantee that the prediction is correct.")

        st.markdown("### Scores for all 10 classes")
        ranked_indices = np.argsort(probabilities)[::-1]
        for index in ranked_indices:
            score = float(probabilities[index])
            st.write(f"**{CLASS_NAMES[index]}** — {score * 100:.2f}%")
            st.progress(score)

    except Exception as error:
        st.error("The image could not be processed or the model could not be loaded.")
        st.exception(error)
else:
    st.info("Upload an image above to get a prediction.")

st.divider()
st.caption("Student AI project • TensorFlow/Keras • CIFAR-10 • CNN")
