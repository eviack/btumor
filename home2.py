import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras.models import load_model, Model

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Function to load the model
@st.cache_resource
def load_local_model():
    model = load_model('btumor2.h5')
    return model

# Function to preprocess and predict
def predict_and_display(image_data, model):
    img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions)

    if class_idx == 0:
        prediction_statement = "No tumor detected."
    else:
        prediction_statement = "Tumor detected."

    confidence_statement = f"Confidence score: {confidence * 100:.2f}%"
    return img, prediction_statement, confidence_statement, class_idx, img_array

# Function to extract the inner VGG16 model
def extract_vgg16_inner_model(wrapped_model):
    for layer in wrapped_model.layers:
        if isinstance(layer, Model) and 'vgg16' in layer.name:
            return layer
    raise ValueError("No VGG16 model found within the wrapped model.")

# Function to generate GradCAM++ heatmap
def generate_gradcam_plusplus(model, img_array, class_idx, last_conv_layer_name):
    vgg16_model = extract_vgg16_inner_model(model)
    if last_conv_layer_name not in [layer.name for layer in vgg16_model.layers]:
        raise ValueError(f"No such layer: {last_conv_layer_name} in the VGG16 model.")
    
    gradcam = GradcamPlusPlus(vgg16_model, model_modifier=ReplaceToLinear(), clone=True)
    score = CategoricalScore(class_idx)
    cam = gradcam(score, img_array, penultimate_layer=last_conv_layer_name)
    return cam[0]

# Function to overlay heatmap on image
def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    return superimposed_img

# Main function
def main():
    st.markdown("## Brain Tumor Detection")
    st.caption('''
            **Upload images in the sidebar to see the report**
                
            * This model when detects a tumor it's correct  **93%** of the time.
            * Out of all the images that actually have a tumor, our model correctly identifies **90%** of them.
            ''')

    # Load the model
    model = load_local_model()

    # Sidebar for uploading multiple images
    st.sidebar.markdown("## Upload MRI Images")
    st.sidebar.caption("You can select *multiple* images")
    uploaded_files = st.sidebar.file_uploader("Choose MRI images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.write("### Results ")
        st.divider()
        for uploaded_file in uploaded_files:
            # Read and decode image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Predict and display
            img, prediction, confidence, class_idx, img_array = predict_and_display(image, model)
            st.write(f"**Prediction for {uploaded_file.name}:** {prediction} \n- {confidence}")

            if(prediction=='No tumor detected.'):
                st.success(prediction)
            else:
                st.error(prediction)

            # Generate GradCAM++ heatmap
            last_conv_layer_name = 'block5_conv3'
            heatmap = generate_gradcam_plusplus(model, img_array, class_idx, last_conv_layer_name)
            superimposed_img = overlay_heatmap_on_image(img, heatmap)

            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption='Original Image', use_column_width=True)
            with col2:
                st.image(superimposed_img, caption='GradCAM++ Overlay(Why does it think so)', use_column_width=True)
            
            st.write("---")
    else:
        st.write("Please upload images through the sidebar to see results.")




if __name__ == "__main__":
    main()
