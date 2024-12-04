import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Load the model
st.header('Image Classification Model')
model = load_model('C:/Users/vivia/School/CS4200_AI/Image-Tagging/Image_classify.keras')

# Define the list of categories (ensure this matches your model's training labels)
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 
            'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 
            'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 
            'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Image size for the model
img_height = 180
img_width = 180

# Upload an image file using Streamlit's file_uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image file using PIL Image
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image', use_container_width=True)
    image = image.resize((img_width, img_height))
    img_arr = np.array(image)
    img_arr = img_arr / 255.0
    img_bat = np.expand_dims(img_arr, axis=0)
    
    # make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    predicted_class_idx = np.argmax(score)
    
    # Display the name and accuracy of the predicted class
    st.write('Veg/Fruit in image is: ' + data_cat[predicted_class_idx])
    
    st.write('With accuracy of: ' + str(np.max(score) * 100) + '%')

else:
    st.write("Please upload an image to classify.")
