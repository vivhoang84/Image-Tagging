import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import backend as K

# Streamlit header
st.header('Image Classification Model')

# Load the trained model
model = load_model('/Users/michelledang/Downloads/Image-Tagging/ImageTagging.keras')

# Categories corresponding to the classes the model can predict
data_cat = ['apple', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 
            'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 
            'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 
            'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 
            'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

img_height = 28
img_width = 28

image = st.text_input('Enter Image name', 'Apple.jpg')

# Load and preprocess the image
image_load = keras_image.load_img(image, target_size=(img_height, img_width), color_mode="grayscale")  
img_arr = keras_image.img_to_array(image_load) 
img_arr = np.expand_dims(img_arr, axis=0) 
img_arr = img_arr / 255.0  # Normalize the image 

# Flatten the image 
img_arr_flattened = img_arr.flatten().reshape(1, 784)


predict = model.predict(img_arr_flattened)

score = tf.nn.softmax(predict[0])  

# Display the result
st.image(image, width=200)  
st.write('Veg/Fruit in image is: ' + data_cat[np.argmax(score)])  
st.write('With accuracy of: ' + str(np.max(score) * 100) + '%')  