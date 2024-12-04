import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 
from PIL import Image

st.header('Fruit and Vegetable Image Classification')
model = load_model('C:/Users/vivia/School/CS4200_AI/Image-Tagging/Image_classify.keras')
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
img_height = 180
img_width = 180
#image =st.text_input('Enter Image name','Apple.jpg')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image file using PIL Image
    image = Image.open(uploaded_file)
    
    # Resize the image to the target size expected by the model
    image = image.resize((img_width, img_height))

    # Convert the image to a numpy array and normalize it
    img_arr = tf.keras.utils.array_to_img(image)
    img_bat=tf.expand_dims(img_arr,0)

    predict = model.predict(img_bat)

    score = tf.nn.softmax(predict)
    st.image(image, width=200)
    st.write('Veg/Fruit in image: ' + data_cat[np.argmax(score)])
    st.write('With accuracy of ' + str(np.max(score)*100))
else:
    st.write("Please upload an image to classify.")