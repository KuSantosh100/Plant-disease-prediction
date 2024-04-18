import streamlit as st
import tensorflow as tf
import numpy as np


def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                  'Tomato___healthy']

st.header("Disease Recognition 🌿🍀🔰")
st.markdown("""
            ### About Dataset
            This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
            This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 52 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
            A new directory containing 44 test images is created later for prediction purpose.
            ###
            """)

with st.expander("CONTENTS 🔖"):
    st.markdown("""
            1. train (77790 images)
            2. test (44 images)
            3. validation (18031 images)
            """)

with st.expander("DATASET LINKS 📚"):
    st.write("Kaggle Data Set 1 [link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data)")
    st.write("Kaggle Data Set 2 for Rice And Wheat [link](https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images)")

test_image = st.file_uploader("Choose an Image:")
if(st.button("Show Image")):
    st.image(test_image,width=4,use_column_width=True)
#Predict button
if(st.button("Predict")):
    st.write("Our Prediction")
    result_index = model_prediction(test_image)
    
    st.success("Model is Predicting it's a {}".format(class_name[result_index]))
    st.balloons()
    st.image(test_image,width=4,use_column_width=True)