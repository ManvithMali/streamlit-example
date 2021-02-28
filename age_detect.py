import streamlit as st
import pandas
import tensorflow as tf
import keras
import numpy as np
from PIL import Image

export_dir='h5model.h5'
newModel = tf.keras.models.load_model(export_dir)


st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    
    im = image.resize((198, 198))

    im = np.array(im) / 255.0

    im = im.reshape(1,198, 198,3)
    print(im.shape)
    test_batch_size = 128
    #test_generator = data_generator.generate_images(test_idx, is_training=False, batch_size=test_batch_size)
    age_pred, gender_pred = newModel.predict(im)

    if gender_pred[0][0] > gender_pred[0][1] :
        st.write("female")
    else:
        st.write("male")
    
    st.write(age_pred)
