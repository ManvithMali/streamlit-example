import streamlit as streamlit
import pandas
import tensorflow as tf
import keras

export_dir='./'
#C:/Users/narae/Desktop/
newModel = tf.keras.models.load_model(export_dir)
#newModel=tf.saved_model.load(
#    export_dir, tags=None, options=None
#)
IM_WIDTH, IM_HEIGHT=198

st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(uploaded_file)

    
    im = image.resize((IM_WIDTH, IM_HEIGHT))

    im = np.array(im) / 255.0

    im = im.reshape(1,IM_WIDTH, IM_HEIGHT,3)
    print(im.shape)
    test_batch_size = 128
    #test_generator = data_generator.generate_images(test_idx, is_training=False, batch_size=test_batch_size)
    age_pred, gender_pred = newModel.predict(im)

    print(age_pred)
    print(gender_pred)
