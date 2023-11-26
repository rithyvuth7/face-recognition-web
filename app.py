import cv2
import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from PIL import Image

model = load('model.joblib')
people_names = pd.read_csv('people_names.csv')







def predict(image):
    img = image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(
        img,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    for x,y,w,h in faces:
        img = img[y:y+h,x:x+w]
        
    if img.shape[1] > 0 and img.shape[0] > 0:
        img = cv2.resize(img, (60,60), interpolation = cv2.INTER_AREA)

        hist, bins = np.histogram(img.flatten(), 256, [0, 256])

        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max(),
        
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min())*255 / (cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        img = cdf[img]
        img = img.reshape(60*60)

    scaler_img = scaler.transform(img)
    pc_img = pca.transform(scaler_img)
    prediction = model.predict(pc_img)

    #get the probability of the prediction
    prob = model.predict_proba(pc_img)
    prob = np.max(prob)
    return prediction, prob
    
def main():
    st.title("Face Recognition Web App")
    st.sidebar.title("Options")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the PIL image to a NumPy array
        image_np = np.array(image)

        # Perform face recognition
        face_locations, face_encodings = predict(image_np)

        # Display the results
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
        
        st.image(image_np, caption="Faces Recognized", use_column_width=True)

if __name__ == "__main__":
    main()