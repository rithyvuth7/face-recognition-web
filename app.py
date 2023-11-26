import cv2
import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

model = load('data/model.pkl')
# people_names = pd.read_csv('data/people_names.csv')
# people_names = people_names.values
people_names = np.genfromtxt('data/people_names.csv', delimiter=',', dtype=str)
#print(people_names)
scaler = load('data/scaler.pkl')
pca = load('data/pca.pkl')




face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# def predict(image):
#     img = image
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     img = img.reshape(-1, 1)
#     faces = face_classifier.detectMultiScale(
#         img,     
#         scaleFactor=1.2,
#         minNeighbors=5,     
#         minSize=(20, 20)
#     )

#     for x,y,w,h in faces:
#         img = img[y:y+h,x:x+w]
        
#     if img.shape[1] > 0 and img.shape[0] > 0:
#         img = cv2.resize(img, (60,60), interpolation = cv2.INTER_AREA)

#         hist, bins = np.histogram(img.flatten(), 256, [0, 256])

#         cdf = hist.cumsum()
#         cdf_normalized = cdf * hist.max() / cdf.max(),
        
#         cdf_m = np.ma.masked_equal(cdf, 0)
#         cdf_m = (cdf_m - cdf_m.min())*255 / (cdf_m.max()-cdf_m.min())
#         cdf = np.ma.filled(cdf_m, 0).astype('uint8')

#         img = cdf[img]
#         img = img.reshape(60*60)

#     scaler_img = scaler.transform(img)
#     pc_img = pca.transform(scaler_img)
#     prediction = model.predict(pc_img)

#     #get the probability of the prediction
#     prob = model.predict_proba(pc_img)
#     prob = np.max(prob)
#     return prediction, prob
def predict(image):
    img = image
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_classifier.detectMultiScale(
        img_gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for x, y, w, h in faces:
        # Crop the face region
        face_region = img_gray[y:y+h, x:x+w]

        # Resize the cropped face to a fixed size
        face_resized = cv2.resize(face_region, (60, 60), interpolation=cv2.INTER_AREA)

        # Perform histogram equalization
        hist, bins = np.histogram(face_resized.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        face_equalized = cdf[face_resized].reshape(60 * 60)

        img = face_equalized

    img_reshaped = img.reshape(1, -1)
    #print(img_reshaped.shape)
    # code for scaling, PCA, and prediction
    scaler_img = scaler.transform(img_reshaped)
    pc_img = pca.transform(scaler_img)
    prediction = model.predict(pc_img)
    predict_name = people_names[prediction[0]]

    # Get the probability of the prediction
    prob = model.predict_proba(pc_img)
    prob = np.max(prob) * 100

    return predict_name, prob

def main():
    st.title("Face Recognition Web App")
    #st.sidebar.title("Options")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the PIL image to a NumPy array
        image_np = np.array(image)

        # Perform face recognition
        predict_name, proba = predict(image_np)

        # Display the results
        st.write("Classifying...")
        st.write(f"Prediction: {predict_name}")
        st.write(f"Probability: {proba}")
        
        #st.image(image_np, caption="Faces Recognized", use_column_width=True)

if __name__ == "__main__":
    main()