from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and preprocess the LFW dataset
lfw_people = fetch_lfw_people(data_home='/data/lfw_funneled', min_faces_per_person=20)
X = lfw_people.data
y = lfw_people.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Build the SVM model with PCA for dimensionality reduction
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced', probability=True)
model = make_pipeline(pca, svc)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
    img = img.resize((62, 47))  # Resize the image to match the LFW dataset dimensions

    # Convert the image to a numpy array
    img_array = np.array(img).reshape(1, -1)

    # Use your model for prediction
    prediction = model.predict(img_array)

    # Get the probability estimates for each class
    probabilities = model.predict_proba(img_array)
    max_probability = np.max(probabilities)  # Get the maximum probability
    percentage = round(max_probability * 100, 2)  # Calculate the percentage, rounded to 2 decimal places
    
    return render_template('index.html', prediction=str(lfw_people.target_names[prediction[0]]), percentage=str(percentage) + '%')

if __name__ == '__main__':
    app.run(debug=True)