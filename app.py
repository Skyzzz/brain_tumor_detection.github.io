from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load model
model = load_model('model/tumor_model.h5')

def predict_tumor(file_path):
    img = load_img(file_path, target_size=(150, 150))  # Sesuaikan dengan ukuran input model Anda
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return 'Tumor Terdeteksi' if prediction[0][0] > 0.5 else 'Tidak Ada Tumor'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            result = predict_tumor(file_path)
            return render_template('index.html', result=result, filename=file.filename)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
