from flask import Flask, request, render_template, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np
import tensorflow as tf
import cv2
import base64
from datetime import datetime
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

app = Flask(__name__)

# Predefined doctor credentials
AUTHORIZED_DOCTORS = {
    "dr_john": generate_password_hash("cardio@123"),
    "dr_smith": generate_password_hash("neuro@456")
}

# Load tumor detection models
DENSENET_MODEL = tf.keras.models.load_model("model/densenet121_model.keras")
XCEPTION_MODEL = tf.keras.models.load_model("model/xception_model.keras")

def predict_tumor(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    densenet_pred = DENSENET_MODEL.predict(img_array)[0][0]
    xception_pred = XCEPTION_MODEL.predict(img_array)[0][0]
    
    return {
        "DenseNet-121": {
            "Prediction": "Tumor" if densenet_pred > 0.5 else "No Tumor",
            "Confidence": float(densenet_pred)
        },
        "Xception": {
            "Prediction": "Tumor" if xception_pred > 0.5 else "No Tumor",
            "Confidence": float(xception_pred)
        }
    }

# Encryption/Decryption functions
def apply_median_filter(image): return cv2.medianBlur(image, 5)
def apply_mean_filter(image): return cv2.blur(image, (5,5))
def apply_maximum_filter(image): return cv2.dilate(image, np.ones((5,5), np.uint8))
def apply_minimum_filter(image): return cv2.erode(image, np.ones((5,5), np.uint8))
def negative_positive(image): return 255 - image
def color_shuffle(image): return cv2.merge(cv2.split(image)[::-1])
def block_smoothing(image, bs=8):
    h,w,_ = image.shape
    for i in range(0, h, bs):
        for j in range(0, w, bs):
            image[i:i+bs,j:j+bs] = np.mean(image[i:i+bs,j:j+bs], axis=(0,1))
    return image
def logistic_sine_encrypt(image):
    h,w,_ = image.shape
    x = np.zeros((h,w))
    u = 3.99
    x[0,0] = 0.5
    for i in range(1,h): x[i,0] = u*x[i-1,0]*(1-x[i-1,0])
    for j in range(1,w): x[:,j] = u*x[:,j-1]*(1-x[:,j-1])
    return image ^ (x[...,np.newaxis].repeat(3,2)*255).astype(np.uint8)
def aes_encrypt(image):
    cipher = AES.new(b'16byteaeskey1234', AES.MODE_CBC, b'1234567890123456')
    return base64.b64encode(cipher.encrypt(pad(image.tobytes(), AES.block_size))).decode()
def aes_decrypt(encrypted_b64):
    cipher = AES.new(b'16byteaeskey1234', AES.MODE_CBC, b'1234567890123456')
    decrypted = unpad(cipher.decrypt(base64.b64decode(encrypted_b64)), AES.block_size)
    return np.frombuffer(decrypted, dtype=np.uint8).reshape(256,256,3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encrypt', methods=['POST'])
def encrypt():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"enc_{timestamp}_{file.filename}"
    orig_path = os.path.join('static/uploads', filename)
    file.save(orig_path)
    
    predictions = predict_tumor(orig_path)
    
    orig_img = cv2.resize(cv2.imread(orig_path), (256,256))
    encrypted = {
        'Median Filter': apply_median_filter(orig_img.copy()),
        'Mean Filter': apply_mean_filter(orig_img.copy()),
        'Maximum Filter': apply_maximum_filter(orig_img.copy()),
        'Minimum Filter': apply_minimum_filter(orig_img.copy()),
        'Negative-Positive': negative_positive(orig_img.copy()),
        'Color Shuffling': color_shuffle(orig_img.copy()),
        'Block Smoothing': block_smoothing(orig_img.copy()),
        'Logistic-Sine': logistic_sine_encrypt(orig_img.copy()),
        'AES': aes_encrypt(orig_img)
    }
    
    enc_paths = {}
    for name, img in encrypted.items():
        if name == 'AES': continue
        fname = f"{name.replace(' ','_')}_{filename}"
        path = os.path.join('static/uploads', fname)
        cv2.imwrite(path, img)
        enc_paths[name] = fname
    
    return render_template('index.html',
                         original=filename,
                         encrypted=enc_paths,
                         aes_encrypted=encrypted['AES'],
                         predictions=predictions)

@app.route('/decrypt', methods=['POST'])
def decrypt():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not (username in AUTHORIZED_DOCTORS and 
            check_password_hash(AUTHORIZED_DOCTORS[username], password)):
        return render_template('index.html', auth_error=True)
    
    file = request.files['encrypted_image']
    enc_type = request.form['encryption_type']
    
    if not file or file.filename == '':
        return redirect(url_for('index'))
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"dec_{timestamp}_{file.filename}"
    enc_path = os.path.join('static/uploads', filename)
    file.save(enc_path)
    
    decrypted = None
    img = cv2.imread(enc_path) if enc_type != 'AES' else None
    
    try:
        if enc_type == 'Negative-Positive':
            decrypted = negative_positive(img)
        elif enc_type == 'Color Shuffling':
            decrypted = color_shuffle(img)
        elif enc_type == 'Logistic-Sine':
            decrypted = logistic_sine_encrypt(img)
        elif enc_type == 'AES':
            decrypted = aes_decrypt(request.form['aes_encrypted'])
    except Exception as e:
        print(f"Decryption error: {e}")
        return redirect(url_for('index'))
    
    dec_filename = f"decrypted_{filename}"
    dec_path = os.path.join('static/uploads', dec_filename)
    cv2.imwrite(dec_path, decrypted)
    
    predictions = predict_tumor(dec_path)
    
    return render_template('index.html',
                         decrypted=dec_filename,
                         enc_type=enc_type,
                         aes_encrypted=request.form.get('aes_encrypted', ''),
                         predictions=predictions)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
