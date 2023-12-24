from . import app
from flask import jsonify,request
from werkzeug.utils import secure_filename
from PIL import Image
import json
import os
import cv2
import shutil
import g4f
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'E:\\Tesseract\\tesseract.exe'

# def preprocess_image(image_path,save_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.GaussianBlur(image, (5, 5), 0)
#     image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
#     cv2.imwrite(save_path, image)
#     return image

@app.route('/process', methods=['POST'])
def text_processing():
    temp_directory = 'Photos'
    permanent_directory = 'SavedImages'

    # Ensure the directories exist
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    if not os.path.exists(permanent_directory):
        os.makedirs(permanent_directory)

    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(temp_directory, filename)
        file.save(temp_path)

        # Save the image to the permanent directory
        saved_image_path = os.path.join(permanent_directory, filename)

        # Simply copy the file instead of preprocessing
        shutil.copy(temp_path, saved_image_path)

        try:
            # Read the saved image using OpenCV
            img_cv = cv2.imread(saved_image_path, cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                raise ValueError("Could not read the image using OpenCV")

            # Extract text using pytesseract
            text = pytesseract.image_to_string(img_cv, lang='eng')
            print(text)
            return jsonify({'text': text}), 200
        except Exception as e:
            logging.exception("Error during processing")
            return jsonify({'error': str(e)}), 500
        finally:
            shutil.rmtree(temp_directory, ignore_errors=True)