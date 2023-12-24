import cv2
import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = r'E:\\Tesseract\\tesseract.exe'

def preprocess_image(image_path, save_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Cannot open image at {image_path}")
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(save_path, image)
    return save_path

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Paths for saving images
preprocessed_directory = os.path.join(current_directory, 'PreprocessedImages')
os.makedirs(preprocessed_directory, exist_ok=True)

# Define the paths
image_path = os.path.join(current_directory, 'archive\\Bvtyyjq.png')
preprocessed_image_path = os.path.join(preprocessed_directory, 'preprocessed_image.png')

# Preprocess the image
preprocessed_image_path = preprocess_image(image_path, preprocessed_image_path)

# Open the preprocessed image with PIL
img = Image.open(preprocessed_image_path)

# Extract text using pytesseract
text = pytesseract.image_to_string(img, lang='eng')

print(text)
