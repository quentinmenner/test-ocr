import boto3
import cv2
import json
import numpy as np
import os
import pytesseract
import sys
pytesseract.pytesseract.tesseract_cmd =r"C:\Users\quentinm\AppData\Local\Tesseract-OCR\tesseract.exe"

filename = 'cyrano-text.PNG'

print("starting..")
print(f"Key: {filename}")

img = cv2.imread(filename)
print('File imported !')
# img = cv2.imdecode(np.fromstring(file_content, np.uint8), cv2.IMREAD_COLOR)
# img = align_images(img, 'documents/skew-test-droite.png')
# img = resize_image(img, 1, 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('Shape of image : ', img.shape)
# clahe
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img)
extract_text = pytesseract.image_to_string(img, lang='fra')
print('OCRed')

print(extract_text)

with open('ocr-test.txt', 'w') as f:
    f.write(extract_text)

print('results saved')