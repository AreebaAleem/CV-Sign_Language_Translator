# main.py

import cv2
import torch
import numpy as np
from segmentation import segment_hand_sign
from feature_extraction import extract_features
from classification import classify_hand_sign

def main():
    # Load an input image (replace 'your_image.jpg' with the actual image file)
    image_path = '/Users/Dell XPS White/Desktop/MATLAB/CV-Sign_Language_Translator/segmented_hand.jpg'
    input_image = cv2.imread(image_path)

    # Step 1: Segment the hand sign using OpenCV
    segmented_image = segment_hand_sign(input_image)

    # Step 2: Extract features using the CNN
    features = extract_features(segmented_image)

    # Step 3: Classify the hand sign using the trained SVM
    predicted_class = classify_hand_sign(features)

    # Display the result
    print(f"Predicted hand sign class: {predicted_class}")

if __name__ == "__main__":
    main()
