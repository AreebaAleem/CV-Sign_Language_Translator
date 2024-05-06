# segmentation.py

import cv2
import numpy as np

def segment_hand_sign(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert to YCbCr color space
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Apply morphological operations (erosion and dilation) for noise reduction
    kernel = np.ones((5, 5), np.uint8)
    morph_image = cv2.morphologyEx(ycbcr_image[:, :, 0], cv2.MORPH_OPEN, kernel)

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(morph_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours and extract the largest one (hand region)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        hand_mask = np.zeros_like(binary_mask)
        cv2.drawContours(hand_mask, [largest_contour], -1, 255, -1)
    else:
        print("No hand contour found!")

    # Apply the mask to the original image
    segmented_hand = cv2.bitwise_and(image, image, mask=hand_mask)

    # Save the segmented hand sign
    cv2.imwrite("segmented_hand.jpg", segmented_hand)

if __name__ == "__main__":
    input_image_path = "/Users/Dell XPS White/Desktop/MATLAB/Data/Hello/Image_1713010124.0340066.jpg"
    segment_hand_sign(input_image_path)