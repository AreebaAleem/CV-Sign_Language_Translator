import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt

# Function for RGB to Binary Image Conversion
def rgb_to_binary(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image

# Function for Skin Detection in YCbCr color space
def detect_skin(image):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycbcr_image, lower_skin, upper_skin)
    skin_extracted_image = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin_extracted_image

# Function for Edge Detection using Canny edge detector
def detect_edges(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# Function for Image Resize
def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

# Function for Feature Extraction using Local Binary Patterns (LBP)
def extract_lbp_features(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate Local Binary Pattern
    radius = 3
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    
    # Calculate histogram of LBP image
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist


# Load dataset from train folder
train_images_folder = '/Users/Dell XPS White/Desktop/MATLAB/CV-Sign_Language_Translator/train/images'
train_labels_folder = '/Users/Dell XPS White/Desktop/MATLAB/CV-Sign_Language_Translator/train/labels'

X_train = []
y_train = []

# Loop through the images and labels folders and load images along with their labels
for filename in os.listdir(train_images_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(train_images_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            X_train.append(image)

            # Assuming label filename corresponds to image filename
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(train_labels_folder, label_filename)
            with open(label_path, 'r') as label_file:
                label = label_file.read().strip()
                y_train.append(label)

# Placeholder for preprocessing each image in X_train
preprocessed_images_train = []
for image in X_train:
    # Preprocessing steps (rgb_to_binary, detect_skin, etc.)
    # Apply preprocessing functions as needed
    preprocessed_image = image  # Placeholder, replace with actual preprocessing steps
    preprocessed_images_train.append(preprocessed_image)

# Extract features
X_train_features = [extract_lbp_features(image) for image in preprocessed_images_train]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train_features, y_train, test_size=0.2, random_state=42)

# Classification
# Using Support Vector Machine (SVM) classifier as an example
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Evaluation
y_pred = svm_classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
