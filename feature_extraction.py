# feature_extraction.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load a pre-trained CNN model (e.g., ResNet or VGG)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.cnn = models.resnet18(pretrained=True)  # You can choose a different model

        # Remove the fully connected layers
        self.features = nn.Sequential(*list(self.cnn.children())[:-2])

    def forward(self, x):
        return self.features(x)

# Load an input image and extract features
def extract_features(image_path):
    # Load the image
    image = Image.open(image_path)

    # Preprocess the image (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor()

    # Extract features
    with torch.no_grad():
        features = feature_extractor(image_tensor)

    return features

if __name__ == "__main__":
    input_image_path = "/Users/Dell XPS White/Desktop/MATLAB/CV-Sign_Language_Translator/segmented_hand.jpg"
    extracted_features = extract_features(input_image_path)
    print("Extracted features shape:", extracted_features.shape)
