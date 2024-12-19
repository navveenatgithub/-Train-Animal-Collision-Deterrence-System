from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt

# Function to load and preprocess image
def process_image(image_path):
    IMAGE_SIZE = (224, 224)
    
    # Define the transforms (same as used in training)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Apply the transformations
    image = transform(image)
    
    # Add batch dimension (batch_size, channels, height, width)
    image = image.unsqueeze(0)
    
    return image

# Function to perform prediction
def predict_image(image_path, model, class_names):
    # Process the image
    image = process_image(image_path)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move image to the same device as the model
    image = image.to(device)
    
    # Put the model in evaluation mode
    model.eval()
    
    # Perform the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    
    # Convert the predicted class index to class name
    predicted_class_name = class_names[predicted_class.item()]
    
    return predicted_class_name

# Function to display the image and prediction result
def display_image_and_prediction(image_path, predicted_class):
    # Open the image
    image = Image.open(image_path)
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.title(f'Predicted Class: {predicted_class}')
    plt.show()

# Example usage:
# Path to the image you want to predict
image_path = r"E:\IPTF\dataset\test\Bull\e4b15d99c00bf127.jpg"

# Load the trained model
model = models.vgg19(weights=None)  # Use weights=None for custom trained model
model.classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 6),  # Change to match your number of classes
    nn.LogSoftmax(dim=1)  # For multi-class classification
)

# Load the model state dictionary with weights_only=True for security
model.load_state_dict(torch.load(r"E:\IPTF\pytorch\MyModel.pth", weights_only=True))  # Correct path

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the actual class names in the same order as in your dataset
class_names = ['Bull', 'Cattle', 'Elephant', 'Horse', 'Lion', 'Tiger']

# Predict the image class
predicted_class = predict_image(image_path, model, class_names)

# Display the image with the predicted result
display_image_and_prediction(image_path, predicted_class)