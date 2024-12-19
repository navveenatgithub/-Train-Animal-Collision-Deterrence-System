import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Function to load and preprocess image
def process_frame(frame):
    IMAGE_SIZE = (224, 224)
    
    # Define the transforms (same as used in training)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Convert frame from OpenCV BGR format to RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply the transformations
    image = transform(image)
    
    # Add batch dimension (batch_size, channels, height, width)
    image = image.unsqueeze(0)
    
    return image

# Function to perform prediction
def predict_frame(frame, model, class_names):
    # Process the frame
    image = process_frame(frame)
    
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

# Function to capture webcam feed and display predictions
def webcam_feed(model, class_names):
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    #cap.set(cv2.CAP_PROP_FPS, 15)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict the class of the current frame
        predicted_class = predict_frame(frame, model, class_names)
        
        # Display the prediction on the frame
        cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow('Animal Detection', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage:

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

# Start webcam feed with predictions
webcam_feed(model, class_names)
