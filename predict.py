# Example code using PyTorch and torchvision
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Grayscale
from network import Net

def preprocess_image(image_path):
    # Define the same transform as used during training
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to the same size as training images
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize using the same values
    ])
    
    # Load the image
    image = Image.open(image_path)
    
    # Check if the image is not grayscale and convert it
    if image.mode != 'L':
        image = image.convert('L')
    
    # Apply the transformations
    image_tensor = transform(image)
    
    # Add a batch dimension (BxCxHxW) since PyTorch models expect batches of images
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def show_bbox(image, pred_bbox=None, pred_class=None):
    """Show image with bbox"""
    image = image.squeeze().numpy()

    plt.imshow(image, cmap='gray')
    imgsize = image.shape
    if pred_bbox is not None: 
        # plot the predicted bounding box (if provided)
        pred_bbox *= [imgsize[1],imgsize[0],imgsize[0],imgsize[1]]
        plt.plot([pred_bbox[0],pred_bbox[0]+pred_bbox[3],pred_bbox[0]+pred_bbox[3],pred_bbox[0],pred_bbox[0]],
                 [pred_bbox[1],pred_bbox[1],pred_bbox[1]+pred_bbox[2],pred_bbox[1]+pred_bbox[2],pred_bbox[1]], c='r')
        
    if pred_class is not None:
        plt.title("Predicted class: {}".format(pred_class))
        
    plt.show()


# Load pre-trained ResNet model
model = Net()
model.load_state_dict(torch.load('mnist-model.pth'))
model.eval()

image_path = 'ex1.jpg'
preprocessed_image = preprocess_image(image_path)
preprocessed_image = preprocessed_image.to('cpu')
# Make prediction
with torch.no_grad():
    predicted_bbox, predicted_cls = model(preprocessed_image)

print(predicted_bbox, predicted_cls)

predicted_bbox = predicted_bbox[0].cpu().numpy()
predicted_class = predicted_cls.data.argmax().item() 

print(predicted_bbox, predicted_class)

sample = {'image': preprocessed_image, 'pred_bbox': predicted_bbox, 'pred_class': predicted_class}
show_bbox(**sample)