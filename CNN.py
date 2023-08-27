import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    """
    Custom dataset class for loading and preprocessing images.
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Constructor.
        
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding labels.
            transform (callable, optional): Transformation to apply to images.
        """
        
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CNNModel(nn.Module):
    """
    Definition of the Convolutional Neural Network model.
    """
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(3, 3)
        
        self.fc1 = nn.Linear(128 * 12 * 12, 2048)   # Large intermediary layer
        self.fc2 = nn.Linear(2048, 1024)            # Another intermediary layer
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        
        x = x.view(-1, 128 * 12 * 12)
        x = nnF.leaky_relu(self.fc1(x))
        x = nnF.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(device, train_dataloader):
    """
    Training function.
    
    Args:
        device (torch.device): Device to train the model on.
        train_dataloader (DataLoader): DataLoader for the training dataset.
    """
    
    # Instantiate the model and move it to the device
    model = CNNModel().to(device)
    model.load_state_dict(torch.load("./models/trained_model.pth"))     # Comment out if you want to train new model
    model.train()
    
    scaler = GradScaler()
    
    criterion = nn.MSELoss()  # Mean Squared Error loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 8
    
    model_path = "./models/trained_model.pth"
    
    for epoch in range(num_epochs):
        running_loss = 0.0
    
        for batch_images, batch_labels in train_dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.float().to(device)
        
            optimizer.zero_grad()
        
            with autocast():
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels.view(-1, 1))
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            running_loss += loss.item()
            print(f"Current LOSS: {loss.item():.2f}, LOSS: {running_loss:.2f}")
    
        print(f"Epoch {epoch+1}, Loss: {(running_loss / len(train_dataloader)):.2f}")
        torch.save(model.state_dict(), model_path)

    
    print("Training Finished!")
    
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at {model_path}")

def test(device, test_dataloader):
    """
    Testing function.
    
    Args:
        device (torch.device): Device to test the model on.
        test_dataloader (DataLoader): DataLoader for the testing dataset.
    """
    
    # Load the trained model for testing
    loaded_model = CNNModel().to(device)
    loaded_model.load_state_dict(torch.load("./models/trained_model.pth"))      # Adjust which model you want to load
    loaded_model.eval()  # Set the model to evaluation mode

    # threshold = 1  # Adjust this threshold based on your use case
    
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_images, batch_labels in test_dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.float().to(device)

            outputs = loaded_model(batch_images)

            # Calculate absolute differences between predictions and labels
            differences = torch.abs(outputs - batch_labels.view(-1, 1))

            # Count correct predictions based on the threshold
            threshold = torch.where(batch_labels.view(-1, 1) * 0.05 > 1, batch_labels.view(-1, 1) * 0.05, 1)
            
            predictions = differences <= threshold
            correct_predictions += torch.sum(predictions).item()
            
            total_samples += batch_labels.size(0)

    accuracy = correct_predictions / total_samples * 100.0

    print(f"Accuracy: {accuracy:.2f}%")

def main():
    data_directory = "./data"
    
    transform = transforms.Compose([
        transforms.Resize((1000, 1000)),  # Resize the images to a consistent size
        transforms.ToTensor(),
    ])
    
    # Load image paths and labels from CSVq                                     
    data_df = pd.read_csv(os.path.join(data_directory, "image_data.csv"))
    image_paths = [os.path.join(data_directory, "output", img) for img in data_df.iloc[:, 0]]
    labels = data_df.iloc[:, 1].astype(int).tolist()
    
    # Manually split data into training and testing sets
    split_ratio = 0.9  # 80% for training, 20% for testing
    split_index = int(len(image_paths) * split_ratio)
    train_image_paths, test_image_paths = image_paths[:split_index], image_paths[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    
    # Create datasets for training and testing
    train_dataset = CustomDataset(image_paths=train_image_paths, labels=train_labels, transform=transform)
    test_dataset = CustomDataset(image_paths=test_image_paths, labels=test_labels, transform=transform)
    
    batch_size = 1  # Define Batch Size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    
    # Comment out whichever function you dont need to use
    #train(device, train_dataloader)
    test(device, test_dataloader)
    
if __name__ == '__main__':
    main()
